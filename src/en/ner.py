"""NER model — BioBERT fine-tuned on BC5CDR for NUTRIENT and DISEASE detection.

CONTRACT: NERModel().predict(text) -> {"FOOD": list, "DISEASE": list, "NUTRIENT": list, "SYMPTOM": list}
Falls back to spaCy keyword lookup when BioBERT model is absent (pre-training).
"""

from __future__ import annotations

import os

import yaml

_CONFIG_PATH = "configs/config.yaml"
_MODEL_PATH_FALLBACK = "models/en/ner_bert"

# entity_group values from aggregation_strategy="simple" → project entity type
_LABEL_TO_TYPE: dict[str, str] = {
    "NUTRIENT": "NUTRIENT",
    "DISEASE":  "DISEASE",
    "FOOD":     "FOOD",
    "SYMPTOM":  "SYMPTOM",
}


def _ner_model_path() -> str:
    try:
        cfg = yaml.safe_load(open(_CONFIG_PATH))
        return cfg.get("ner_model_path", _MODEL_PATH_FALLBACK)
    except Exception:
        return _MODEL_PATH_FALLBACK


def _model_ready(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        device = self.transitions.device
        emissions = emissions.to(device)
        tags = tags.to(device)
        mask = mask.to(device)
        
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)
        score = self._compute_score(emissions, tags, mask)
        partition = self._compute_partition(emissions, mask)
        return torch.sum(partition - score)

    def _compute_score(self, emissions, tags, mask):
        seq_len, batch_size, _ = emissions.shape
        score = self.start_transitions[tags[0]] + emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_len):
            trans_score = self.transitions[tags[i], tags[i-1]]
            emit_score = emissions[i, torch.arange(batch_size), tags[i]]
            score = score + (trans_score + emit_score) * mask[i]
        last_valid_indices = mask.sum(dim=0) - 1
        last_tags = tags[last_valid_indices, torch.arange(batch_size)]
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_partition(self, emissions, mask):
        seq_len, batch_size, num_tags = emissions.shape
        alpha = self.start_transitions + emissions[0]
        for i in range(1, seq_len):
            alpha_expanded = alpha.unsqueeze(1)
            trans_expanded = self.transitions.unsqueeze(0)
            next_alpha = torch.logsumexp(alpha_expanded + trans_expanded, dim=2) + emissions[i]
            alpha = torch.where(mask[i].unsqueeze(1), next_alpha, alpha)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        device = self.transitions.device
        emissions = emissions.to(device)
        mask = mask.to(device)
        
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)
        seq_len, batch_size, num_tags = emissions.shape
        viterbi = self.start_transitions + emissions[0]
        backpointers = []
        for i in range(1, seq_len):
            max_viterbi, argmax_viterbi = torch.max(viterbi.unsqueeze(1) + self.transitions.unsqueeze(0), dim=2)
            viterbi = torch.where(mask[i].unsqueeze(1), max_viterbi + emissions[i], viterbi)
            backpointers.append(argmax_viterbi)
        viterbi = viterbi + self.end_transitions.unsqueeze(0)
        
        best_paths = []
        for b in range(batch_size):
            seq_l = mask[:, b].sum().item()
            if seq_l == 0:
                best_paths.append([])
                continue
            last_viterbi = viterbi[b]
            best_tag = torch.argmax(last_viterbi).item()
            path = [best_tag]
            for i in range(int(seq_l) - 2, -1, -1):
                best_tag = backpointers[i][b][best_tag].item()
                path.append(best_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths

class BertCRFForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            mask = (labels != -100) & (attention_mask == 1)
            clean_labels = torch.where(labels != -100, labels, torch.zeros_like(labels))
            loss = self.crf(emissions, clean_labels, mask)
        if not return_dict:
            output = (emissions,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=emissions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NERModel:
    """Token-classification NER.

    Uses BioBERT (fine-tuned on BC5CDR) when model weights are present;
    falls back to spaCy keyword matching otherwise.
    """

    def __init__(self):
        model_path = _ner_model_path()
        if _model_ready(model_path):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = BertCRFForTokenClassification.from_pretrained(model_path)
            self.model.eval()
            self._fallback: _SpacyFallback | None = None
        else:
            self.tokenizer = None
            self.model = None
            self._fallback = _SpacyFallback()

    def predict(self, text: str) -> dict[str, list[str]]:
        if self._fallback is not None:
            return self._fallback.predict(text)
        
        # Truncate text to a safe length (e.g. 150 words) to avoid exceeding BERT's 512 token limit
        words = text.split()
        if len(words) > 150:
            text = " ".join(words[:150])
            
        try:
            return self._run_manual_bert(text)
        except Exception:
            # Graceful fallback to spaCy keyword matcher if BERT model fails
            return _SpacyFallback().predict(text)

    def _run_manual_bert(self, text: str) -> dict[str, list[str]]:
        import torch
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )
        
        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}
        
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            
        logits = outputs.logits  # shape: (1, seq_len, num_tags)
        mask = (model_inputs["attention_mask"] == 1)
        
        # Run Viterbi decoding via CRF module
        decoded_paths = self.model.crf.decode(logits, mask)
        predictions = decoded_paths[0]
        
        offset_mapping = inputs["offset_mapping"][0].tolist()[:len(predictions)]
        
        entities = []
        current_ent = None
        id2label = self.model.config.id2label
        
        for pred_id, offset in zip(predictions, offset_mapping):
            start, end = offset
            if start == end:  # Skip special tokens like [CLS], [SEP]
                continue
                
            pred_label = id2label.get(pred_id, "O")
            if pred_label == "O":
                if current_ent is not None:
                    entities.append(current_ent)
                    current_ent = None
                continue
                
            if "-" in pred_label:
                bio_tag, ent_type = pred_label.split("-", 1)
            else:
                bio_tag = "B"
                ent_type = pred_label
                
            project_type = _LABEL_TO_TYPE.get(ent_type)
            if not project_type:
                if current_ent is not None:
                    entities.append(current_ent)
                    current_ent = None
                continue
                
            if current_ent is not None and current_ent["type"] == project_type:
                # Merge if the new token starts exactly where the current entity ends (subwords/no space)
                # OR if it has an 'I-' tag (standard BIO continuation across spaces)
                if start == current_ent["end"] or bio_tag == "I":
                    current_ent["end"] = end
                    continue
                
            if current_ent is not None:
                entities.append(current_ent)
            current_ent = {
                "type": project_type,
                "start": start,
                "end": end
            }
                    
        if current_ent is not None:
            entities.append(current_ent)
            
        result: dict[str, list[str]] = {"FOOD": [], "DISEASE": [], "NUTRIENT": [], "SYMPTOM": []}
        for ent in entities:
            etype = ent["type"]
            word = text[ent["start"]:ent["end"]].strip()
            if word and word not in result[etype]:
                result[etype].append(word)
        return result


# ---------------------------------------------------------------------------
# Fallback — spaCy keyword MVP, active until BioBERT training completes
# ---------------------------------------------------------------------------

_FOOD_TOKENS: set[str] = {
    "chicken", "rice", "beef", "pork", "fish", "salmon", "tuna", "shrimp",
    "broccoli", "spinach", "carrot", "tomato", "potato", "onion", "garlic",
    "egg", "milk", "cheese", "yogurt", "butter", "bread", "oat", "oatmeal",
    "apple", "banana", "orange", "grape", "strawberry", "blueberry",
    "almond", "walnut", "peanut", "tofu", "lentil", "bean", "quinoa",
}
_DISEASE_TOKENS: set[str] = {
    "diabetes", "hypertension", "obesity", "gout", "anemia", "arthritis",
    "cancer", "cholesterol", "osteoporosis", "asthma", "depression",
    "anxiety", "insomnia", "constipation", "diarrhea", "gastritis",
    "celiac", "ibd", "ibs", "ckd", "nafld",
}
_NUTRIENT_TOKENS: set[str] = {
    "protein", "carbohydrate", "fat", "fiber", "calorie", "vitamin",
    "mineral", "calcium", "iron", "sodium", "potassium", "magnesium",
    "zinc", "phosphorus", "folate", "omega-3", "omega-6", "antioxidant",
    "cholesterol", "glucose", "fructose", "lactose",
}
_SYMPTOM_TOKENS: set[str] = {
    "fatigue", "nausea", "headache", "dizziness", "bloating", "cramp",
    "insomnia", "weakness", "swelling", "inflammation", "pain", "fever",
}


class _SpacyFallback:
    def predict(self, text: str) -> dict[str, list[str]]:
        words = set(text.lower().split())
        return {
            "FOOD":     list(words & _FOOD_TOKENS),
            "DISEASE":  list(words & _DISEASE_TOKENS),
            "NUTRIENT": list(words & _NUTRIENT_TOKENS),
            "SYMPTOM":  list(words & _SYMPTOM_TOKENS),
        }
