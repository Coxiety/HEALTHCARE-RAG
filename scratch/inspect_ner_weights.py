import os
import sys
import torch
import pandas as pd

# Add the project root to the python path so we can import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.en.ner import NERModel

def main():
    print("Loading NER model...")
    ner = NERModel()
    
    if ner.model is None:
        print("Error: NER Model weights are not loaded. The fallback Spacy model is being used.")
        return
    
    model = ner.model
    id2label = model.config.id2label
    
    print("\n" + "="*50)
    print("1. MODEL ARCHITECTURE & SUMMARY")
    print("="*50)
    print(model)
    
    print("\n" + "="*50)
    print("2. MODEL PARAMETERS (WEIGHTS SHAPES)")
    print("="*50)
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"- {name:<60} | Shape: {str(list(param.shape)):<20} | Params: {param_count:,}")
    print(f"\nTotal trainable parameters: {total_params:,}")

    print("\n" + "="*50)
    print("3. INSPECTING THE CRF TRANSITION MATRIX WEIGHTS")
    print("="*50)
    # The CRF transitions matrix represents transition scores from tag j to tag i.
    # self.transitions is of shape (num_tags, num_tags).
    # self.transitions[i, j] is the score of transitioning from tag j to tag i.
    transitions = model.crf.transitions.detach().cpu().numpy()
    
    # Let's map these to label names for readability
    labels = [model.config.id2label.get(i, model.config.id2label.get(str(i))) for i in range(len(id2label))]
    df_transitions = pd.DataFrame(transitions, index=labels, columns=labels)
    
    print("CRF Transitions Matrix (rows = TO label, columns = FROM label):")
    print(df_transitions.to_string())
    
    print("\nCRF Start Transitions (probabilities of starting with each label):")
    start_transitions = model.crf.start_transitions.detach().cpu().numpy()
    for label, val in zip(labels, start_transitions):
        print(f"- {label:<12}: {val:.4f}")
        
    print("\nCRF End Transitions (probabilities of ending with each label):")
    end_transitions = model.crf.end_transitions.detach().cpu().numpy()
    for label, val in zip(labels, end_transitions):
        print(f"- {label:<12}: {val:.4f}")

    print("\n" + "="*50)
    print("4. INSPECTING CLASSIFIER WEIGHTS")
    print("="*50)
    # Classifier projects hidden states (768) to label space (7)
    clf_weight = model.classifier.weight.detach().cpu().numpy()
    clf_bias = model.classifier.bias.detach().cpu().numpy()
    
    print(f"Classifier Weight Shape: {clf_weight.shape} (num_labels, hidden_size)")
    print(f"Classifier Bias Shape: {clf_bias.shape} (num_labels)")
    print("\nClassifier Bias values per label:")
    for label, val in zip(labels, clf_bias):
        print(f"- {label:<12}: {val:.4f}")

if __name__ == "__main__":
    main()
