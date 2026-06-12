# HEALTHCARE-RAG Classifier BERT Workflow

Tài liệu này mô tả chi tiết cách thức hoạt động, kiến trúc lớp và luồng xử lý dữ liệu của cấu phần **Classifier BERT** (`QueryClassifier`) trong hệ thống **HEALTHCARE-RAG**.

---

## 1. Sơ đồ Luồng hoạt động của Classifier (Activity Diagram)

Mô tả chi tiết luồng xử lý từ khâu khởi tạo đối tượng, kiểm tra tính sẵn sàng của mô hình học sâu cho đến khi đưa ra quyết định phân loại ý định (Intent Routing).

```plantuml
@startuml
skinparam ActivityBackgroundColor #F2F4F7
skinparam ActivityBorderColor #333333
skinparam ActivityFontSize 12
skinparam ArrowColor #4A90E2
skinparam NoteBackgroundColor #FFF9E6

start
partition "Khởi tạo (__init__)" {
    :Đọc configs/config.yaml lấy 'classifier_model_path';
    if (Tìm thấy config.json trong thư mục model?) then (Có - Model Sẵn sàng)
        :Import Hugging Face transformers pipeline;
        :Khởi tạo pipeline 'text-classification'
        - Model: classifier_bert
        - Device: CPU (-1);
        :Đặt self._fallback = None;
    else (Không - Dùng Fallback)
        :Khởi tạo đối tượng _RuleBasedFallback;
        :Đặt self._pipe = None;
    endif
}

partition "Phân loại (classify)" {
    :Nhận tham số 'text' (Standalone Query);
    if (self._fallback != None?) then (Đúng - Chạy Rule-Based)
        :Chuyển text về chữ thường (lowercase);
        fork
            :Kiểm tra từ khóa dinh dưỡng (_NUTRITION_KW)
            Ví dụ: calorie, protein, fat, vitamin...;
        fork again
            :Kiểm tra từ khóa y tế (_HEALTH_KW)
            Ví dụ: diabetes, symptom, heart, healthy...;
        end fork
        
        if (Có cả từ khóa Dinh dưỡng & Y tế?) then (Có)
            :Trả về intent "BOTH";
        elseif (Chỉ có từ khóa Dinh dưỡng?) then (Có)
            :Trả về intent "NUTRITION_LOOKUP";
        else (Chỉ có từ khóa Y tế / Khác)
            :Trả về intent "HEALTH_ADVICE";
        endif
    else (Sai - Chạy Deep Learning BERT)
        :Đưa văn bản vào pipeline của HuggingFace;
        note right
            Bên dưới pipeline thực hiện:
            1. Tokenizer: Chuyển text thành Input IDs, Attention Mask (max_length=128)
            2. BERT Model: Trích xuất đặc trưng văn bản thông qua Transformer Layers
            3. Classification Head: Lấy vector ẩn của token [CLS], nhân ma trận tuyến tính
            4. Softmax: Chuyển đổi Logits thành phân phối xác suất 3 lớp
        end note
        :Trích xuất nhãn có điểm xác suất cao nhất
        ("NUTRITION_LOOKUP", "HEALTH_ADVICE", hoặc "BOTH");
        :Trả về nhãn phân loại;
    endif
}
stop
@endum
```

---

## 2. Quy trình Huấn luyện Mô hình BERT (Sequence Diagram)

Quy trình chuẩn bị dữ liệu tổng hợp (synthetic data) và huấn luyện mô hình để lưu vào đĩa cứng trước khi chạy ứng dụng thực tế.

```plantuml
@startuml
autonumber
actor Developer
participant "training_classifier.ipynb" as Notebook
database "Synthetic Data\n(3-class)" as Data
participant "BERT Tokenizer" as Tokenizer
participant "Pre-trained BERT\n(e.g., bert-base-uncased)" as Pretrained
participant "HuggingFace Trainer" as Trainer
database "Disk Storage\n(models/classifier_bert)" as Disk

Developer -> Notebook : Chạy tiến trình Huấn luyện
Notebook -> Data : Tải/Tạo tập dữ liệu câu hỏi tổng hợp
note left of Data
    Ví dụ:
    - "how many calories in apple" -> NUTRITION_LOOKUP
    - "treatment for hypertension" -> HEALTH_ADVICE
    - "does sugar affect diabetes" -> BOTH
end note
Data --> Notebook : Trả về Dataset (Train / Val)

Notebook -> Tokenizer : Tokenize dữ liệu văn bản
Tokenizer --> Notebook : Trả về tensors (input_ids, attention_mask)

Notebook -> Pretrained : Load pre-trained BERT model\nkèm Classification Head (3 lớp)
Pretrained --> Notebook : Trả về Model instance

Notebook -> Trainer : Khởi tạo Trainer (Model, Dataset, TrainingArguments)
Notebook -> Trainer : Gọi train()
loop Qua từng Epoch (Huấn luyện)
    Trainer -> Trainer : Tính Forward Pass
    Trainer -> Trainer : Tính Loss (CrossEntropyLoss)
    Trainer -> Trainer : Tính Backward Pass (Cập nhật Weights)
end
Trainer --> Notebook : Huấn luyện hoàn tất

Notebook -> Disk : Lưu mô hình (save_model)
note right of Disk
    Lưu các file:
    - config.json (Cấu hình nhãn và model)
    - model.safetensors hoặc pytorch_model.bin
    - vocab.txt, tokenizer.json (Dữ liệu Tokenizer)
end note
Disk --> Developer : Sẵn sàng cho Inference tại Runtime
@endum
```

---

## 3. Cấu trúc Lớp trong Python (Class Diagram)

```plantuml
@startuml
class QueryClassifier {
    - _pipe: pipeline | None
    - _fallback: _RuleBasedFallback | None
    + __init__()
    + classify(text: str): str
}

class _RuleBasedFallback {
    + classify(text: str): str
}

QueryClassifier *-- _RuleBasedFallback

note top of QueryClassifier
  Nhiệm vụ: Nhận diện ý định của người dùng 
  để chuyển tiếp yêu cầu (Routing) đến đúng 
  nhánh xử lý cơ sở dữ liệu.
end note

note bottom of _RuleBasedFallback
  Kích hoạt khi chưa tải hoặc chưa huấn luyện 
  mô hình BERT. Quét qua 2 tập từ khóa:
  - _NUTRITION_KW (23 từ khóa)
  - _HEALTH_KW (29 từ khóa)
end note
@endum
```
