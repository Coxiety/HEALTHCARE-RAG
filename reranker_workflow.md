# HEALTHCARE-RAG Cross-Encoder Reranker Workflow

Tài liệu này mô tả chi tiết cách thức hoạt động, kiến trúc lớp và luồng xử lý dữ liệu của cấu phần **Reranker** trong hệ thống **HEALTHCARE-RAG**.

---

## 1. Sơ đồ Hoạt động tại Runtime (Activity Diagram)

Mô tả luồng xếp hạng lại (reranking) các tài liệu thô được trả về từ bộ lọc truy xuất lai (Hybrid Retrieval).

```plantuml
@startuml
skinparam ActivityBackgroundColor #F2F4F7
skinparam ActivityBorderColor #333333
skinparam ActivityFontSize 12
skinparam ArrowColor #4A90E2
skinparam NoteBackgroundColor #FFF9E6

start
:Nhận đầu vào: Standalone Query và Danh sách RetrievedChunks thô;

if (Danh sách RetrievedChunks rỗng?) then (Có)
    :Trả về danh sách rỗng [];
    stop
else (Không)
    partition "Lazy Loading Model" {
        if (self._model đã được khởi tạo?) then (Không)
            :Khởi tạo CrossEncoder với model_name_or_path
            (Mặc định: cross-encoder/ms-marco-MiniLM-L-6-v2);
        else (Có)
            :Sử dụng model đã có trong cache;
        endif
    }
    
    partition "Score Prediction (Dự đoán Điểm số)" {
        :Ghép cặp câu hỏi với từng chunk văn bản:
        pairs = [(query, chunk.text) for chunk in chunks];
        
        :Đưa các cặp văn bản vào mô hình CrossEncoder;
        note right
            Khác biệt của Cross-Encoder so với Bi-Encoder (Dense embedding):
            - Bi-Encoder embed Câu hỏi & Tài liệu một cách riêng lẻ.
            - Cross-Encoder nhận CẢ HAI cùng lúc đi qua mô hình BERT.
            - Cho phép Attention toàn cục (Full Cross-Attention) giữa từng từ
              trong Câu hỏi với từng từ trong Tài liệu.
            - Sinh ra điểm số tương quan (similarity score) có độ chính xác cực cao.
        end note
        
        :Model trả về mảng điểm số (scores);
    }
    
    partition "Re-ordering & Selection (Xếp hạng & Lọc)" {
        :Hợp nhất điểm số mới với các RetrievedChunk tương ứng;
        :Sắp xếp (sort) danh sách theo điểm số giảm dần;
        :Cắt lấy top_k phần tử đầu tiên (Mặc định top_k = 3 hoặc 5);
    }
    
    :Trả về danh sách RetrievedChunks đã được xếp hạng lại;
endif
stop
@endum
```

---

## 2. Quy trình Huấn luyện Mô hình Reranker (Sequence Diagram)

Mô tả quy trình huấn luyện tinh chỉnh (Fine-tuning) mô hình Cross-Encoder để tối ưu hóa khả năng hiểu thuật ngữ y tế/dinh dưỡng từ tập dữ liệu **NFCorpus** (sử dụng mã nguồn `main/train_reranker.py`).

```plantuml
@startuml
autonumber
actor Developer
participant "train_reranker.py" as Script
database "NFCorpus Data\n(queries/corpus/qrels)" as Data
participant "BM25 Engine" as BM25
participant "CrossEncoder Model" as Model
participant "CrossEncoderTrainer" as Trainer
database "Disk Output\n(models/reranker_domain)" as Disk

Developer -> Script : Chạy lệnh python main/train_reranker.py
Script -> Data : Tải câu hỏi (queries), văn bản (corpus) và nhãn liên quan (qrels)
Data --> Script : Trả về dữ liệu thô

Script -> BM25 : Xây dựng chỉ mục BM25 trên toàn bộ corpus
BM25 --> Script : Chỉ mục BM25 sẵn sàng

loop Với mỗi câu hỏi (Query) trong tập Train
    Script -> Script : Lấy danh sách tài liệu liên quan thực tế (Positive Docs - label=1.0)
    Script -> BM25 : Truy vấn câu hỏi để tìm các tài liệu tương đồng từ vựng cao
    BM25 --> Script : Trả về danh sách BM25 scores
    Script -> Script : Loại bỏ các positive docs để thu về Hard Negatives (label=0.0)
    Script -> Script : Lấy ngẫu nhiên một số tài liệu khác làm Random Negatives (label=0.0)
    Script -> Script : Tổng hợp thành các cặp (query, document_text, label)
end

Script -> Model : Khởi tạo CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
Model --> Script : Model instance sẵn sàng

Script -> Trainer : Khởi tạo Trainer với:
note left of Trainer
    - train_dataset (cặp 1.0 & 0.0)
    - loss = BinaryCrossEntropyLoss
    - evaluator = CrossEncoderRerankingEvaluator (NFCorpus-dev)
end note
Trainer --> Script : Trainer instance sẵn sàng

Script -> Trainer : Gọi train()
loop Qua từng epoch huấn luyện
    Trainer -> Trainer : Huấn luyện tinh chỉnh weights bằng BCE Loss
    Trainer -> Trainer : Chạy Evaluator tính chỉ mục MRR@10 trên tập dev
end
Trainer --> Script : Trả về mô hình tốt nhất (Best Checkpoint)

Script -> Disk : Lưu mô hình pretrained đã được tinh chỉnh (safe_serialization=True)
Disk --> Developer : Hoàn tất. Reranker sẵn sàng phục vụ RAG.
@endum
```

---

## 3. Cấu trúc Lớp Reranker (Class Diagram)

```plantuml
@startuml
class Reranker {
    + MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    - model_name_or_path: str
    - _model: CrossEncoder | None
    + __init__(model_name_or_path: str | None)
    + rerank(query: str, chunks: list[RetrievedChunk], top_k: int): list[RetrievedChunk]
    - _get_model(): CrossEncoder
}

class RetrievedChunk {
    + text: str
    + source: str
    + score: float
}

Reranker ..> RetrievedChunk : Nhận vào & Trả về
@endum
```
