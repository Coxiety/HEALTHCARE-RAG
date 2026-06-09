# Tài liệu chi tiết Hệ thống HEALTHCARE-RAG

## 1. Giới thiệu Tổng quan
Hệ thống **HEALTHCARE-RAG** là một ứng dụng áp dụng kỹ thuật Retrieval-Augmented Generation (RAG) kết hợp giữa tra cứu dinh dưỡng có cấu trúc và tư vấn sức khỏe dựa trên các tài liệu y khoa. 

Hệ thống được thiết kế với kiến trúc phân lớp rõ ràng, kết hợp linh hoạt giữa các mô hình học sâu nhỏ chạy local (Ollama, Hugging Face) và các API mạnh mẽ (Google Gemini).

## 2. Kiến trúc và Cách hoạt động của Pipeline (EN Pipeline)
Cốt lõi của hệ thống xử lý ngôn ngữ nằm ở thư mục `src/en/`. Quy trình xử lý một câu hỏi (query) từ người dùng đi qua các bước sau:

### 2.1. Query Rewriter (Xử lý ngữ cảnh đa lượt)
- Khi người dùng đặt câu hỏi, nếu có lịch sử trò chuyện (history), hệ thống sẽ chạy module **Query Rewriter** để giải quyết hiện tượng đồng tham chiếu (co-reference resolution).
- **Ví dụ:** Lượt 1: *"Can you tell me about Apple?"* -> Lượt 2: *"What is its protein content?"*. Rewriter sẽ viết lại thành: *"What is the protein content of Apple?"*.
- **Mô hình sử dụng:** Hỗ trợ Gemini API hoặc mô hình LLM local thông qua Ollama (Mặc định cấu hình dùng `llama3.1:8b`).

### 2.2. Query Classification (Phân loại ý định)
- Câu hỏi sau khi được làm rõ sẽ được đưa qua `QueryClassifier` (`src/en/classifier.py`).
- **Nhiệm vụ:** Phân loại câu hỏi vào một trong ba nhóm (Intent):
  1. `NUTRITION_LOOKUP`: Các câu hỏi thuần túy tra cứu lượng dinh dưỡng (ví dụ: calories, protein trong thực phẩm).
  2. `HEALTH_ADVICE`: Các câu hỏi tư vấn sức khỏe, bệnh lý, chế độ ăn (ví dụ: người bệnh tiểu đường nên ăn gì).
  3. `BOTH`: Giao thoa giữa cả hai (ví dụ: lượng đường trong gạo và tác động lên bệnh tiểu đường).
- **Mô hình AI:** Sử dụng mô hình **BERT (Fine-tuned)** để phân loại. Nếu model BERT chưa có (lúc đang train), hệ thống tự động fallback xuống dùng các tập keyword dựa trên Rule-based.

### 2.3. Named Entity Recognition (Nhận diện Thực thể - NER)
- Dữ liệu tiếp tục qua `NERModel` (`src/en/ner.py`) để trích xuất các thực thể sinh học/y tế.
- **Thực thể mục tiêu:** `FOOD` (Thực phẩm), `DISEASE` (Bệnh lý), `NUTRIENT` (Dưỡng chất), `SYMPTOM` (Triệu chứng).
- **Mô hình AI:** Cốt lõi sử dụng **BioBERT** được fine-tune trên tập dữ liệu y tế BC5CDR. Mô hình sẽ quét câu và gán nhãn cho từng từ.
- Trở ngại: Nếu BioBERT thất bại, hệ thống dự phòng (Fallback) bằng cơ chế rule-based của `spaCy` quét qua từ khóa để tránh lỗi hệ thống toàn cục.

### 2.4. Khối Truy xuất Thông tin (Retrieval Engine)
Khối này chia làm hai luồng tùy thuộc vào Intent đã được phân loại:

#### A. Structured Database Retrieval (Tra cứu Dinh dưỡng)
- Nếu Intent là `NUTRITION_LOOKUP` hoặc `BOTH`, hệ thống sử dụng tên `FOOD` trích xuất từ NER để truy vấn vào cơ sở dữ liệu SQLite (`data/usda_food.db`).
- Dữ liệu trong SQLite là kho dữ liệu chuẩn ánh xạ từ **USDA FoodData Central**, cung cấp con số tuyệt đối về Calories, Protein, Fat, v.v.

#### B. Semantic Document Retrieval (Tra cứu Tài liệu Y khoa)
- Nếu Intent là `HEALTH_ADVICE` hoặc `BOTH`, hệ thống sẽ tìm kiếm thông tin y khoa qua module `HybridRetriever` (`src/en/retriever.py`).
- **Hybrid Retrieval (Truy xuất Lai):** Hệ thống lấy các tài liệu liên quan từ 2 nguồn:
  - **BM25 (Sparse Retrieval):** Đếm tần suất từ vựng, hỗ trợ tìm kiếm keyword chính xác.
  - **Dense Retrieval:** Dùng Vector Database (ChromaDB) kết hợp nhúng (Embedding) qua model `sentence-transformers/all-MiniLM-L6-v2`, giúp hiểu theo ngữ nghĩa sâu của từ vựng.
  - **Hợp nhất (RRF):** Kết quả của hai bộ máy được hợp nhất (Fuse) qua thuật toán **Reciprocal Rank Fusion**.
- **Reranker:** Những văn bản lấy lên được quét lần cuối bởi Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) để xếp hạng lại độ liên quan một cách chuẩn xác nhất, lọc ra top `k` kết quả hữu ích nhất.

### 2.5. Khối Khởi tạo LLM (Generator)
- Khối `Generator` (`src/generation/generator.py`) tổng hợp mọi dữ liệu: Câu hỏi, Bảng dinh dưỡng USDA và Các tài liệu y khoa.
- **Đặc biệt (Fast-path):** Nếu câu hỏi thuần `NUTRITION_LOOKUP` và hệ thống tra được duy nhất một món ăn đầy đủ số liệu, nó sẽ cấu trúc hóa chuỗi kết quả và gửi lại ngay, không qua LLM để tối ưu chi phí và độ trễ.
- **Prompt Engineering:** Hệ thống tiêm (inject) các ràng buộc khắc nghiệt cho LLM:
  - Phải tuân theo số liệu của USDA, không tự bịa số.
  - Nếu so sánh thực phẩm, phải dùng Bảng Markdown (Markdown Table).
  - Không được trả lời tư vấn bệnh lý nếu không tìm thấy trong tài liệu truy xuất (Tránh Hallucination trong lĩnh vực y tế).
- **Xử lý LLM:** Chạy qua Google Gemini API, hoặc Ollama (với `llama3.1:8b`). Có cơ chế tự động Fallback về các LLM cục bộ nếu API Gemini gặp sự cố.

---

## 3. Hệ thống Đánh giá (Evaluation)
Hệ thống được phát triển chuyên nghiệp với một pipeline tự động đánh giá chất lượng tại `src/evaluation/rag_evaluator.py`.

### Các Metrics (Chỉ số) Quan Trọng:
1. **Đánh giá Intent (Intent Accuracy):** 
   - Đo lường mức độ mô hình (BERT Classifier) phân loại đúng nhóm `NUTRITION`, `HEALTH`, `BOTH`.
2. **Đánh giá Retrieval:**
   - **Hit Rate / Recall@k / Precision@k:** Khả năng truy tìm được đúng các văn bản tài liệu thiết yếu so với nhãn mẫu (Golden standard).
   - **MRR@k (Mean Reciprocal Rank):** Vị trí của câu trả lời đúng xuất hiện sớm đến mức nào (Càng gần vị trí Top 1 thì MRR càng cao).
3. **Đánh giá Text Generation:**
   - **Token F1:** Đánh giá mức độ trùng lặp nội dung giữa văn bản LLM sinh ra và văn bản mẫu của chuyên gia.
   - **Keyword Hits:** Tính tỉ lệ các từ khóa y tế bắt buộc có xuất hiện trong câu trả lời sinh ra hay không.
   - **Source Keyword Hits:** Đảm bảo LLM có trích dẫn đúng các nguồn tài liệu y khoa đã quy định.
4. **Latency:** Đo lường độ trễ từng cấu phần (classifier, retrieval, generator) để phục vụ tối ưu hóa tốc độ.

---

## 4. Giải thích Sâu về tính Agentic AI trong dự án

### 4.1. Router-Driven AI (Agentic Routing)
RAG truyền thống chỉ lấy VectorDB đập chung với câu hỏi rồi gửi LLM. Hệ thống HEALTHCARE-RAG thể hiện sự trưởng thành của một **Agent** khi có bộ não phân loại riêng (Classifier). Nó "suy nghĩ" (quyết định) xem có nên đi qua nhánh tra cứu bảng SQLite (dữ liệu cứng) hay nhánh VectorDB (dữ liệu mềm), hoặc đi cả hai. Đây là biểu hiện căn bản của Agentic Workflow.

### 4.2. Khắc phục Điểm Yếu của Dense Retrieval (Embeddings)
- Embeddings rất giỏi trong việc hiểu ngữ nghĩa khái quát nhưng thường dở tệ trong việc tìm kiếm các mã số cụ thể (ví dụ: "Vitamin B12" hoặc "fdc_id=123"). Hệ thống HEALTHCARE-RAG giải quyết điểm yếu này triệt để thông qua Hybrid Retrieval (kết hợp Dense với BM25) để vừa hiểu ngữ cảnh chung, vừa không đánh rơi từ khóa cứng.
- Cross-Encoder Re-ranker đóng vai trò như một bộ phán xử tinh xảo, nó đọc cẩn thận cả câu hỏi và văn bản thay vì chỉ tính khoảng cách Cosine như Dense Retrieval, từ đó nâng độ chính xác lên cực đại.

### 4.3. Kiến trúc Micro-Models (Chia để trị)
Thay vì dùng 1 LLM siêu lớn để làm mọi thứ (vừa nhận diện thực thể, phân loại, vừa trả lời), hệ thống khéo léo dùng các mô hình cực nhỏ (SLM - Small Language Models) như BioBERT và MiniLM. Việc huấn luyện (fine-tune) riêng cho các Model nhỏ này giúp tốc độ của pipeline cực kỳ nhanh, chạy mượt mà ngay cả trên thiết lập CPU, đồng thời nhường lại việc suy luận cuối cùng cho các LLM lớn.

### 4.4. Guardrails chống Hallucination (Ảo giác AI)
Hệ thống chèn các chỉ thị rất cụ thể vào LLM prompt (nhấn mạnh qua viết hoa "MUST", "IMPORTANT") để chặn đứng LLM đưa ra các lời khuyên y tế vô căn cứ. Trong bối cảnh Y Tế (Healthcare), việc một AI chế ra sai lệch dinh dưỡng hay chẩn đoán bệnh là vô cùng nguy hiểm. Pipeline này đã tính đến điều kiện trên bằng cách lấy dữ liệu USDA làm mỏ neo tuyệt đối cho các con số dinh dưỡng.
