# HEALTHCARE-RAG Project Architecture & Design Documentation

Tài liệu này mô tả chi tiết kiến trúc, luồng xử lý dữ liệu và mối quan hệ giữa các thành phần của hệ thống **HEALTHCARE-RAG** - ứng dụng RAG (Retrieval-Augmented Generation) hỗ trợ tra cứu dinh dưỡng và tư vấn sức khỏe.

---

## 1. Kiến trúc Hệ thống (System Architecture)

Hệ thống được thiết kế theo kiến trúc phân tách rõ ràng giữa **Frontend Web Application (Spring Boot)** và **RAG Backend Server (FastAPI)**, tích hợp các thành phần cơ sở dữ liệu và mô hình AI chạy cục bộ.

### Biểu đồ Component (PlantUML)

```plantuml
@startuml
skinparam actorStyle awesome
actor User

package "Frontend Web Application (Spring Boot)" {
    [Web UI] as UI
    [ChatController] as CC
    [AuthController] as AC
    database "App Database\n(SQLite/MySQL)" as AppDB
}

package "RAG Backend (FastAPI)" {
    [FastAPI Server] as API
    [ENPipeline] as Pipe
    [QueryClassifier (BERT)] as Classif
    [NERModel (BioBERT)] as NER
    [HybridRetriever] as Retriever
    [Reranker (Cross-Encoder)] as Ranker
    [Generator] as Gen
    
    database "USDA SQLite Database" as USDADB
    database "Chroma Vector Database" as ChromaDB
}

package "External Models / Services" {
    [Ollama Server\n(Local LLM)] as Ollama
    [Hugging Face / Transformers] as HF
}

User --> UI : Tương tác giao diện
UI --> CC : Gửi yêu cầu REST API (/api/chat)
CC --> AppDB : Lưu lịch sử hội thoại & nhật ký ăn uống
CC --> API : Chuyển tiếp câu hỏi REST API (/ask)

API --> Pipe : answer(query, history)
Pipe --> Classif : classify(query) -> Intent
Pipe --> NER : predict(query) -> Entities
Pipe --> USDADB : Tra cứu dinh dưỡng (via SqliteManager)
Pipe --> Retriever : Tìm tài liệu y khoa (via HybridRetriever)
Retriever --> ChromaDB : Dense Search (Semantic)
Retriever --> USDADB : Sparse Search (BM25 corpus)
Pipe --> Ranker : rerank(query, chunks) -> Ranked Chunks
Pipe --> Gen : generate(query, nutrition, chunks, intent, history)

Gen --> Ollama : Sinh văn bản (/api/chat)
Classif --> HF : Inference (BERT)
NER --> HF : Inference (BioBERT)
Ranker --> HF : Inference (MiniLM)
@endum
```

---

## 2. Luồng xử lý yêu cầu (Sequence Diagram)

Quy trình từ lúc người dùng gửi câu hỏi đến khi hệ thống trả về câu trả lời tối ưu kết hợp giữa cơ sở dữ liệu dinh dưỡng USDA và các bài báo khoa học.

### Biểu đồ Sequence (PlantUML)

```plantuml
@startuml
autonumber
actor User
participant "Spring Boot\nChatController" as CC
database "Spring Boot CSDL\n(SQLite/MySQL)" as AppDB
participant "FastAPI\nrag_server" as FastAPI
participant "ENPipeline" as Pipe
participant "QueryClassifier" as Clf
participant "NERModel" as NER
database "USDA SQLite" as SQLite
participant "HybridRetriever" as Ret
participant "Reranker" as Rerank
participant "Generator" as Gen
participant "Ollama Local" as Ollama

User -> CC : Gửi tin nhắn ("Is apple healthy for diabetes?")
CC -> AppDB : Lưu tin nhắn User (ChatMessage)
CC -> CC : Lấy lịch sử chat (tối đa 5 lượt)
CC -> FastAPI : POST /ask {message, history}
FastAPI -> Pipe : answer(message, history)

alt Nếu có lịch sử chat (Xử lý ngữ cảnh)
    Pipe -> Gen : Viết lại câu hỏi (Condense Query)
    Gen -> Ollama : POST /api/chat (Query rewrite prompt)
    Ollama --> Gen : Trả về standalone query
    Gen --> Pipe : Standalone query ("Is apple healthy for diabetes?")
end

Pipe -> Clf : classify(search_query)
Clf --> Pipe : Trả về intent ("BOTH")

Pipe -> NER : predict(search_query)
NER --> Pipe : Trả về entities {"FOOD": ["apple"], "DISEASE": ["diabetes"]}

alt Intent là NUTRITION_LOOKUP hoặc BOTH
    Pipe -> SQLite : lookup_en("apple")
    SQLite --> Pipe : Trả về USDA nutrition facts (Protein, Lipid, Carbs, Calories, Fiber)
end

alt Intent là HEALTH_ADVICE hoặc BOTH
    Pipe -> Ret : retrieve("Is apple healthy for diabetes?")
    Ret --> Pipe : Trả về các văn bản y khoa thô (BM25 + ChromaDB)
    Pipe -> Rerank : rerank(search_query, chunks)
    Rerank --> Pipe : Trả về top k văn bản có điểm số cao nhất (Cross-Encoder)
end

Pipe -> Gen : generate(search_query, nutrition, chunks, intent, history)
alt Trường hợp NUTRITION_LOOKUP & chỉ có 1 sản phẩm
    Gen --> Pipe : (Fast-path) Trả về chuỗi Markdown format trực tiếp
else Các trường hợp khác
    Gen -> Gen : Ghép Prompt y khoa/dinh dưỡng (Inject Guardrails)
    Gen -> Ollama : POST /api/chat (System prompt + History + Context + Query)
    Ollama --> Gen : Sinh câu trả lời (Markdown)
    Gen --> Pipe : Trả về kết quả câu trả lời + sources
end

Pipe --> FastAPI : Trả về dict kết quả (answer, intent, entities, sources)
FastAPI --> CC : HTTP 200 OK json
CC -> AppDB : Lưu nhật ký món ăn (FoodRecord) nếu tìm thấy FOOD
CC -> AppDB : Lưu tin nhắn AI (ChatMessage với metadata)
CC --> User : Trả về câu trả lời, intent, entities, sources
@endum
```

---

## 3. Mô tả Chi tiết các Thành phần trong Project

### 3.1. Frontend Web Application (`chatbot/`)
Được phát triển bằng **Spring Boot**, chịu trách nhiệm quản lý người dùng, giao tiếp UI và quản trị dữ liệu ứng dụng.
*   **Controllers**:
    *   [ChatController](file:///d:/Data/File%20for%20Google%20Drive%20real/Project/Nutrition_RAG/HEALTHCARE-RAG/chatbot/src/main/java/com/webdinhduong/chatbot/controller/ChatController.java): Điểm tiếp nhận tin nhắn từ giao diện web, tích hợp cơ chế phân tích lịch sử chat (tối đa 5 lượt nhắn gần nhất), gọi RAG API cục bộ, lưu trữ thực thể món ăn vào nhật ký cá nhân và lưu lịch sử chat vào CSDL.
    *   `AuthController`: Xử lý đăng ký, đăng nhập và bảo mật thông qua JWT (JSON Web Token).
*   **Entities (Models)**:
    *   `User`: Lưu thông tin tài khoản người dùng.
    *   `ChatMessage`: Lưu nội dung chat, vai trò (`user` / `ai`), ý định (`intent`), các thực thể (`entitiesJson`), nguồn trích dẫn (`sourcesJson`) và năng lượng của món ăn liên quan.
    *   `FoodRecord`: Nhật ký món ăn được trích xuất tự động khi người dùng hỏi về thực phẩm, hỗ trợ theo dõi calo.
*   **Security & Database Config**:
    *   `SecurityConfig` & `JwtFilter`: Cấu hình phân quyền truy cập API.
    *   `DatabaseConfig`: Cấu hình kết nối cơ sở dữ liệu SQLite của ứng dụng.

### 3.2. FastAPI Server (`main/rag_server.py`)
Là REST API backend viết bằng **FastAPI**, làm cầu nối trung gian giữa ứng dụng Spring Boot và Pipeline Python.
*   Khởi tạo `ENPipeline` thông qua cơ chế `lifespan` bất đồng bộ để tránh trễ trong quá trình tải các mô hình học sâu.
*   Cung cấp endpoint `/ask` tiếp nhận payload gồm `message` và `history` của cuộc hội thoại, chuyển đổi kiểu dữ liệu tương thích và chạy pipeline xử lý trong luồng riêng biệt (`run_in_threadpool`) nhằm không gây nghẽn máy chủ.

### 3.3. RAG Core Pipeline (`src/en/` & `src/database/` & `src/generation/`)
Trái tim của hệ thống xử lý ngôn ngữ tự nhiên và truy xuất thông tin:

1.  **Preprocessor (`src/en/preprocessor.py`)**: Chuẩn hóa văn bản đầu vào.
2.  **Query Condenser (trong [ENPipeline](file:///d:/Data/File%20for%20Google%20Drive%20real/Project/Nutrition_RAG/HEALTHCARE-RAG/src/en/pipeline.py))**:
    *   Giải quyết vấn đề đồng tham chiếu (Co-reference resolution). Sử dụng Local LLM (Ollama) để viết lại câu hỏi follow-up dựa trên tối đa 5 lượt hội thoại lịch sử thành một câu hỏi độc lập (Standalone question).
3.  **Query Classifier (`src/en/classifier.py`)**:
    *   Phân loại ý định của người dùng thành `NUTRITION_LOOKUP` (Tra cứu dinh dưỡng), `HEALTH_ADVICE` (Tư vấn sức khỏe), hoặc `BOTH` (Cả hai).
    *   Sử dụng mô hình **BERT** đã được fine-tune hoặc cơ chế fallback **Rule-Based** (phân tích tập từ khóa) nếu mô hình BERT chưa được tải.
4.  **NER Model (`src/en/ner.py`)**:
    *   Nhận diện các thực thể sinh học và y tế: `FOOD`, `DISEASE`, `NUTRIENT`, `SYMPTOM`.
    *   Sử dụng mô hình **BioBERT** fine-tuned trên BC5CDR hoặc cơ chế fallback sử dụng **spaCy** keyword matching.
5.  **SqliteManager (`src/database/sqlite_manager.py`)**:
    *   Thực hiện truy vấn dữ liệu cứng từ USDA SQLite Database (`data/usda_food.db`).
    *   Thực hiện thuật toán tìm kiếm gần đúng dựa trên các từ khóa (Keyword search), lọc bỏ các từ chung chung và sắp xếp ưu tiên các thực phẩm nguyên bản (`foundation_food`) so với các thực phẩm chế biến sẵn.
6.  **HybridRetriever (`src/en/retriever.py`)**:
    *   **Sparse Retrieval (BM25)**: Sử dụng thư viện `rank_bm25` để tìm kiếm chính xác các từ khóa y học trên tập dữ liệu corpus (`data/en/corpus.jsonl`).
    *   **Dense Retrieval**: Sử dụng ChromaDB để tìm kiếm ngữ nghĩa sâu thông qua nhúng vector embeddings bằng model `all-MiniLM-L6-v2`.
    *   **Reciprocal Rank Fusion (RRF)**: Hợp nhất kết quả từ hai phương thức truy xuất trên với hệ số điều chỉnh `k=10` để lấy ra các đoạn văn bản tối ưu nhất.
7.  **Reranker (`src/en/reranker.py`)**:
    *   Sử dụng Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) để tính toán điểm số tương quan trực tiếp giữa câu hỏi và từng đoạn tài liệu được chọn, sắp xếp lại để lấy ra top kết quả chuẩn xác nhất.
8.  **Generator (`src/generation/generator.py`)**:
    *   **Fast-path**: Nếu câu hỏi chỉ tra cứu dinh dưỡng của một loại thực phẩm cụ thể, hệ thống sẽ format kết quả thành Markdown và trả về trực tiếp mà không cần gọi LLM, giúp tiết kiệm tài nguyên.
    *   **LLM Generation**: Lắp ráp prompt với dữ liệu dinh dưỡng (nếu có), các tài liệu y khoa tham khảo và chèn các chỉ thị nghiêm ngặt (Guardrails) để chống ảo giác AI (Hallucination). Gọi API của local Ollama (`/api/chat` hoặc `/api/generate`) để sinh câu trả lời.
