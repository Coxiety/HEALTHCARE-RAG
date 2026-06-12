# HEALTHCARE-RAG Detailed Pipeline Workflow

Tài liệu này mô tả chi tiết cách hoạt động của quy trình **RAG (Retrieval-Augmented Generation)** trong dự án **HEALTHCARE-RAG**, bao gồm sơ đồ luồng hoạt động (Activity Diagram) và sơ đồ cấu trúc lớp (Class Diagram).

---

## 1. Luồng Hoạt động Chi tiết của RAG (Activity Diagram)

Quy trình RAG được thiết kế tối ưu với khả năng định tuyến thông minh (Routing), xử lý ngữ cảnh lịch sử hội thoại, truy xuất cơ sở dữ liệu có cấu trúc kết hợp với cơ sở dữ liệu không cấu trúc, xếp hạng lại (Reranking) và áp dụng các quy tắc kiểm soát an toàn (Guardrails) để chống ảo giác thông tin y khoa.

```plantuml
@startuml
skinparam ActivityBackgroundColor #F2F4F7
skinparam ActivityBorderColor #333333
skinparam ActivityFontSize 12
skinparam ArrowColor #4A90E2
skinparam NoteBackgroundColor #FFF9E6

start
:Nhận tin nhắn (Query) và lịch sử chat (History);

partition "1. Query Rewriter (Giải quyết Ngữ cảnh)" {
    if (Có lịch sử chat?) then (Có)
        :Gửi prompt viết lại câu hỏi kèm 5 lượt chat gần nhất đến Ollama;
        :Ollama trả về Standalone Query (loại bỏ từ thay thế như 'it', 'its');
    else (Không)
        :Giữ nguyên Query gốc làm Standalone Query;
    endif
}

partition "2. NLU & Routing (Phân tích & Phân luồng)" {
    fork
        :**Intent Classifier (BERT)**\nPhân loại ý định câu hỏi;
        note right
            Intent có thể là:
            - NUTRITION_LOOKUP
            - HEALTH_ADVICE
            - BOTH
            (Fallback dùng Rule-based từ khóa)
        end note
    fork again
        :**NER Model (BioBERT)**\nTrích xuất thực thể y tế/thực phẩm;
        note right
            Nhận diện:
            - FOOD (Thực phẩm)
            - DISEASE (Bệnh lý)
            - NUTRIENT (Dinh dưỡng)
            - SYMPTOM (Triệu chứng)
            (Fallback dùng spaCy matcher)
        end note
    end fork
}

partition "3. Structured Nutrition Retrieval (Luồng Dinh dưỡng)" {
    if (Intent thuộc NUTRITION_LOOKUP hoặc BOTH?) then (Có)
        :Lấy danh sách thực phẩm (FOOD) trích xuất;
        if (Không tìm thấy FOOD từ NER?) then (Có)
            :Dùng spaCy Noun Chunks trích xuất FOOD từ câu hỏi;
        endif
        
        :**SqliteManager.lookup_en(food_name)**\nTruy vấn CSDL USDA SQLite;
        note right
            - Tìm kiếm gần đúng (LIKE) trên bảng `foods`
            - Ưu tiên foundation_food, loại bỏ đồ chế biến sẵn
            - Truy vấn bảng `food_nutrients` lấy:
              Protein, Energy, Lipid, Carbs, Fiber trên 100g
        end note
    else (Không)
        :Bỏ qua tra cứu USDA;
    endif
}

partition "4. Unstructured Document Retrieval (Luồng Tài liệu Y khoa)" {
    if (Intent thuộc HEALTH_ADVICE hoặc BOTH?) then (Có)
        fork
            :**BM25 Retriever (Sparse)**\nTìm kiếm từ khóa chính xác trên\ntập corpus.jsonl bằng RankBM25;
        fork again
            :**Dense Retriever (Vector)**\n- Embed câu hỏi bằng all-MiniLM-L6-v2\n- Tìm kiếm Cosine Similarity trên ChromaDB;
        end fork
        
        :**Reciprocal Rank Fusion (RRF)**\nHợp nhất kết quả từ BM25 & Dense (K=10);
        
        :**Cross-Encoder Reranker**\nXếp hạng lại các tài liệu bằng model\nms-marco-MiniLM-L-6-v2 với câu hỏi độc lập;
        :Chọn ra Top K tài liệu y khoa tham khảo tốt nhất;
    else (Không)
        :Bỏ qua tra cứu VectorDB/Corpus;
    endif
}

partition "5. Answer Generation (Sinh Câu trả lời & Áp Guardrails)" {
    if (Intent == NUTRITION_LOOKUP\n&& Có đúng 1 thực phẩm khớp USDA?) then (Đúng - Fast-path)
        :**Fast-path Bypass LLM**\nĐịnh dạng bảng dinh dưỡng cứng thành Markdown;
        :Trả về câu trả lời trực tiếp;
    else (Sai)
        :**Build Prompt & Inject Guardrails**;
        note right
            Các quy tắc nghiêm ngặt được chèn vào:
            - So sánh nhiều thức ăn: Bắt buộc dùng Markdown Table
            - Phải dùng chính xác số liệu USDA, không tự chế số
            - Không trả lời y khoa nếu tài liệu tham khảo không nhắc tới
            - Không đề cập đến hệ thống hay luật lệ prompt
        end note
        
        :Gửi Prompt + Lịch sử chat đến Ollama (llama3.1:8b);
        :Lọc bỏ thẻ suy nghĩ <think> nếu có trong câu trả lời;
        :Tổng hợp câu trả lời và các nguồn trích dẫn (Sources);
    endif
}

stop
@endum
```

---

## 2. Sơ đồ các Lớp xử lý RAG trong Python (Class Diagram)

Sơ đồ lớp dưới đây thể hiện cấu trúc mã nguồn Python trong thư mục `src/`, quản lý việc phân loại ý định, trích xuất thực thể, truy xuất dữ liệu từ các kho và tổng hợp lời giải.

```plantuml
@startuml
class ENPipeline {
    - prep: Preprocessor
    - clf: QueryClassifier
    - ner: NERModel
    - vs: VectorStore
    - retriever: HybridRetriever
    - reranker: Reranker
    - db: SqliteManager
    - generator: Generator
    - top_k: int
    - rewriter_model: str
    + answer(query: str, history: list[dict]): dict
    - _condense_query(query: str, history: list[dict]): str
}

class Preprocessor {
    + clean(text: str): str
}

class QueryClassifier {
    - _pipe: pipeline (HuggingFace)
    - _fallback: _RuleBasedFallback
    + classify(text: str): str
}

class _RuleBasedFallback {
    + classify(text: str): str
}

class NERModel {
    - _pipe: pipeline (HuggingFace)
    - _fallback: _SpacyFallback
    + predict(text: str): dict[str, list[str]]
}

class _SpacyFallback {
    + predict(text: str): dict[str, list[str]]
}

class SqliteManager {
    - db_path: Path
    - _connect(): sqlite3.Connection
    - _find_food(food_name: str): sqlite3.Row
    - _get_nutrient(fdc_id: int, nutrient_name: str): dict
    + lookup_en(food_name: str, nutrient_name: str): dict
}

class HybridRetriever {
    - rrf_k: int
    - bm25: BM25Retriever
    - dense: DenseRetriever
    + retrieve(query: str, top_k: int): list[RetrievedChunk]
}

class BM25Retriever {
    - _bm25: BM25Okapi
    - _corpus: list[dict]
    - _build_index()
    - _tokenize(text: str): list[str]
    + retrieve(query: str, top_k: int): list[RetrievedChunk]
}

class DenseRetriever {
    - vs: VectorStore
    + retrieve(query: str, top_k: int): list[RetrievedChunk]
}

class VectorStore {
    - persist_dir: str
    - collection_name: str
    - embedding_model: str
    - _client: PersistentClient
    - _collection: Collection
    - _embedder: SentenceTransformer
    + embed(texts: list[str]): list[list[float]]
    + add(chunks: list[dict])
    + query(text: str, top_k: int): list[RetrievedChunk]
    + get_all_chunks(): list[dict]
}

class Reranker {
    - model_name_or_path: str
    - _model: CrossEncoder
    + rerank(query: str, chunks: list[RetrievedChunk], top_k: int): list[RetrievedChunk]
}

class Generator {
    - model: str
    - host: str
    + build_prompt(query: str, nutrition_data: dict, health_chunks: list, query_type: str): str
    + generate(query: str, nutrition_data: dict, health_chunks: list, query_type: str, history: list[dict]): dict
    - _call_ollama(prompt: str, history: list[dict]): str
    - _call_ollama_generate(prompt: str): str
    - _format_nutrition_answer(nutrition_data: dict): str
    - _format_single_nutrition_data(data: dict): str
}

ENPipeline *-- Preprocessor
ENPipeline *-- QueryClassifier
ENPipeline *-- NERModel
ENPipeline *-- VectorStore
ENPipeline *-- HybridRetriever
ENPipeline *-- Reranker
ENPipeline *-- SqliteManager
ENPipeline *-- Generator

QueryClassifier *-- _RuleBasedFallback
NERModel *-- _SpacyFallback
HybridRetriever *-- BM25Retriever
HybridRetriever *-- DenseRetriever
DenseRetriever *-- VectorStore
@endum
```
