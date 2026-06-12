# HEALTHCARE-RAG AI Core Workflow

Tài liệu này cung cấp sơ đồ hoạt động (Activity Diagram) chi tiết nhất về luồng xử lý RAG AI cốt lõi của dự án **HEALTHCARE-RAG**, tập trung hoàn toàn vào các mô hình học sâu, truy xuất dữ liệu y khoa và sinh câu trả lời bằng LLM (không bao gồm phần Web/Spring Boot).

---

## Sơ đồ Hoạt động RAG AI Cốt lõi (PlantUML)

```plantuml
@startuml
skinparam ActivityBackgroundColor #F2F4F7
skinparam ActivityBorderColor #333333
skinparam ActivityFontSize 12
skinparam ArrowColor #4A90E2
skinparam NoteBackgroundColor #FFF9E6

start

:Nhận câu hỏi người dùng (Query) và lịch sử chat (History);

' ==========================================
' PARTITION 1: QUERY REWRITER
' ==========================================
partition "1. Query Rewriter (Ollama)" {
    if (Có lịch sử chat?) then (Có)
        :Dựng prompt viết lại câu hỏi (Context-Aware Rewrite);
        note right
            - Ghép tối đa 5 lượt hội thoại lịch sử.
            - Yêu cầu thay thế các đại từ (it, its, they,...) bằng thực phẩm tương ứng.
            - Nếu câu hỏi mới hoàn toàn, giữ nguyên nội dung.
        end note
        :Gọi Ollama API (/api/chat hoặc /api/generate);
        :Nhận kết quả Standalone Query (Search Query);
    else (Không)
        :Search Query = Query gốc;
    endif
}

' ==========================================
' PARTITION 2: NLU & INTENT/NER
' ==========================================
partition "2. NLU Processing (BERT & BioBERT)" {
    fork
        label clf_start
        if (Mô hình BERT Classifier sẵn sàng?) then (Có)
            :Đưa Search Query qua pipeline 'text-classification';
            :BERT trích xuất đặc trưng văn bản & dự đoán Logits;
            :Áp dụng Softmax phân loại ý định (Intent):
            - NUTRITION_LOOKUP
            - HEALTH_ADVICE
            - BOTH;
        else (Không)
            :Chuyển Search Query về lowercase;
            :So khớp từ khóa với 2 tập từ điển:
            - _NUTRITION_KW (calo, fat, protein, sugar...)
            - _HEALTH_KW (diabetes, symptom, disease...);
            if (Chứa từ khóa của cả hai nhóm?) then (Có)
                :Intent = BOTH;
            elseif (Chỉ chứa từ khóa Dinh dưỡng?) then (Có)
                :Intent = NUTRITION_LOOKUP;
            else (Không có hoặc chỉ có Y tế)
                :Intent = HEALTH_ADVICE;
            endif
        endif
    fork again
        label ner_start
        if (Mô hình BioBERT NER sẵn sàng?) then (Có)
            if (Độ dài Search Query > 150 từ?) then (Có)
                :Cắt ngắn câu hỏi chỉ giữ lại 150 từ đầu tiên;
            endif
            try
                :Đưa văn bản qua pipeline 'token-classification';
                :BioBERT phân tách tokens và dự đoán nhãn BIO;
                :Sử dụng Aggregation Strategy 'first' để ghép từ;
                :Áp dụng Label Mapping sang các loại thực thể:
                - FOOD, DISEASE, NUTRIENT, SYMPTOM;
            catch (Exception)
                :Kích hoạt Rule-Based spaCy Matcher;
                :So khớp các từ đơn với tập từ khóa định nghĩa sẵn;
            end try
        else (Không)
            :Kích hoạt Rule-Based spaCy Matcher;
            :So khớp các từ đơn với tập từ khóa định nghĩa sẵn;
        endif
    end fork
}

' ==========================================
' PARTITION 3: STRUCTURED NUTRITION RETRIEVAL
' ==========================================
partition "3. Structured Nutrition Retrieval (USDA SQLite)" {
    if (Intent thuộc NUTRITION_LOOKUP hoặc BOTH?) then (Có)
        :Lấy danh sách thực phẩm (FOOD) từ NER;
        if (NER không tìm thấy thực thể FOOD?) then (Có)
            :Dùng spaCy Noun Chunks trích xuất các cụm danh từ FOOD
            (Loại bỏ các từ chung chung như calorie, water, food...);
        endif
        
        if (Có thực phẩm cần tra cứu?) then (Có)
            :Duyệt qua tối đa 3 thực phẩm đầu tiên;
            loop Cho mỗi thực phẩm
                :Tìm kiếm gần đúng (LIKE) trên bảng 'foods';
                note left
                    Thuật toán sắp xếp thứ tự kết quả tìm kiếm:
                    - Loại bỏ thực phẩm chế biến sẵn (hot dog, sausage, salami...)
                    - Ưu tiên thực phẩm cơ bản (foundation_food)
                    - Ưu tiên thực phẩm có nhiều dữ liệu dinh dưỡng nhất
                    - Ưu tiên thực phẩm có mô tả ngắn gọn nhất
                end note
                if (Tìm thấy thực phẩm phù hợp?) then (Có)
                    :Truy vấn bảng 'food_nutrients' để lấy thông số:
                    Protein, Energy, Lipid, Carbs, Fiber trên 100g;
                    :Lưu dữ liệu dinh dưỡng tương ứng;
                endif
            end loop
        endif
    else (Không)
        :Bỏ qua tra cứu SQLite;
    endif
}

' ==========================================
' PARTITION 4: UNSTRUCTURED MEDICAL RETRIEVAL
' ==========================================
partition "4. Unstructured Document Retrieval (Hybrid Search & Reranker)" {
    if (Intent thuộc HEALTH_ADVICE hoặc BOTH?) then (Có)
        fork
            :**Sparse Retrieval (BM25)**
            - Phân tách và lọc stop words trên Search Query
            - Tính toán điểm BM25Okapi trên corpus.jsonl
            - Lấy ra top 20 ứng viên văn bản y khoa
            - Chuẩn hóa điểm số thô về khoảng [0.0, 1.0];
        fork again
            :**Dense Retrieval (ChromaDB)**
            - Nhúng Search Query bằng all-MiniLM-L6-v2
            - Tìm kiếm Cosine Similarity trong Vector Database
            - Lấy ra top 20 ứng viên văn bản y khoa
            - Chuyển Cosine Distance [0, 2] về Similarity [0, 1];
        end fork
        
        :**Reciprocal Rank Fusion (RRF)**
        Hợp nhất 2 danh sách ứng viên (BM25 & Dense):
        Score = Sum( 1 / (10 + Rank + 1) )
        Sắp xếp giảm dần điểm RRF;
        
        :**Cross-Encoder Reranker**
        - Tạo các cặp (Search Query, Document Text)
        - Đưa qua mô hình ms-marco-MiniLM-L-6-v2 (Cross-Attention)
        - Dự đoán độ tương quan ngữ nghĩa trực tiếp
        - Sắp xếp giảm dần theo điểm Rerank
        - Cắt lấy Top K tài liệu y khoa tốt nhất;
    else (Không)
        :Bỏ qua tra cứu y khoa;
    endif
}

' ==========================================
' PARTITION 5: LLM GENERATION
' ==========================================
partition "5. Generation & Guardrails (Ollama)" {
    if (Intent == "NUTRITION_LOOKUP" \n&& Có duy nhất một thực phẩm khớp USDA?) then (Đúng - Fast-path)
        :Bypass LLM hoàn toàn;
        :Định dạng bảng dinh dưỡng thô thành Markdown;
        :Đính kèm nguồn USDA (fdc_id);
        :used_llm = False;
    else (Sai - Chạy LLM)
        :Xây dựng Prompt theo khuôn mẫu (Template):
        1. Ghép dữ liệu dinh dưỡng (Nếu so sánh nhiều món -> Ép dùng Markdown Table)
        2. Ghép tài liệu y khoa tham khảo (Giới hạn 500 ký tự mỗi chunk)
        3. Áp dụng các Guardrails nghiêm ngặt chống ảo giác
        4. Ghép câu hỏi người dùng (Search Query);
        
        :Khởi tạo cấu trúc chat: System prompt, Lịch sử chat, User Prompt;
        :POST /api/chat tới Ollama;
        
        if (Ollama phản hồi thành công?) then (Có)
            :Nhận câu trả lời từ LLM;
            :Loại bỏ các thẻ suy nghĩ <think>...</think> (nếu có);
            :used_llm = True;
        else (Lỗi kết nối / Timeout)
            :Chuyển tiếp sang luồng dự phòng;
            :Ghép thô dữ liệu calo và văn bản y khoa thô thành câu trả lời;
            :used_llm = False;
        endif
    endif
}

:Tổng hợp kết quả cuối cùng: answer, intent, entities, sources;
stop
@endum
```
