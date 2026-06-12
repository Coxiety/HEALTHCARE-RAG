# HEALTHCARE-RAG LLM Generation Workflow

Tài liệu này mô tả chi tiết cách thức hoạt động, prompt engineering, kiểm soát an toàn dữ liệu (Guardrails), cơ chế đường tắt (Fast-path) và hệ thống dự phòng (Fallback) của cấu phần **LLM Generation** (`Generator`) trong hệ thống **HEALTHCARE-RAG**.

---

## 1. Sơ đồ Luồng xử lý Sinh câu trả lời (Activity Diagram)

Mô tả chi tiết cách hệ thống quyết định đi theo nhánh sinh lời giải trực tiếp (Fast-path), gọi mô hình LLM, hay chuyển tiếp sang cơ chế dự phòng (Rule-based Fallback).

```plantuml
@startuml
skinparam ActivityBackgroundColor #F2F4F7
skinparam ActivityBorderColor #333333
skinparam ActivityFontSize 12
skinparam ArrowColor #4A90E2
skinparam NoteBackgroundColor #FFF9E6

start
:Nhận đầu vào: Standalone Query, Dữ liệu dinh dưỡng (USDA), 
Tài liệu y khoa (Chunks), Intent, Lịch sử chat (History);

partition "1. Fast-Path Check (Kiểm tra đường tắt)" {
    if (Intent == "NUTRITION_LOOKUP"\n&& Có duy nhất một thực phẩm khớp dữ liệu USDA?) then (Đúng - Fast Path)
        :Gọi _format_nutrition_answer(nutrition_data);
        note left
            Bypass LLM hoàn toàn để tối ưu hóa chi phí và độ trễ.
            Format thông số Protein, Lipid, Carbs, Calories, Fiber
            thành Markdown thô và đính kèm nguồn USDA (fdc_id).
        end note
        :used_llm = False;
        :Trả về kết quả trực tiếp;
        stop
    else (Sai)
        :Đi tiếp tới luồng sinh prompt bằng LLM;
    endif
}

partition "2. Prompt Builder & Guardrails" {
    :Khởi tạo danh sách các phần của Prompt (sections);
    
    if (Cần thông tin dinh dưỡng?) then (Có)
        if (Có dữ liệu dinh dưỡng?) then (Có)
            if (Có nhiều hơn 1 thực phẩm (So sánh)?) then (Có)
                :Format dữ liệu dinh dưỡng cho từng món;
                :Chèn chỉ thị **Comparison Instruction** bắt buộc:
                - Phải sử dụng bảng so sánh Markdown (Markdown Table)
                - Cột: Nutrient, Tên món 1, Tên món 2...
                - Trình bày toàn bộ dưỡng chất thiết yếu per 100g;
            else (Không - 1 thực phẩm)
                :Format dữ liệu dinh dưỡng cho thực phẩm đơn lẻ;
            endif
            :Đóng gói dữ liệu cứng cùng chỉ thị khóa:
            "IMPORTANT: Use ONLY these exact values. Do NOT use other numbers.";
        else (Không có dữ liệu USDA)
            :Chèn chỉ thị thông báo cho người dùng không tìm thấy dữ liệu dinh dưỡng,
            cấm LLM tự bịa, đoán hoặc ước lượng các con số;
        endif
    endif
    
    if (Cần thông tin y khoa && Có tài liệu chunks?) then (Có)
        :Trích xuất tối đa 500 ký tự đầu của mỗi chunk văn bản y học;
        :Ghép các nguồn trích dẫn: [Reference Documents]\n[Source_1] text...;
    endif
    
    :Gộp các chỉ thị hệ thống (System Instructions):
    - Trả lời tự nhiên bằng tiếng Anh
    - Cấm đề cập các luật lệ prompt hay meta-comments
    - Bắt buộc dựa trên tài liệu y khoa và trích nguồn cụ thể
    - Nếu không đủ thông tin y khoa, phải từ chối trả lời (chống ảo giác);
}

partition "3. LLM API Call" {
    :Dựng cấu trúc chat messages:
    1. System Message (Tính cách của chuyên gia dinh dưỡng)
    2. History Messages (Lịch sử cuộc hội thoại)
    3. User Message (Prompt vừa dựng);
    
    :POST /api/chat đến Ollama (llama3.1:8b, temperature=0.3, num_ctx=4096);
    
    if (Kết nối thành công && Trả về text?) then (Có)
        :used_llm = True;
        :Gọi _strip_thinking(answer) để xóa bỏ các thẻ <think>...</think>
        (Đảm bảo an toàn cho các mô hình Reasoning như DeepSeek R1);
    else (Lỗi kết nối / Ollama tắt)
        partition "4. Graceful Fallback" {
            :used_llm = False;
            :Gọi _fallback_answer(query, nutrition_data, health_chunks, intent);
            note right
                Bộ xử lý dự phòng tự động lắp ghép dữ liệu thô:
                - Text calo/dinh dưỡng từ USDA SQLite
                - Văn bản y khoa thô trích xuất trực tiếp từ vector store
                Đảm bảo giao diện người dùng luôn nhận được thông tin.
            end note
        }
    endif
}

:Hợp nhất nguồn trích dẫn (sources):
- File corpus y khoa
- Mã USDA fdc_id;
:Trả về dict kết quả: {answer, sources, used_llm};
stop
@endum
```

---

## 2. Quy trình Tương tác với Ollama và Fallback (Sequence Diagram)

Mô tả sự tương tác giữa RAG Pipeline, Generator, Máy chủ Ollama cục bộ và cơ chế bảo vệ dự phòng khi xảy ra sự cố.

```plantuml
@startuml
autonumber
participant "ENPipeline" as Pipe
participant "Generator" as Gen
participant "Ollama API\n(/api/chat)" as Ollama
database "Fallback Processor" as Fallback

Pipe -> Gen : generate(query, nutrition, chunks, intent, history)
activate Gen

Gen -> Gen : build_prompt()
note left
    - Ghép dữ liệu dinh dưỡng USDA
    - Ghép tài liệu y khoa tham khảo
    - Áp dụng 4 Guardrails chống ảo giác
end note

Gen -> Gen : Dựng cấu trúc hội thoại [System, History, User Prompt]

alt Tiến trình chạy LLM bình thường
    Gen -> Ollama : POST /api/chat {model: llama3.1:8b, messages, stream: false}
    activate Ollama
    Ollama --> Gen : HTTP 200 OK {"message": {"content": "<think>...</think> Answer content"}}
    deactivate Ollama
    
    Gen -> Gen : _strip_thinking(content)
    Gen --> Pipe : Trả về {"answer": "Answer content", "sources": [...], "used_llm": true}
else Ollama bị sập hoặc quá tải (Timeout/Network Error)
    Gen -> Ollama : POST /api/chat (Thất bại)
    activate Ollama
    note right of Ollama: Không phản hồi hoặc lỗi
    Ollama --/ Gen : Connection Refused / Timeout
    deactivate Ollama
    
    Gen -> Fallback : Gọi _fallback_answer()
    activate Fallback
    Fallback -> Fallback : Định dạng dữ liệu thô USDA và văn bản thô từ Chunks
    Fallback --> Gen : Trả về câu trả lời dạng văn bản cấu trúc thô
    deactivate Fallback
    
    Gen --> Pipe : Trả về {"answer": "Fallback content", "sources": [...], "used_llm": false}
end
deactivate Gen
@endum
```

---

## 3. Cấu trúc Lớp Generator (Class Diagram)

```plantuml
@startuml
class Generator {
    + model: str
    + host: str
    + __init__(model: str, host: str)
    + generate(query: str, nutrition_data: list|dict|None, health_chunks: list, query_type: str, history: list): dict
    + build_prompt(query: str, nutrition_data: list|dict|None, health_chunks: list, query_type: str): str
    - _call_ollama(prompt: str, history: list): str | None
    - _call_ollama_generate(prompt: str): str | None
    - _format_single_nutrition_data(data: dict): str
    - _format_nutrition_answer(nutrition_data: dict): str
    {static} - _strip_thinking(text: str): str
    {static} - _fallback_answer(query: str, nutrition_data: list|dict|None, health_chunks: list, query_type: str): str
}

class ENPipeline {
    - generator: Generator
    + answer(query: str, history: list): dict
}

ENPipeline *-- Generator
@endum
```
