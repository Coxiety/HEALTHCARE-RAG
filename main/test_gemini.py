import os
import sys

# Ensure project root is on PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.en.pipeline import ENPipeline

def test_rewriter():
    print("--- Testing ENPipeline Query Rewriter ---")
    
    # Check if GEMINI_API_KEY is present
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set!")
        print("We will attempt to initialize, but it will fallback to Ollama if no credentials are found.")

    pipeline = ENPipeline()
    
    print(f"Rewriter Backend: {pipeline.rewriter_backend}")
    print(f"Rewriter Model: {pipeline.rewriter_model}")
    print(f"Generator Backend: {pipeline.generator.backend}")
    print(f"Generator Model: {pipeline.generator.model}")

    mock_history = [
        {"role": "user", "content": "What are the nutrition values of a Banana?"},
        {"role": "model", "content": "Direct Answer: One banana has 105 kcal..."},
        {"role": "user", "content": "Is it good for diabetes?"},
        {"role": "model", "content": "Direct Answer: Yes, fruit consumption..."},
        {"role": "user", "content": "How about chicken breast?"},
        {"role": "model", "content": "Direct Answer: There is no specific information regarding chicken breast and diabetes..."},
    ]
    
    broken_user_input = "Its nutrition values"
    
    print(f"\nOriginal follow-up query: '{broken_user_input}'")
    rewritten = pipeline._condense_query(broken_user_input, mock_history)
    print(f"Rewritten query result: '{rewritten}'")

    print("\n--- Testing Pipeline Answer (Full Flow) ---")
    try:
        result = pipeline.answer(broken_user_input, mock_history)
        print(f"Answer Intent: {result.get('intent')}")
        print(f"Answer Entities: {result.get('entities')}")
        print(f"Answer Sources: {result.get('sources')}")
        print(f"Used LLM: {result.get('used_llm')}")
        print(f"Answer Text:\n{result.get('answer')}")
    except Exception as e:
        print(f"Error executing answer: {e}")

if __name__ == "__main__":
    test_rewriter()
