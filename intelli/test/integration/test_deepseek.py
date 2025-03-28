from intelli.model.deepseek.wrapper import DeepSeekWrapper
import os
import torch

def test_model():
    model = DeepSeekWrapper(
        repo_id="deepseek-ai/deepseek-llm-7b-chat",
        filename="deepseek-llm-7b-chat.Q4_K_M.gguf",
        model_repo_id="TheBloke/deepseek-llm-7B-chat-GGUF",
        quantized=True
    )
    
    prompt = "Explain what is Python in one sentence:"
    result = model.generate(
        {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }
    )
    print(f"\nPrompt: {prompt}")
    print(f"Response: {result}")
    
    prompts = [
        "What is the capital of France?",
        "Write a haiku about programming:",
        "What is the capital of Italy?"
    ]
    
    inputs_list = [{"prompt": p, "max_tokens": 50} for p in prompts]
    results = model.batch_generate(inputs_list)
    print("\nBatch Generation Results:")
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")

if __name__ == "__main__":
    test_model() 