from intelli.model.input.chatbot_input import ChatModelInput
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper

# Initialize DeepSeek wrapper
wrapper = DeepSeekWrapper(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
print("Loading model...")
wrapper.load_model(device="cpu")
print("Model loaded successfully!")

# Create input for chat
input_prompt = ChatModelInput("You are a helpful coding assistant.")
input_prompt.add_user_message("Write a simple function to add two numbers.")

# Get response
print("\nGenerating response...")
response = wrapper.chat(input_prompt)

print("\nModel response:")
if isinstance(response, dict) and "choices" in response:
    print(response["choices"][0]["text"])
else:
    print("Unexpected response format:", response)