from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput

# Initialize DeepSeek chatbot
bot = Chatbot(
    api_key=None,  # DeepSeek doesn't require an API key for local models
    provider=ChatProvider.DEEPSEEK,
    options={
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Using a smaller model for faster loading
        # Uncomment the line below to specify a local model path
        # "model_path": "./path/to/your/model"
    }
)

# Create input
input = ChatModelInput("You are a helpful coding assistant.")
input.add_user_message("Write a Python function to calculate fibonacci numbers.")

# Get response
try:
    response = bot.chat(input)

    # Handle different response formats
    if isinstance(response, dict) and "choices" in response:
        # Direct DeepSeekWrapper response format
        print(response["choices"][0]["text"])
    elif isinstance(response, list) and len(response) > 0:
        # Chatbot class response format
        print(response[0])
    else:
        print("Unexpected response format:", response)

except Exception as e:
    print(f"Error during chat: {str(e)}")