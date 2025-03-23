from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput

# Initialize DeepSeek chatbot
bot = Chatbot(
    api_key=None,  # DeepSeek doesn't require an API key for local models
    provider=ChatProvider.DEEPSEEK,
    options={"model_id": "deepseek-ai/DeepSeek-R1"}
)

# Create input
input = ChatModelInput("You are a helpful coding assistant.")
input.add_user_message("Write a Python function to calculate fibonacci numbers.")

# Get response
response = bot.chat(input)
print(response["choices"][0]["text"]) 