"""
Test GPT-5 features using local development version of Intelli
This script adds the local intelli package to the Python path before importing
"""

import sys
import os

# Add the parent directory to Python path to use local development version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from local version
from dotenv import load_dotenv
from intelli.function.chatbot import Chatbot, ChatProvider
from intelli.model.input.chatbot_input import ChatModelInput

# Load environment variables
load_dotenv()

# Get OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def main():
    print("🚀 Testing GPT-5 with local Intelli version\n")
    
    # Initialize chatbot
    chatbot = Chatbot(api_key=OPENAI_KEY, provider=ChatProvider.OPENAI)
    
    # Test 1: Basic GPT-5 usage
    print("1️⃣ Basic GPT-5 Test:")
    chat_input = ChatModelInput("You are a helpful assistant.")
    chat_input.add_user_message("What is the capital of France?")
    
    response = chatbot.chat(chat_input)
    print(f"Response: {response[0]}\n")
    
    # Test 2: GPT-5 with reasoning effort
    print("2️⃣ GPT-5 with medium reasoning effort:")
    chat_input = ChatModelInput(
        "You are a problem solver.",
        model="gpt-5",
        reasoning_effort="medium"
    )
    chat_input.add_user_message(
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
    )
    
    response = chatbot.chat(chat_input)
    print(f"Response: {response[0]}\n")
    
    # Test 3: GPT-5 with verbosity (string)
    print("3️⃣ GPT-5 with verbosity (string 'high'):")
    chat_input = ChatModelInput(
        "You are a creative writer.",
        model="gpt-5",
        reasoning_effort="low",
        verbosity="high"
    )
    chat_input.add_user_message("Write a haiku about coding.")
    
    response = chatbot.chat(chat_input)
    print(f"Response: {response[0]}\n")
    
    # Test 4: GPT-5 with verbosity (integer - tests conversion)
    print("4️⃣ GPT-5 with verbosity (integer 2 -> 'medium'):")
    chat_input = ChatModelInput(
        "You are a helpful assistant.",
        model="gpt-5",
        reasoning_effort="minimal",
        verbosity=2  # Should be converted to 'medium'
    )
    chat_input.add_user_message("What is 1+1?")
    
    response = chatbot.chat(chat_input)
    print(f"Response: {response[0]}\n")
    
    print("✅ All GPT-5 tests completed successfully with local version!")

if __name__ == "__main__":
    if not OPENAI_KEY:
        print("❌ Error: Please set OPENAI_API_KEY in your .env file")
    else:
        main()
