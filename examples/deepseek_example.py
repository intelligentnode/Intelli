from intelli.model.input.chatbot_input import ChatModelInput
from intelli.wrappers.deepseek_wrapper import DeepSeekWrapper
from intelli.model.deepseek.deepseek_tokenizer import DeepSeekTokenizer

# Model ID to use
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Using a smaller model for faster loading

# Initialize DeepSeek wrapper directly
wrapper = DeepSeekWrapper(model_id=model_id)
wrapper.load_model(device="cpu")  # Load the model on CPU

# Get direct access to the tokenizer
tokenizer = wrapper.tokenizer

# Test sentence
test_sentence = "Hello, world!"
print(f"Tokenize this sentence: '{test_sentence}'")

# Display tokens in a clean, readable format
if hasattr(tokenizer, 'display_tokens'):
    # Use our new method to display tokens properly
    print(tokenizer.display_tokens(test_sentence))
else:
    print("Tokenizer doesn't have display_tokens method. Using fallback.")
    # Fallback to manual token display
    token_ids = tokenizer.encode(test_sentence)
    print(f"Token IDs: {token_ids}")

    # Try to get token strings
    tokens = []
    for token_id in token_ids:
        if hasattr(tokenizer, 'rev_vocab') and token_id in tokenizer.rev_vocab:
            token_str = tokenizer.rev_vocab[token_id]
            tokens.append(repr(token_str))
        else:
            tokens.append(f"<unknown-{token_id}>")

    print(f"Tokens: {tokens}")

# Print the raw token representation
print("\nRaw token representation:")
token_ids = tokenizer.encode(test_sentence)
if hasattr(tokenizer, 'hf_tokenizer') and tokenizer.hf_tokenizer is not None:
    # Use HF tokenizer's tokens directly
    encoding = tokenizer.hf_tokenizer.encode(test_sentence)
    tokens = encoding.tokens
    # Add BOS token if needed
    if token_ids[0] == tokenizer.bos_token_id and (not tokens or tokens[0] != '<s>'):
        print(f"  BOS token: <s> (ID: {tokenizer.bos_token_id})")

    # Print each token
    for i, token in enumerate(tokens):
        token_id = encoding.ids[i]
        print(f"  Token {i+1}: '{token}' (ID: {token_id})")
else:
    # Fallback to manual token display
    for i, token_id in enumerate(token_ids):
        if token_id in tokenizer.rev_vocab:
            token = tokenizer.rev_vocab[token_id]
            print(f"  Token {i+1}: '{token}' (ID: {token_id})")
        else:
            print(f"  Token {i+1}: <unknown> (ID: {token_id})")

# Test decoding
decoded_text = tokenizer.decode(token_ids)
print(f"\nDecoded text: '{decoded_text}'")

# You can still use the wrapper for chat if needed
try:
    # Create input for chat
    input = ChatModelInput("You are a helpful coding assistant.")
    input.add_user_message("Write a simple function to add two numbers.")

    # Get response
    response = wrapper.chat(input)

    print("\nWrapper response:")
    # Handle response format
    if isinstance(response, dict) and "choices" in response:
        print(response["choices"][0]["text"])
    else:
        print("Unexpected response format:", response)

except Exception as e:
    print(f"Error during chat: {str(e)}")