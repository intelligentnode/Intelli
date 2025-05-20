## wrapper tests
# mistral
python3 -m unittest intelli.test.integration.test_mistralai_wrapper

# gemini
python3 -m unittest intelli.test.integration.test_geminiai_wrapper

# openai
python3 -m unittest intelli.test.integration.test_openai_wrapper

# intellicloud
python3 -m unittest intelli.test.integration.test_intellicloud_wrapper

# stability testing
python3 -m unittest intelli.test.integration.test_stability_wrapper

# google
python3 -m unittest intelli.test.integration.test_googleai_wrapper

# anthropic
python3 -m unittest intelli.test.integration.test_anthropic_wrapper

# wrapper with llama.cpp
pytest -s intelli/test/integration/test_llama_cpp_wrapper.py

## controllers
# embedding
python3 -m unittest intelli.test.integration.test_remote_embed_model

# images
python3 -m unittest intelli.test.integration.test_remote_image_model

# vision
python3 -m unittest intelli.test.integration.test_remote_vision_model

# speech
python3 -m unittest intelli.test.integration.test_remote_speech_model

## functions
# chatbot
python3 -m unittest intelli.test.integration.test_chatbot

# chatbot azure
python3 -m unittest intelli.test.integration.test_azure_chatbot

# chatbot with data
python3 -m unittest intelli.test.integration.test_chatbot_with_data

# chatbot with llama.cpp
pytest -s intelli/test/integration/test_chatbot_cpp.py

## flows
# basic flow
python -m unittest intelli.test.integration.test_flow_sequence
# map flow
python -m unittest intelli.test.integration.test_flow_map
# keras nlp
python -m unittest intelli.test.integration.test_keras_agent
# memory
python -m unittest intelli.test.integration.test_flow_memory
python -m unittest intelli.test.integration.test_flow_with_dbmemory


# mcp
python -m unittest intelli.test.integration.test_mcp_openai_flow
python -m unittest intelli.test.integration.
