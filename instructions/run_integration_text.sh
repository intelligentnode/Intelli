## wrapper tests
# mistral
python -m unittest intelli.test.integration.test_mistralai_wrapper

# gemini
python -m unittest intelli.test.integration.test_geminiai_wrapper

# openai
python -m unittest intelli.test.integration.test_openai_wrapper

# intellicloud
python -m unittest intelli.test.integration.test_intellicloud_wrapper

# stability testing
python -m unittest intelli.test.integration.test_stability_wrapper

# google
python -m unittest intelli.test.integration.test_googleai_wrapper

# anthropic
python -m unittest intelli.test.integration.test_anthropic_wrapper

# wrapper with llama.cpp
pytest -s intelli/test/integration/test_llama_cpp_wrapper.py

## controllers
# embedding
python -m unittest intelli.test.integration.test_remote_embed_model

# images
python -m unittest intelli.test.integration.test_remote_image_model

# vision
python -m unittest intelli.test.integration.test_remote_vision_model

# speech
python -m unittest intelli.test.integration.test_remote_speech_model

## functions
# chatbot
python -m unittest intelli.test.integration.test_chatbot

# chatbot azure
python -m unittest intelli.test.integration.test_azure_chatbot

# chatbot with data
python -m unittest intelli.test.integration.test_chatbot_with_data

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
python -m unittest intelli.test.integration.test_mcp_dataframe_flow


# mcp tools routing
python -m unittest intelli.test.integration.test_flow_mcp_tools
python -m unittest intelli.test.integration.test_chatbot_tools
python -m unittest intelli.test.integration.test_flow_tool_routing

# GPT5
python -m unittest intelli.test.integration.test_chatbot_gpt5 

# azure openai
python -m unittest intelli.test.integration.test_azure_openai_wrapper 
python -m unittest intelli.test.integration.test_azure_whisper_wrapper
python -m unittest intelli.test.integration.test_azure_assistant_wrapper

# vibe
python -m unittest intelli.test.integration.test_vibeflow_simple
python -m unittest intelli.test.integration.test_vibeflow_blog_poste
