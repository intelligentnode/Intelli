## wrapper tesers
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


## controllers
# embedding
python3 -m unittest intelli.test.integration.test_remote_embed_model

# images
python3 -m unittest intelli.test.integration.test_remote_image_model

## functions
# chatbot
python3 -m unittest intelli.test.integration.test_chatbot

# chatbot azure
python3 -m unittest intelli.test.integration.test_azure_chatbot

## flows
# basic flow
python3 -m unittest intelli.test.integration.test_flow_sequence
