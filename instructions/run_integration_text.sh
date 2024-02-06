## wrapper tesers
# mistral
python3 -m unittest test/integration/test_mistralai_wrapper.py

# gemini
python3 -m unittest test/integration/test_geminiai_wrapper.py

# openai
python3 -m unittest test/integration/test_openai_wrapper.py

# intellicloud
python3 -m unittest test/integration/test_intellicloud_wrapper.py

# stability testing
python3 -m unittest test/integration/test_stability_wrapper.py


## controllers
# embedding
python3 -m unittest test/integration/test_remote_embed_model.py

# images
python3 -m unittest test/integration/test_remote_image_model.py

## functions
# chatbot
python3 -m unittest test/integration/test_chatbot.py

# chatbot azure
python3 -m unittest test/integration/test_azure_chatbot.py

## flows
# basic flow
python3 -m unittest test/integration/test_flow_sequence.py
