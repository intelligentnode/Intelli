## wrapper tesers
# mistral
python3 -m unittest tests/integration/test_mistralai_wrapper.py

# gemini
python3 -m unittest tests/integration/test_geminiai_wrapper.py

# openai
python3 -m unittest tests/integration/test_openai_wrapper.py

# intellicloud
python3 -m unittest tests/integration/test_intellicloud_wrapper.py


# stability testing
python3 -m unittest tests/integration/test_stability_wrapper.py

## controllers
# embedding
python3 -m unittest tests/integration/test_remote_embed_model.py
