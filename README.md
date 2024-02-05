<p align="center">
<img src="assets/flow_logo-round.png" width="200em">
</p>

# IntelliPy
Create chatbots and AI agent workflows with unified access.

# Install
```bash
pip install intelli
```

# Code Examples

## Create AI flows
... WIP ...

## Create Chatbot
... WIP ...


# Connect Your Data 
... WIP ...


# The repository setup
1. Initial setup.
```shell
pip install -r requirements.txt
cd intelli
```

2. Rename `.example.env` to `.env` and fill the keys.

3. Run the test cases, examples below.
```shell
# images
python3 -m unittest test/integration/test_remote_image_model.py

# chatbot
python3 -m unittest test/integration/test_chatbot.py

# mistral
python3 -m unittest test/integration/test_mistralai_wrapper.py
```

# Pillars
- **The wrapper layer** provides low-level access to the latest AI models.
- **The controller layer** offers a unified input to any AI model by handling the differences.
- **The function layer** provides abstract functionality that extends based on the app's use cases. 
- **Flows**: create a flow of ai agents working toward user tasks.
