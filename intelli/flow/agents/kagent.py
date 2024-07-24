from intelli.flow.agents.agent import BasicAgent
from intelli.flow.input.agent_input import AgentInput, TextAgentInput
from intelli.wrappers.keras_wrapper import KerasWrapper

class KerasAgent(BasicAgent):
    def __init__(self, agent_type, provider="", mission="", model_params={}, options=None, log=False, external=False):
        super().__init__()
        self.type = agent_type
        self.provider = provider
        self.mission = mission
        self.model_params = model_params
        self.options = options if options is not None else {}
        self.log = log
        self.external = external
        
        if not external:
            self.wrapper = KerasWrapper(self.model_params["model_name"], self.model_params)
        else:
            self.wrapper = None
    
    def set_keras_model(self, model, model_params):
        if not self.external:
            raise Exception("Initiate the agent with external flag to set the model.")
        if not self.wrapper:
            self.wrapper = KerasWrapper()
        self.wrapper.set_model(model, model_params)
    
    def update_model_params(self, model_params):
        self.model_params = model_params
        if self.wrapper:
            self.wrapper.update_model_params(model_params)
    
    def execute(self, agent_input: AgentInput, new_params={}):
        if not isinstance(agent_input, TextAgentInput):
            raise ValueError("This agent requires a TextAgentInput.")

        custom_params = dict(self.model_params)
        if new_params and isinstance(new_params, dict):
            custom_params.update(new_params)
        
        max_length = custom_params.get("max_length", 180)
        model_input = agent_input.desc if not self.mission else self.mission + ": " + agent_input.desc
        
        if self.wrapper:
            if self.log:
                print("Call the model generate with input: ", model_input)
            generated_output = self.wrapper.generate(model_input, max_length=max_length)
            if isinstance(generated_output, str) and generated_output.startswith(model_input):
                generated_output = generated_output.replace(model_input, "", 1).strip()
            return generated_output
        else:
            raise ValueError("Model wrapper is not set.")

    def fine_tune_model_with_lora(self, fine_tuning_config, enable_lora=True, custom_loss=None, custom_metrics=None):
        if not self.wrapper:
            raise ValueError("Model wrapper is not set.")
        self.wrapper.fine_tune(
            dataset=fine_tuning_config.get('dataset'),
            fine_tuning_config=fine_tuning_config,
            enable_lora=enable_lora,
            custom_loss=custom_loss,
            custom_metrics=custom_metrics
        )
