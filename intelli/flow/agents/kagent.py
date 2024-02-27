from intelli.flow.agents.agent import BasicAgent
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput
import os  

class KerasAgent(BasicAgent):
    def __init__(self, agent_type, provider="", mission="", model_params={}, options=None):
        super().__init__()
        
        # set the parameters
        self.agent_type = agent_type
        self.provider = provider
        self.mission = mission
        self.model_params = model_params
        self.options = options if options is not None else {}
        
        self.model = self.load_model()
    
    def load_model(self):
        """
        Dynamically load a model based on `model_params`.
        This example demonstrates loading Gemma models, but you should add similar logic for other models.
        """
        try:
            # import keras
            import keras_nlp
            self.keras_nlp = keras_nlp 
            model_param = self.model_params['model']
            
            if "gemma" in model_param:
                print('start gemma model')
                from keras_nlp.models import GemmaCausalLM
                
                # set the username and password
                if "KAGGLE_USERNAME" in self.model_params:
                    os.environ["KAGGLE_USERNAME"] = self.model_params["KAGGLE_USERNAME"]
                    os.environ["KAGGLE_KEY"] = self.model_params["KAGGLE_KEY"]
                
                return GemmaCausalLM.from_preset(model_param)
            # ------------------------------------------------------------------ #
            # Add similar conditions for models like Mistral, RoBERTa, or BERT   #
            # ------------------------------------------------------------------ #
            else:
                raise Exception("The received model not supported in this version.")
            
        except ImportError as e:
            raise ImportError("keras_nlp is not installed or model is not supported.") from e
    
    def execute(self, agent_input: AgentInput):
        """
        Execute the agent task based on input.
        """
        
        if not isinstance(agent_input, TextAgentInput):
            raise ValueError("This agent requires a TextAgentInput.")
                
        max_length = self.model_params.get("max_length", 64)
        model_input = agent_input.desc if not self.mission else self.mission + ": " + agent_input.desc
        
        if hasattr(self.model, 'generate'):     
            return self.model.generate(model_input, max_length=max_length)
        else:
            raise NotImplementedError("Model does not support text generation.")
