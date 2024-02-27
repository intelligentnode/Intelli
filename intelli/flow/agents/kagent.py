from intelli.flow.agents.agent import BasicAgent
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput

class KerasAgent(BasicAgent):
    def __init__(self, agent_type, provider, mission, model_params, options=None):
        super().__init__(agent_type, provider, mission, model_params, options)
        try:
            import keras_nlp
            self.keras_nlp = keras_nlp 
            self.KERAS_AVAILABLE = True
        except ImportError:
            self.KERAS_AVAILABLE = False
    
    def execute(self, agent_input: AgentInput):
        if self.KERAS_AVAILABLE:
            # todo add the execute logic
            pass
        else:
            raise Exception("keras_nlp is not available. This function cannot proceed.")