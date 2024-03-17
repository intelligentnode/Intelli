from intelli.flow.agents.agent import BasicAgent
from intelli.flow.input.agent_input import AgentInput, TextAgentInput, ImageAgentInput
import os  

class KerasAgent(BasicAgent):
    def __init__(self, agent_type, provider="", mission="", model_params={}, options=None, log=False, external=False):
        super().__init__()
        
        # set the parameters
        self.type = agent_type
        self.provider = provider
        self.mission = mission
        self.model_params = model_params
        self.options = options if options is not None else {}
        self.log = log
        self.external = external
        try:
            import keras_nlp
            import keras
            self.nlp_manager = keras_nlp
            self.keras_manager = keras
            if not external:
                self.model = self.load_model()
        except ImportError as e:
            raise ImportError("keras_nlp is not installed or model is not supported.") from e
    
    def set_keras_model(self, model, model_params):
        if not self.external:
            raise Exception("Initiate the agent with external flag to set the model.")
        
        if not hasattr(model, "generate"):
            raise ValueError("The provided model does not have a 'generate' method, which is required for this agent.")
        
        self.model = model
        self.model_params = model_params
    
    def update_model_params(self, model_params):
        
        self.model_params = model_params
        
    def load_model(self):
        """
        Dynamically load a model based on `model_params`.
        This example demonstrates loading Gemma models, but you should add similar logic for other models.
        """
        
        model_name = self.model_params["model"]
        
        # set the username and password
        if "KAGGLE_USERNAME" in self.model_params:
            os.environ["KAGGLE_USERNAME"] = self.model_params["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = self.model_params["KAGGLE_KEY"]
        
        if "gemma" in model_name:
            print("start gemma model")
            return self.nlp_manager.models.GemmaCausalLM.from_preset(model_name)
        elif "mistral" in model_name:
            print("start mistral model")
            return self.nlp_manager.models.MistralCausalLM.from_preset(model_name)
        # ------------------------------------------------------------------ #
        # Add similar conditions for models like Mistral, RoBERTa, or BERT   #
        # ------------------------------------------------------------------ #
        else:
            raise Exception("The received model not supported in this version.")
    
    def execute(self, agent_input: AgentInput, new_params={}):
        """
        Execute the agent task based on input.
        """
        
        if not isinstance(agent_input, TextAgentInput):
            raise ValueError("This agent requires a TextAgentInput.")
        
        custom_params = dict(self.model_params)
        if new_params is not None and isinstance(new_params, dict) and new_params and self.model_params is not None:
            custom_params.update(new_params)
                
        max_length = self.model_params.get("max_length", 100)
        model_input = agent_input.desc if not self.mission else self.mission + ": " + agent_input.desc
            
        if hasattr(self.model, "generate"):     
            if self.log:
                print("Call the model generate with input: ", model_input)
            
            generated_output = self.model.generate(model_input, max_length=max_length)
            
            if isinstance(generated_output, str) and generated_output.startswith(model_input):
                generated_output = generated_output.replace(model_input, "", 1).strip()
            
            return generated_output
        else:
            raise NotImplementedError("Model does not support text generation.")

    def fine_tune_model_with_lora(self, fine_tuning_config, enable_lora=True,
                                  custom_loss=None, custom_metrics=None):
        """
        Finetunes the model as per the provided config.
        """
        print("Fine tuning model...")
        
        # rank=4 reduce the number of trainable parameters.
        lora_rank = fine_tuning_config.get("lora_rank", 4)
        
        # Enable lora for the model learning
        if enable_lora:
            self.model.backbone.enable_lora(rank=lora_rank)
        
        # Set the preprocessor sequence_length
        self.model.preprocessor.sequence_length = fine_tuning_config.get("sequence_length", 512)
        
        # Use AdamW optimizer.
        learning_rate = fine_tuning_config.get("learning_rate", 0.001)
        weight_decay = fine_tuning_config.get("weight_decay", 0.004)
        beta_1 = fine_tuning_config.get("beta_1", 0.9)
        beta_2 = fine_tuning_config.get("beta_2", 0.999)
        
        optimizer = self.keras_manager.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2
        )
        
        # Exclude layernorm and bias terms from decay.
        optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
        
        # Compile the model
        custom_loss = self.keras_manager.losses.SparseCategoricalCrossentropy(from_logits=True) if not custom_loss else custom_loss
        custom_metrics = [self.keras_manager.metrics.SparseCategoricalAccuracy()] if not custom_metrics else custom_metrics
        self.model.compile(
            loss=custom_loss,
            optimizer=optimizer,
            weighted_metrics=custom_metrics,
        )
        
        # Fit using input dataset, epochs and batch size
        dataset = fine_tuning_config.get('dataset')
        epochs = fine_tuning_config.get('epochs', 3)
        batch_size = fine_tuning_config.get("batch_size")
        if batch_size:
            self.model.fit(dataset, epochs=epochs, batch_size=batch_size)
        else:
            self.model.fit(dataset, epochs=epochs)
