import os

class KerasWrapper:
    
    def __init__(self, model_name=None, model_params=None):
        self.model_name = model_name
        self.model_params = model_params
        self.model = self.load_model() if model_name else None

    def load_model(self):
        try:
            import keras_nlp
            import keras
            self.nlp_manager = keras_nlp
            self.keras_manager = keras
        except ImportError as e:
            raise ImportError("keras_nlp is not installed or model is not supported.") from e

        if "KAGGLE_USERNAME" in self.model_params:
            os.environ["KAGGLE_USERNAME"] = self.model_params["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = self.model_params["KAGGLE_KEY"]
        
        if "gemma" in self.model_name:
            return self.nlp_manager.models.GemmaCausalLM.from_preset(self.model_name)
        elif "mistral" in self.model_name:
            return self.nlp_manager.models.MistralCausalLM.from_preset(self.model_name)
        elif "llama" in self.model_name:
            return self.nlp_manager.models.Llama3CausalLM.from_preset(self.model_name)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def update_model_params(self, model_params):
        self.model_params = model_params
    
    def set_model(self, model, model_params):
        self.model = model
        self.model_params = model_params

    def generate(self, input_text, max_length=180):
        if not self.model:
            raise ValueError("Model is not set.")
        generated_output = self.model.generate(input_text, max_length=max_length)
        if isinstance(generated_output, str) and generated_output.startswith(input_text):
            generated_output = generated_output.replace(input_text, "", 1).strip()
        return generated_output

    def fine_tune(self, dataset, fine_tuning_config, enable_lora=True, custom_loss=None, custom_metrics=None):
        if not self.model:
            raise ValueError("Model is not set.")
        try:
            import keras_nlp
            import keras
        except ImportError as e:
            raise ImportError("keras_nlp is not installed or model is not supported.") from e

        lora_rank = fine_tuning_config.get("lora_rank", 4)
        if enable_lora:
            self.model.backbone.enable_lora(rank=lora_rank)
        self.model.preprocessor.sequence_length = fine_tuning_config.get("sequence_length", 512)

        learning_rate = fine_tuning_config.get("learning_rate", 0.001)
        weight_decay = fine_tuning_config.get("weight_decay", 0.004)
        beta_1 = fine_tuning_config.get("beta_1", 0.9)
        beta_2 = fine_tuning_config.get("beta_2", 0.999)

        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2
        )
        optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

        custom_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) if not custom_loss else custom_loss
        custom_metrics = [keras.metrics.SparseCategoricalAccuracy()] if not custom_metrics else custom_metrics

        self.model.compile(
            loss=custom_loss,
            optimizer=optimizer,
            weighted_metrics=custom_metrics,
        )

        epochs = fine_tuning_config.get('epochs', 3)
        batch_size = fine_tuning_config.get("batch_size", 32)
        self.model.fit(dataset, epochs=epochs, batch_size=batch_size)

