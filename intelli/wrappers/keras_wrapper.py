from intelli.utils.whisper_helper import WhisperHelper
import os


class KerasWrapper:
    # credentials read from the environment when a preset is downloaded
    CREDENTIAL_KEYS = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "HF_TOKEN")
    # from_preset options accepted in model_params
    PRESET_KEYS = ("dtype", "load_weights")
    SAMPLERS = {
        "greedy": "GreedySampler",
        "top_k": "TopKSampler",
        "top_p": "TopPSampler",
        "beam": "BeamSampler",
        "random": "RandomSampler",
        "contrastive": "ContrastiveSampler",
    }

    def __init__(self, model_name=None, model_params=None):
        self.model_name = model_name
        self.model_params = model_params
        self.whisper_helper = None
        self.model = None
        self.nlp_manager = None
        self.keras_manager = None
        self._sampler_config = None
        self._lora_enabled = False
        # without a model name the model is provided later using set_model
        if model_name:
            self._load_model()

    def _import_keras(self):
        try:
            import keras

            try:
                import keras_hub as hub
            except ImportError:
                # older releases were published as keras-nlp
                import keras_nlp as hub
        except ImportError as e:
            raise ImportError(
                f"keras-hub is not installed or failed to load ({e}). "
                "Install via:\n\n  pip install intelli[offline]\n"
            ) from e

        self.nlp_manager = hub
        self.keras_manager = keras
        return hub, keras

    def _load_model(self):
        hub, _ = self._import_keras()

        model_params = self.model_params or {}
        for key in self.CREDENTIAL_KEYS:
            if model_params.get(key):
                os.environ[key] = str(model_params[key])

        if "whisper" in self.model_name.lower():
            try:
                backbone = hub.models.WhisperBackbone.from_preset(self.model_name)
                self.whisper_helper = WhisperHelper(
                    model_name=self.model_name, backbone=backbone
                )
            except ImportError:
                raise
            except Exception as e:
                raise ValueError(f"Error loading Whisper model: {e}") from e
        else:
            preset_params = {
                key: model_params[key]
                for key in self.PRESET_KEYS
                if key in model_params
            }
            # the preset can be a built-in name, kaggle://, hf:// handle or a local directory
            self.model = self._get_model_class(hub).from_preset(
                self.model_name, **preset_params
            )

    def _get_model_class(self, hub):
        class_name = (self.model_params or {}).get("model_class")
        if class_name:
            if not hasattr(hub.models, class_name):
                raise ValueError(f"Unsupported model class: {class_name}")
            return getattr(hub.models, class_name)

        model_name = self.model_name.lower()
        if "gemma4" in model_name and "assistant" not in model_name:
            # the gemma4 presets are shared with an assistant only task class
            return getattr(hub.models, "Gemma4CausalLM", hub.models.CausalLM)

        # resolve the task class from the preset (gemma, llama, mistral, gpt2, qwen, etc.)
        return hub.models.CausalLM

    def update_model_params(self, model_params):
        self.model_params = model_params

    def set_model(self, model, model_params):
        self.model = model
        self.model_params = model_params
        self._sampler_config = None
        self._lora_enabled = False

    def _get_prompt_length(self, input_text):
        """
        Count the prompt tokens including the start token, None if the model has no tokenizer.
        """
        try:
            tokenizer = self.model.preprocessor.tokenizer
            prompts = [input_text] if isinstance(input_text, str) else list(input_text)
            return max(len(tokenizer.tokenize(prompt)) for prompt in prompts) + 1
        except Exception:
            return None

    def _set_sampler(self, sampler=None, temperature=None, top_k=None, top_p=None, seed=None):
        """
        Update the model sampler when the sampling options change.
        """
        sampler_config = (sampler, temperature, top_k, top_p, seed)
        if all(value is None for value in sampler_config):
            return
        if sampler_config == self._sampler_config:
            return
        if not hasattr(self.model, "sampler"):
            raise ValueError("The model does not support the sampling options.")

        if sampler is None:
            if temperature == 0:
                sampler = "greedy"
            else:
                sampler = "top_p" if top_p is not None else "top_k"

        if isinstance(sampler, str):
            if sampler not in self.SAMPLERS:
                raise ValueError(
                    f"Unsupported sampler: {sampler}. Send any sampler from: "
                    + " - ".join(self.SAMPLERS)
                )
            hub = self.nlp_manager or self._import_keras()[0]
            sampler_params = {}
            if temperature and sampler != "greedy":
                sampler_params["temperature"] = temperature
            if top_k is not None and sampler in ("top_k", "top_p", "contrastive"):
                sampler_params["k"] = top_k
            if top_p is not None and sampler == "top_p":
                sampler_params["p"] = top_p
            if seed is not None and sampler in ("top_k", "top_p", "random"):
                sampler_params["seed"] = seed
            sampler = getattr(hub.samplers, self.SAMPLERS[sampler])(**sampler_params)

        # keep the compiled optimizer and rebuild the generate function only
        self.model.sampler = sampler
        self.model.generate_function = None
        self._sampler_config = sampler_config

    def _strip_prompt(self, input_text, generated_output):
        if isinstance(generated_output, str):
            if isinstance(input_text, str) and generated_output.startswith(input_text):
                generated_output = generated_output.replace(input_text, "", 1).strip()
            return generated_output

        if isinstance(generated_output, (list, tuple)) and not isinstance(input_text, str):
            return [
                self._strip_prompt(prompt, output)
                for prompt, output in zip(input_text, generated_output)
            ]
        return generated_output

    def generate(
        self,
        input_text,
        max_length=180,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
        sampler=None,
        stop_token_ids="auto",
    ):
        """
        Generate text from a prompt or a list of prompts.

        max_length counts the prompt tokens, send max_new_tokens to limit the generated tokens only.
        The sampling options (temperature, top_k, top_p, seed, sampler) stay active until changed.
        """
        if not self.model:
            raise ValueError("Model is not set.")

        self._set_sampler(sampler, temperature, top_k, top_p, seed)

        prompt_length = self._get_prompt_length(input_text)
        if max_new_tokens:
            max_length = (
                prompt_length + max_new_tokens if prompt_length else max_new_tokens
            )
        elif prompt_length and max_length and prompt_length >= max_length:
            raise ValueError(
                f"The prompt has {prompt_length} tokens and max_length is {max_length}. "
                "Increase max_length or send max_new_tokens."
            )

        generate_params = {"max_length": max_length}
        if stop_token_ids != "auto":
            generate_params["stop_token_ids"] = stop_token_ids

        generated_output = self.model.generate(input_text, **generate_params)
        return self._strip_prompt(input_text, generated_output)

    def fine_tune(
        self,
        dataset,
        fine_tuning_config,
        enable_lora=True,
        custom_loss=None,
        custom_metrics=None,
    ):
        if not self.model:
            raise ValueError("Model is not set.")
        _, keras = self._import_keras()

        fine_tuning_config = fine_tuning_config or {}
        lora_rank = fine_tuning_config.get("lora_rank", 4)
        # lora can be enabled once for the model
        if enable_lora and not self._lora_enabled:
            self.model.backbone.enable_lora(rank=lora_rank)
            self._lora_enabled = True
        self.model.preprocessor.sequence_length = fine_tuning_config.get(
            "sequence_length", 512
        )

        learning_rate = fine_tuning_config.get("learning_rate", 0.001)
        weight_decay = fine_tuning_config.get("weight_decay", 0.004)
        beta_1 = fine_tuning_config.get("beta_1", 0.9)
        beta_2 = fine_tuning_config.get("beta_2", 0.999)

        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2,
        )
        optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

        custom_loss = (
            keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            if not custom_loss
            else custom_loss
        )
        custom_metrics = (
            [keras.metrics.SparseCategoricalAccuracy()]
            if not custom_metrics
            else custom_metrics
        )

        compile_params = {
            "loss": custom_loss,
            "optimizer": optimizer,
            "weighted_metrics": custom_metrics,
        }
        # compile resets the sampler, keep the current one
        if getattr(self.model, "sampler", None) is not None:
            compile_params["sampler"] = self.model.sampler
        self.model.compile(**compile_params)

        epochs = fine_tuning_config.get("epochs", 3)
        batch_size = fine_tuning_config.get("batch_size", 32)
        if isinstance(dataset, (list, tuple, dict)) or hasattr(dataset, "shape"):
            return self.model.fit(dataset, epochs=epochs, batch_size=batch_size)
        # datasets and generators are already batched
        return self.model.fit(dataset, epochs=epochs)

    def save_lora(self, file_path):
        """
        Save the LoRA weights after fine tuning, the file name should end with ".lora.h5".
        """
        if not self.model:
            raise ValueError("Model is not set.")
        self.model.backbone.save_lora_weights(file_path)

    def load_lora(self, file_path, lora_rank=4):
        """
        Load LoRA weights saved with save_lora, lora_rank should match the saved weights.
        """
        if not self.model:
            raise ValueError("Model is not set.")
        if not self._lora_enabled:
            self.model.backbone.enable_lora(rank=lora_rank)
            self._lora_enabled = True
        self.model.backbone.load_lora_weights(file_path)

    def save_preset(self, preset_dir):
        """
        Save the model to a local preset directory, load it again by sending the directory as model_name.
        """
        if not self.model:
            raise ValueError("Model is not set.")
        self.model.save_to_preset(preset_dir)

    def transcript(
        self,
        audio_data,
        sample_rate=16000,
        language=None,
        user_prompt=None,
        condition_on_previous_text=False,
        max_steps=80,
        max_chunk_sec=30,
    ):
        """
        Convert speech to text using the Whisper model.
        """
        if not self.whisper_helper:
            raise ValueError(
                "Whisper is not initialized. Make sure you used a 'whisper_*' model_name."
            )

        return self.whisper_helper.transcribe(
            audio_data=audio_data,
            sample_rate=sample_rate,
            language=language,
            max_steps=max_steps,
            max_chunk_sec=max_chunk_sec,
            user_prompt=user_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )
