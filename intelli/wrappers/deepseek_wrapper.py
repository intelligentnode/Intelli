from typing import Dict, Any, Optional
import os
import torch
from intelli.model.deepseek.deepseek_loader import DeepSeekLoader
from intelli.model.deepseek.deepseek_tokenizer import DeepSeekTokenizer

class DeepSeekWrapper:
    """Wrapper for DeepSeek models following Intelli's wrapper pattern."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 model_id: Optional[str] = None,
                 api_key: Optional[str] = None): # api_key kept for compatibility
        self.model_path = model_path
        self.model_id = model_id
        self.model = None
        self.loader = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def chat(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        """Chat interface matching the project's chatbot pattern."""
        if not self.model:
            self.load_model()

        prompt = input_params.get("prompt", "")
        return self.generate_text({
            "prompt": prompt,
            "max_length": input_params.get("max_length", 100),
            "temperature": input_params.get("temperature", 0.7)
        })

    def load_model(self, device: str = "cuda", quantize: bool = False) -> None:
        """Load the model with specified configuration."""
        print(f"Initializing DeepSeek model from {'path' if self.model_path else 'ID'}")

        try:
            # Convert relative path to absolute if needed
            model_path = self.model_path
            if model_path and not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
                print(f"Using absolute path: {model_path}")

            self.loader = DeepSeekLoader(
                model_path=model_path,
                model_id=self.model_id,
                device=device,
                quantize=quantize
            )
            self.model = self.loader.load_model()
            self.tokenizer = DeepSeekTokenizer(
                model_path=str(self.loader.model_path),
                model_id=self.model_id
            )
            self.device = device

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def generate_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text based on input parameters."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not loaded. Call load_model() first.")

        try:
            input_text = params.get("prompt", "")
            input_ids = self.tokenizer.encode(input_text)

            max_length = params.get("max_length", 100)
            temperature = params.get("temperature", 0.7)

            output_ids = self._generate(input_ids, max_length, temperature)
            output_text = self.tokenizer.decode(output_ids)

            return {
                "choices": [{
                    "text": output_text
                }]
            }

        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            raise

    def _generate(self, input_ids, max_length, temperature):
        with torch.no_grad():
            # Convert to tensor if not already
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)

            # Initialize with input_ids
            generated = input_ids.clone()

            # Simple autoregressive generation
            for _ in range(max_length):
                # Get logits for next token
                logits = self._forward(generated)

                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature

                # Sample from distribution
                probs = torch.softmax(logits[:, -1], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)

                # Check for end of sequence token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            return generated.tolist()

    def _forward(self, input_ids):
        # This is a placeholder for the actual forward pass
        # In a real implementation, this would use the loaded model
        return torch.randn(1, input_ids.shape[1], 32000)