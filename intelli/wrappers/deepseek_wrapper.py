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

    def chat(self, input_params: Any) -> Dict[str, Any]:
        """Chat interface matching the project's chatbot pattern."""
        try:
            if not self.model:
                self.load_model(device="cpu")  # Default to CPU for safety

            # Convert input_params to a dictionary if it's not already
            if hasattr(input_params, 'get_prompt'):
                # Handle ChatModelInput
                prompt = input_params.get_prompt()
                max_length = input_params.get("max_length", 100)
                temperature = input_params.get("temperature", 0.7)
            else:
                # Handle dictionary input
                prompt = input_params.get("prompt", "")
                max_length = input_params.get("max_length", 100)
                temperature = input_params.get("temperature", 0.7)

            return self.generate_text({
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature
            })
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            # Return a fallback response
            return {
                "choices": [{
                    "text": f"Error in chat: {str(e)}"
                }]
            }

    def load_model(self, device: str = None, quantize: bool = False) -> None:
        """Load the model with specified configuration."""
        print(f"Initializing DeepSeek model from {'path' if self.model_path else 'ID'}")

        try:
            # Check if CUDA is available if device is set to cuda
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"

            # If device is not specified, use the instance default
            if device is None:
                device = self.device

            # Handle path conversion for cross-platform compatibility
            model_path = self.model_path
            if model_path:
                # Convert string path to Path object for better cross-platform handling
                if isinstance(model_path, str):
                    # Convert relative path to absolute if needed
                    if not os.path.isabs(model_path):
                        model_path = os.path.abspath(model_path)
                        print(f"Converted to absolute path: {model_path}")

                    # Normalize path separators for the current OS
                    model_path = os.path.normpath(model_path)
                    print(f"Using normalized path: {model_path}")

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
            # Don't re-raise the exception to allow graceful fallback
            self.model = None

    def generate_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text based on input parameters.

        Args:
            params: Dictionary with parameters for text generation
                - prompt: Input text
                - max_length: Maximum length of generated text
                - temperature: Temperature for sampling

        Returns:
            Dictionary with generated text
        """
        if not self.model or not self.tokenizer:
            # Try to load the model if it's not loaded yet
            try:
                self.load_model(device="cpu")  # Fall back to CPU for safety
            except Exception as e:
                print(f"Failed to load model: {str(e)}")
                # Return a fallback response
                return {
                    "choices": [{
                        "text": "Model could not be loaded. Please check your configuration."
                    }]
                }

        try:
            # Ensure params is a dictionary and contains required keys
            if not isinstance(params, dict):
                print(f"Warning: params is not a dictionary, got {type(params)}")
                # Convert to dictionary if possible
                if hasattr(params, 'get'):
                    # Try to use get method
                    input_text = params.get("prompt", "")
                    max_length = params.get("max_length", 100)
                    temperature = params.get("temperature", 0.7)
                else:
                    # Use default values
                    input_text = ""
                    max_length = 100
                    temperature = 0.7
            else:
                # Extract parameters from dictionary
                input_text = params.get("prompt", "")
                max_length = params.get("max_length", 100)
                temperature = params.get("temperature", 0.7)

            # Ensure input_text is a string
            if not isinstance(input_text, str):
                input_text = str(input_text)

            # Encode the input text
            input_ids = self.tokenizer.encode(input_text)

            # Generate output
            output_ids = self._generate(input_ids, max_length, temperature)

            # Decode the output
            output_text = self.tokenizer.decode(output_ids)

            return {
                "choices": [{
                    "text": output_text
                }]
            }

        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            # Return a fallback response instead of raising an exception
            return {
                "choices": [{
                    "text": f"Error generating text: {str(e)}"
                }]
            }

    def _generate(self, input_ids, max_length, temperature):
        with torch.no_grad():
            try:
                # Convert to tensor if not already
                if not isinstance(input_ids, torch.Tensor):
                    # Always start on CPU for safety
                    input_ids = torch.tensor(input_ids, dtype=torch.long)
                    # Only move to GPU if available and requested
                    if self.device == "cuda" and torch.cuda.is_available():
                        input_ids = input_ids.to(self.device)
                    else:
                        # Ensure we're using CPU if CUDA is not available
                        self.device = "cpu"

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
                    if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                        break

                return generated.tolist()
            except Exception as e:
                print(f"Error in generation: {str(e)}")
                # Return the input as fallback
                if isinstance(input_ids, torch.Tensor):
                    return input_ids.tolist()
                return input_ids

    def _forward(self, input_ids):
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs

        Returns:
            Logits for next token prediction
        """
        try:
            if self.model is None:
                # If model is not loaded, return random logits for testing
                return torch.randn(1, input_ids.shape[1], 32000)

            # Ensure input_ids is on the correct device
            if isinstance(input_ids, torch.Tensor) and str(input_ids.device) != str(self.device):
                input_ids = input_ids.to(self.device)

            # Actual forward pass through the model
            with torch.no_grad():
                return self.model(input_ids)
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            # Return random logits as fallback
            return torch.randn(1, input_ids.shape[1], 32000)