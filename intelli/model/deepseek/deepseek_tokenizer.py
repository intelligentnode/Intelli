import os
import json
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from huggingface_hub import hf_hub_download

class DeepSeekTokenizer:
    def __init__(self, model_path: Optional[str] = None, model_id: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        self.model_id = model_id

        # Handle path conversion
        if model_path:
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
            self.model_path = Path(model_path)
        else:
            self.model_path = None

        self.vocab = None
        self.eos_token_id = 2  # Default EOS token ID
        self.load_vocab()

    def load_vocab(self):
        if not self.model_path:
            cache_dir = Path.home() / ".cache" / "intelli" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)

            vocab_file = hf_hub_download(
                repo_id=self.model_id,
                filename="tokenizer.json",
                cache_dir=cache_dir
            )
            self.model_path = Path(vocab_file).parent

        # Find tokenizer file
        tokenizer_file = None

        # First, look for files with 'tokenizer' in the name
        for root, _, files in os.walk(self.model_path):
            for file in files:
                if 'tokenizer' in file and file.endswith('.json'):
                    tokenizer_file = os.path.join(root, file)
                    break
            if tokenizer_file:
                break

        # If not found, look in subdirectories
        if not tokenizer_file:
            print(f"Tokenizer file not found in {self.model_path}, searching subdirectories...")
            for root, dirs, _ in os.walk(self.model_path):
                for dir_name in dirs:
                    subdir = os.path.join(root, dir_name)
                    for subroot, _, subfiles in os.walk(subdir):
                        for file in subfiles:
                            if 'tokenizer' in file and file.endswith('.json'):
                                tokenizer_file = os.path.join(subroot, file)
                                print(f"Found tokenizer file in subdirectory: {tokenizer_file}")
                                break
                        if tokenizer_file:
                            break
                    if tokenizer_file:
                        break
                if tokenizer_file:
                    break

        # If still not found, try any JSON file that might be a tokenizer
        if not tokenizer_file:
            print("No tokenizer file found. Looking for any JSON file that might be a tokenizer...")
            for root, _, files in os.walk(self.model_path):
                for file in files:
                    if file.endswith('.json') and ('vocab' in file.lower() or 'token' in file.lower() or 'dict' in file.lower()):
                        tokenizer_file = os.path.join(root, file)
                        print(f"Using {tokenizer_file} as tokenizer file")
                        break
                if tokenizer_file:
                    break

        # Last resort: use any JSON file
        if not tokenizer_file:
            print("No tokenizer-like file found. Using any JSON file as fallback...")
            for root, _, files in os.walk(self.model_path):
                for file in files:
                    if file.endswith('.json'):
                        tokenizer_file = os.path.join(root, file)
                        print(f"Using {tokenizer_file} as tokenizer file (fallback)")
                        break
                if tokenizer_file:
                    break

        if not tokenizer_file:
            print(f"Tokenizer file not found in {self.model_path}. Creating a minimal tokenizer.")
            # Create a minimal vocabulary as fallback
            self.vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}
            return

        print(f"Loading tokenizer from: {tokenizer_file}")
        try:
            # Try UTF-8 encoding first
            with open(tokenizer_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        except UnicodeDecodeError:
            # Fall back to latin-1 encoding
            print("UTF-8 encoding failed, trying latin-1 encoding...")
            with open(tokenizer_file, 'r', encoding='latin-1') as f:
                self.vocab = json.load(f)
        except Exception as e:
            # Last resort: try binary mode
            print(f"Error loading tokenizer: {str(e)}")
            print("Trying binary mode as last resort...")
            try:
                with open(tokenizer_file, 'rb') as f:
                    import io
                    content = f.read()
                    # Try to decode with different encodings
                    for encoding in ['utf-8-sig', 'utf-16', 'cp1252']:
                        try:
                            text = content.decode(encoding)
                            self.vocab = json.loads(text)
                            print(f"Successfully loaded with {encoding} encoding")
                            break
                        except:
                            continue
                    else:
                        # If all encodings fail, create a minimal vocab
                        print("All encodings failed, creating minimal vocab")
                        self.vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}
            except Exception as e2:
                print(f"Failed to load tokenizer: {str(e2)}")
                # Create a minimal vocabulary as fallback
                self.vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)

            # Simple tokenization by splitting on whitespace
            # In a real implementation, this would use a more sophisticated tokenizer
            tokens = text.split()

            # Convert tokens to IDs using vocabulary
            # Default to unknown token ID (0) if token not in vocabulary
            return [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
        except Exception as e:
            print(f"Error in encoding: {str(e)}")
            # Return a single token ID as fallback
            return [0]

    def decode(self, token_ids: Union[List[int], List[List[int]], torch.Tensor]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode, can be nested lists or tensors

        Returns:
            Decoded text
        """
        try:
            # Ensure token_ids is a flat list of integers
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()

            # Handle nested lists
            if isinstance(token_ids, list):
                # Check if it's a nested list
                if token_ids and isinstance(token_ids[0], list):
                    # Flatten the list
                    flat_ids = []
                    for sublist in token_ids:
                        if isinstance(sublist, list):
                            flat_ids.extend(sublist)
                        else:
                            flat_ids.append(sublist)
                    token_ids = flat_ids
                # If it's still not a list of integers, try to convert
                if token_ids and not isinstance(token_ids[0], int):
                    token_ids = [int(id) for id in token_ids]
            else:
                # Default to a single unknown token
                token_ids = [0]

            # Create reverse vocabulary mapping (ID -> token)
            rev_vocab = {v: k for k, v in self.vocab.items() if isinstance(v, int)}

            # Convert IDs to tokens and join with spaces
            return ' '.join(rev_vocab.get(id, '<unk>') for id in token_ids)
        except Exception as e:
            print(f"Error in decoding: {str(e)}")
            # Return a fallback message
            return f"Error decoding tokens: {str(e)}"