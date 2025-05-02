import os
import json
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from huggingface_hub import hf_hub_download

# Try to import the Hugging Face tokenizers library
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Hugging Face tokenizers library not found. Using fallback implementation.")

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
        self.merges = None
        self.eos_token_id = 2  # Default EOS token ID
        self.bos_token_id = 1  # Default BOS token ID
        self.unk_token_id = 0  # Default UNK token ID
        self.rev_vocab = None  # Reverse vocabulary for decoding
        self.hf_tokenizer = None  # Hugging Face tokenizer instance
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
            self.merges = []
            self.rev_vocab = {0: '<unk>', 1: '<s>', 2: '</s>'}

            # Add special handling for common whitespace characters
            self.vocab[' '] = 3
            self.vocab['\t'] = 4
            self.vocab['\n'] = 5
            self.vocab['\r'] = 6
            self.rev_vocab[3] = ' '
            self.rev_vocab[4] = '\t'
            self.rev_vocab[5] = '\n'
            self.rev_vocab[6] = '\r'
            print("Added special handling for whitespace characters")
            return

        print(f"Loading tokenizer from: {tokenizer_file}")

        # Try to use Hugging Face tokenizers library if available
        if TOKENIZERS_AVAILABLE:
            try:
                # Load the tokenizer using Hugging Face tokenizers
                from tokenizers import Tokenizer
                self.hf_tokenizer = Tokenizer.from_file(tokenizer_file)

                # Extract special token IDs
                if hasattr(self.hf_tokenizer, "token_to_id"):
                    # Try to get special token IDs quietly
                    unk_token = self.hf_tokenizer.token_to_id("<unk>")
                    if unk_token is not None:
                        self.unk_token_id = unk_token

                    bos_token = self.hf_tokenizer.token_to_id("<s>")
                    if bos_token is not None:
                        self.bos_token_id = bos_token

                    eos_token = self.hf_tokenizer.token_to_id("</s>")
                    if eos_token is not None:
                        self.eos_token_id = eos_token

                # Also load the vocabulary for compatibility with existing code
                try:
                    # Load the tokenizer data to extract vocabulary
                    with open(tokenizer_file, 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)

                    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                        self.vocab = tokenizer_data["model"]["vocab"]

                        # Check if we have merges for BPE
                        if "merges" in tokenizer_data["model"]:
                            self.merges = [tuple(pair.split()) for pair in tokenizer_data["model"]["merges"]]
                        else:
                            self.merges = []

                        # Create reverse vocabulary for decoding
                        self.rev_vocab = {v: k for k, v in self.vocab.items() if isinstance(v, int)}

                        # Successfully loaded tokenizer
                except Exception as vocab_error:
                    print(f"Error loading vocabulary from tokenizer file: {str(vocab_error)}")
                    # Create minimal vocab for compatibility
                    self.vocab = {'<unk>': self.unk_token_id, '<s>': self.bos_token_id, '</s>': self.eos_token_id}
                    self.rev_vocab = {self.unk_token_id: '<unk>', self.bos_token_id: '<s>', self.eos_token_id: '</s>'}

                return
            except Exception as hf_error:
                print(f"Error loading tokenizer with Hugging Face tokenizers: {str(hf_error)}")
                print("Falling back to custom implementation")
                self.hf_tokenizer = None

        # Fallback to custom implementation if Hugging Face tokenizers is not available
        try:
            # Try UTF-8 encoding first
            with open(tokenizer_file, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)

                # Check if this is a standard HuggingFace tokenizer.json format
                if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                    # Extract vocab and merges for BPE
                    self.vocab = tokenizer_data["model"]["vocab"]

                    # Check if we have merges for BPE
                    if "merges" in tokenizer_data["model"]:
                        self.merges = [tuple(pair.split()) for pair in tokenizer_data["model"]["merges"]]
                        print(f"Loaded BPE tokenizer with {len(self.vocab)} vocab entries and {len(self.merges)} merge rules")
                    else:
                        # No merges found, but we still have a vocabulary
                        self.merges = []
                        print(f"Loaded tokenizer with {len(self.vocab)} vocab entries (no merge rules found)")

                    # Check for special tokens
                    if "added_tokens" in tokenizer_data:
                        for token in tokenizer_data["added_tokens"]:
                            if token.get("special", False):
                                token_content = token.get("content", "")
                                token_id = token.get("id", -1)
                                if token_content == "<unk>" or token_content == "[UNK]":
                                    self.unk_token_id = token_id
                                elif token_content == "<s>" or token_content == "[BOS]":
                                    self.bos_token_id = token_id
                                elif token_content == "</s>" or token_content == "[EOS]":
                                    self.eos_token_id = token_id
                                print(f"Found special token: {token_content} with ID {token_id}")
                else:
                    # Handle other tokenizer formats
                    self.vocab = tokenizer_data
                    self.merges = []
                    print(f"Loaded tokenizer with {len(self.vocab)} vocab entries (no merge rules found)")

                # Create reverse vocabulary for decoding
                self.rev_vocab = {v: k for k, v in self.vocab.items() if isinstance(v, int)}

                # Add whitespace characters to vocabulary if needed
                self._add_whitespace_to_vocab()

        except UnicodeDecodeError:
            # Fall back to latin-1 encoding
            print("UTF-8 encoding failed, trying latin-1 encoding...")
            with open(tokenizer_file, 'r', encoding='latin-1') as f:
                tokenizer_data = json.load(f)

                # Check if this is a standard HuggingFace tokenizer.json format
                if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                    # Extract vocab and merges for BPE
                    self.vocab = tokenizer_data["model"]["vocab"]

                    # Check if we have merges for BPE
                    if "merges" in tokenizer_data["model"]:
                        self.merges = [tuple(pair.split()) for pair in tokenizer_data["model"]["merges"]]
                        print(f"Loaded BPE tokenizer with {len(self.vocab)} vocab entries and {len(self.merges)} merge rules")
                    else:
                        # No merges found, but we still have a vocabulary
                        self.merges = []
                        print(f"Loaded tokenizer with {len(self.vocab)} vocab entries (no merge rules found)")

                    # Check for special tokens
                    if "added_tokens" in tokenizer_data:
                        for token in tokenizer_data["added_tokens"]:
                            if token.get("special", False):
                                token_content = token.get("content", "")
                                token_id = token.get("id", -1)
                                if token_content == "<unk>" or token_content == "[UNK]":
                                    self.unk_token_id = token_id
                                elif token_content == "<s>" or token_content == "[BOS]":
                                    self.bos_token_id = token_id
                                elif token_content == "</s>" or token_content == "[EOS]":
                                    self.eos_token_id = token_id
                                print(f"Found special token: {token_content} with ID {token_id}")
                else:
                    # Handle other tokenizer formats
                    self.vocab = tokenizer_data
                    self.merges = []
                    print(f"Loaded tokenizer with {len(self.vocab)} vocab entries (no merge rules found)")

                # Create reverse vocabulary for decoding
                self.rev_vocab = {v: k for k, v in self.vocab.items() if isinstance(v, int)}

                # Add whitespace characters to vocabulary if needed
                self._add_whitespace_to_vocab()

        except Exception as e:
            # Last resort: try binary mode
            print(f"Error loading tokenizer: {str(e)}")
            print("Trying binary mode as last resort...")
            try:
                with open(tokenizer_file, 'rb') as f:
                    content = f.read()
                    # Try to decode with different encodings
                    for encoding in ['utf-8-sig', 'utf-16', 'cp1252']:
                        try:
                            text = content.decode(encoding)
                            tokenizer_data = json.loads(text)

                            # Check if this is a standard HuggingFace tokenizer.json format
                            if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                                # Extract vocab and merges for BPE
                                self.vocab = tokenizer_data["model"]["vocab"]

                                # Check if we have merges for BPE
                                if "merges" in tokenizer_data["model"]:
                                    self.merges = [tuple(pair.split()) for pair in tokenizer_data["model"]["merges"]]
                                    print(f"Loaded BPE tokenizer with {len(self.vocab)} vocab entries and {len(self.merges)} merge rules")
                                else:
                                    # No merges found, but we still have a vocabulary
                                    self.merges = []
                                    print(f"Loaded tokenizer with {len(self.vocab)} vocab entries (no merge rules found)")

                                # Check for special tokens
                                if "added_tokens" in tokenizer_data:
                                    for token in tokenizer_data["added_tokens"]:
                                        if token.get("special", False):
                                            token_content = token.get("content", "")
                                            token_id = token.get("id", -1)
                                            if token_content == "<unk>" or token_content == "[UNK]":
                                                self.unk_token_id = token_id
                                            elif token_content == "<s>" or token_content == "[BOS]":
                                                self.bos_token_id = token_id
                                            elif token_content == "</s>" or token_content == "[EOS]":
                                                self.eos_token_id = token_id
                                            print(f"Found special token: {token_content} with ID {token_id}")
                            else:
                                # Handle other tokenizer formats
                                self.vocab = tokenizer_data
                                self.merges = []
                                print(f"Loaded tokenizer with {len(self.vocab)} vocab entries (no merge rules found)")

                            # Create reverse vocabulary for decoding
                            self.rev_vocab = {v: k for k, v in self.vocab.items() if isinstance(v, int)}

                            # Add whitespace characters to vocabulary if needed
                            self._add_whitespace_to_vocab()

                            print(f"Successfully loaded with {encoding} encoding")
                            break
                        except:
                            continue
                    else:
                        # If all encodings fail, create a minimal vocab
                        print("All encodings failed, creating minimal vocab")
                        self.vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}
                        self.merges = []
                        self.rev_vocab = {0: '<unk>', 1: '<s>', 2: '</s>'}

                        # Add special handling for common whitespace characters
                        self.vocab[' '] = 3
                        self.vocab['\t'] = 4
                        self.vocab['\n'] = 5
                        self.vocab['\r'] = 6
                        self.rev_vocab[3] = ' '
                        self.rev_vocab[4] = '\t'
                        self.rev_vocab[5] = '\n'
                        self.rev_vocab[6] = '\r'
                        print("Added special handling for whitespace characters")
            except Exception as e2:
                print(f"Failed to load tokenizer: {str(e2)}")
                # Create a minimal vocabulary as fallback
                self.vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}
                self.merges = []
                self.rev_vocab = {0: '<unk>', 1: '<s>', 2: '</s>'}

                # Add special handling for common whitespace characters
                self.vocab[' '] = 3
                self.vocab['\t'] = 4
                self.vocab['\n'] = 5
                self.vocab['\r'] = 6
                self.rev_vocab[3] = ' '
                self.rev_vocab[4] = '\t'
                self.rev_vocab[5] = '\n'
                self.rev_vocab[6] = '\r'
                print("Added special handling for whitespace characters")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using Byte-level BPE.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)

            # Use Hugging Face tokenizer if available
            if TOKENIZERS_AVAILABLE and self.hf_tokenizer is not None:
                # Encode using Hugging Face tokenizer
                encoding = self.hf_tokenizer.encode(text)

                # Get the token IDs
                token_ids = encoding.ids

                # Add BOS token if not already present
                if token_ids and token_ids[0] != self.bos_token_id:
                    token_ids = [self.bos_token_id] + token_ids
                elif not token_ids:
                    token_ids = [self.bos_token_id]

                return token_ids

            # Fallback to custom implementation
            # If we don't have merges, we can't do BPE
            if not self.merges:
                print("No merge rules found, using fallback tokenization")
                # Start with BOS token
                token_ids = [self.bos_token_id]
                # Simple tokenization by splitting on whitespace as fallback
                tokens = text.split()
                # Convert tokens to IDs using vocabulary
                token_ids.extend([self.vocab.get(token, self.unk_token_id) for token in tokens])
                return token_ids

            # Start with BOS token
            token_ids = [self.bos_token_id]

            # Implement proper Byte-level BPE tokenization
            try:
                # Convert text to UTF-8 bytes
                bytes_encoded = text.encode("utf-8")

                # Process bytes using BPE
                # This is a simplified implementation of byte-level BPE
                # For proper implementation, we should use a library like tokenizers

                # Convert bytes to individual characters
                tokens = []
                for b in bytes_encoded:
                    # Convert byte to string representation
                    byte_str = chr(b)
                    tokens.append(byte_str)

                # Apply merges in order
                i = 0
                while i < len(self.merges) and len(tokens) > 1:
                    pair = self.merges[i]
                    a, b = pair

                    # Find all occurrences of the pair and merge them
                    j = 0
                    while j < len(tokens) - 1:
                        if tokens[j] == a and tokens[j+1] == b:
                            tokens[j:j+2] = [a+b]  # Merge the pair
                        else:
                            j += 1

                    i += 1

                # Convert tokens to IDs
                for token in tokens:
                    if token in self.vocab:
                        token_ids.append(self.vocab[token])
                    else:
                        # If token not in vocab, try to encode it byte by byte
                        found = False
                        for char in token:
                            if char in self.vocab:
                                token_ids.append(self.vocab[char])
                                found = True
                            else:
                                token_ids.append(self.unk_token_id)

                        # If we couldn't encode any part of the token, handle it specially
                        if not found:
                            # Handle common whitespace characters
                            if token in [' ', '\t', '\n', '\r']:
                                # Try to find the token in the vocabulary again (it should be there now)
                                if token in self.vocab:
                                    token_ids.append(self.vocab[token])
                                else:
                                    # If still not found, use UNK but don't print a warning
                                    token_ids.append(self.unk_token_id)
                            else:
                                # For non-whitespace tokens, print a warning and use UNK
                                print(f"Unknown token: {repr(token)}")
                                token_ids.append(self.unk_token_id)
            except Exception as bpe_error:
                print(f"Error in BPE encoding: {str(bpe_error)}")
                # Fallback to simple tokenization
                token_ids = [self.bos_token_id]
                tokens = text.split()
                token_ids.extend([self.vocab.get(token, self.unk_token_id) for token in tokens])

            return token_ids

        except Exception as e:
            print(f"Error in encoding: {str(e)}")
            # Return a minimal token sequence as fallback
            return [self.bos_token_id]  # Just return start token

    def _add_whitespace_to_vocab(self):
        """Add common whitespace characters to the vocabulary if they're not already present."""
        # Check if common whitespace characters are in the vocabulary
        whitespace_chars = [' ', '\t', '\n', '\r']

        # Find the next available token ID
        next_id = max(self.rev_vocab.keys()) + 1 if self.rev_vocab else 3

        # Add each whitespace character if it's not already in the vocabulary
        for char in whitespace_chars:
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.rev_vocab[next_id] = char
                print(f"Added whitespace character {repr(char)} to vocabulary with ID {next_id}")
                next_id += 1

    def display_tokens(self, text: str) -> str:
        """Display the tokens for a given text in a clean, readable format.

        Args:
            text: The input text to tokenize

        Returns:
            A string representation of the tokens
        """
        try:
            # Encode the text to get token IDs
            token_ids = self.encode(text)

            # Use Hugging Face tokenizer if available
            if TOKENIZERS_AVAILABLE and self.hf_tokenizer is not None:
                try:
                    # Get the tokens from the Hugging Face tokenizer
                    encoding = self.hf_tokenizer.encode(text)
                    hf_tokens = encoding.tokens

                    # Format the output
                    result = []
                    result.append(f"Input text: '{text}'")
                    result.append(f"Token count: {len(token_ids)}")
                    result.append("Tokens:")

                    # Add BOS token if needed
                    if token_ids[0] == self.bos_token_id and (not hf_tokens or hf_tokens[0] != '<s>'):
                        result.append(f"  1. ID={self.bos_token_id}: <s>")
                        offset = 1
                    else:
                        offset = 0

                    # Display each token with its ID
                    for i, (token_id, token) in enumerate(zip(token_ids[offset:], hf_tokens)):
                        result.append(f"  {i+1+offset}. ID={token_id}: '{token}'")

                    return "\n".join(result)
                except Exception as hf_error:
                    print(f"Error displaying tokens with Hugging Face tokenizer: {str(hf_error)}")
                    print("Falling back to custom implementation")

            # Fallback to custom implementation
            # Get the token strings from the vocabulary
            tokens = []
            for token_id in token_ids:
                if token_id in self.rev_vocab:
                    token_str = self.rev_vocab[token_id]
                    # Clean up the token for display
                    if isinstance(token_str, str) and token_str.startswith('Ġ'):  # This is a space prefix in BBPE
                        token_str = '▁' + token_str[1:]  # Replace with visible space marker
                    # Escape special characters for better display
                    token_str = repr(token_str)
                    tokens.append(token_str)
                else:
                    tokens.append(f"<unknown-{token_id}>")

            # Format the output
            result = []
            result.append(f"Input text: '{text}'")
            result.append(f"Token count: {len(token_ids)}")
            result.append("Tokens:")

            # Display each token with its ID
            for i, (token_id, token_str) in enumerate(zip(token_ids, tokens)):
                result.append(f"  {i+1}. ID={token_id}: {token_str}")

            return "\n".join(result)
        except Exception as e:
            return f"Error displaying tokens: {str(e)}"

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
                token_ids = [self.unk_token_id]

            # Skip special tokens like BOS
            filtered_ids = [id for id in token_ids if id != self.bos_token_id]

            # If we have no tokens after filtering, return empty string
            if not filtered_ids:
                return ""

            # Use Hugging Face tokenizer if available
            if TOKENIZERS_AVAILABLE and self.hf_tokenizer is not None:
                try:
                    # Decode using Hugging Face tokenizer
                    decoded_text = self.hf_tokenizer.decode(filtered_ids)

                    # Clean up the text by replacing special BPE characters
                    decoded_text = decoded_text.replace('Ġ', ' ')  # Replace Ġ with space
                    decoded_text = decoded_text.replace('▁', ' ')  # Replace ▁ with space

                    # Remove any control characters except newlines, tabs, etc.
                    decoded_text = ''.join(ch for ch in decoded_text if ord(ch) >= 32 or ch in '\n\r\t')

                    return decoded_text
                except Exception as hf_error:
                    # Fallback to custom implementation without printing error
                    pass

            # Fallback to custom implementation
            # Get the tokens from the IDs
            tokens = []
            for id in filtered_ids:
                if id in self.rev_vocab:
                    tokens.append(self.rev_vocab[id])
                else:
                    # Add unknown token placeholder without printing
                    tokens.append('<unk>')

            # Filter out special tokens
            filtered_tokens = [t for t in tokens if t not in ['<s>', '</s>']]

            # If we have no tokens after filtering, return empty string
            if not filtered_tokens:
                return ""

            # Process tokens to handle special prefixes
            processed_tokens = []
            for i, token in enumerate(filtered_tokens):
                if isinstance(token, str):
                    # Handle the 'Ġ' prefix which represents a space in BBPE
                    if token.startswith('Ġ'):
                        processed_tokens.append(' ' + token[1:])
                    # If this is not the first token and the previous token didn't add a space
                    # and this token doesn't start with a space marker, add a space
                    elif i > 0 and not filtered_tokens[i-1].startswith('Ġ') and not token.startswith(' '):
                        # Check if we need to add a space based on context
                        # Don't add space after punctuation or at the beginning of sentences
                        prev_token = filtered_tokens[i-1]
                        if not prev_token.endswith(('.', ',', '!', '?', ':', ';', '-', '(', '[', '{', '"', "'")):
                            processed_tokens.append(' ' + token)
                        else:
                            processed_tokens.append(token)
                    else:
                        processed_tokens.append(token)
                else:
                    # Handle non-string tokens (should be rare)
                    processed_tokens.append(str(token))

            # Join tokens
            text = ''.join(processed_tokens)

            # Clean up the text - only remove control characters
            # Remove any control characters except newlines, tabs, etc.
            text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\r\t')

            return text

        except Exception as e:
            # Return a fallback message without printing the error
            return f"Error decoding tokens: {str(e)}"