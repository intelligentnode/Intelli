import os
import base64
import json


class FlowHelper:
    """Utility class for common flow operations like saving/loading outputs"""

    @staticmethod
    def ensure_directory(directory):
        """Ensure a directory exists"""
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_text_output(content, file_path):
        """Save text content to a file"""
        try:
            with open(file_path, "w") as f:
                f.write(content)
            return file_path
        except Exception as e:
            print(f"Error saving text output: {e}")
            return None

    @staticmethod
    def save_audio_output(audio_data, file_path):
        """Save audio data to a file, handling different formats"""
        try:
            # Handle different audio formats
            audio_bytes = None

            if isinstance(audio_data, bytes):
                audio_bytes = audio_data
                print("Using direct binary audio data")
            elif isinstance(audio_data, str) and audio_data.startswith("data:audio"):
                # Handle data URI format
                audio_base64 = audio_data.split(",")[1]
                audio_bytes = base64.b64decode(audio_base64)
                print("Using data URI audio format")
            elif isinstance(audio_data, str) and "base64" in audio_data:
                # Handle raw base64 format
                audio_base64 = audio_data.split("base64,")[-1]
                audio_bytes = base64.b64decode(audio_base64)
                print("Using base64 audio format")
            else:
                try:
                    audio_bytes = base64.b64decode(audio_data) if isinstance(audio_data, str) else audio_data
                    print(f"Used fallback audio handling for type: {type(audio_data)}")
                except Exception as e:
                    print(f"Warning: Could not decode audio data: {e}")
                    audio_bytes = audio_data  # Use as-is as last resort
                    print(f"Using raw audio data of type: {type(audio_data)}")

            # Save to file
            if audio_bytes:
                with open(file_path, "wb") as f:
                    f.write(audio_bytes)
                return file_path, len(audio_bytes)
            return None, 0
        except Exception as e:
            print(f"Error saving audio output: {e}")
            return None, 0

    @staticmethod
    def save_image_output(image_data, file_path):
        """Save image data to a file, handling different formats"""
        try:
            # Convert image data to bytes
            if isinstance(image_data, str):
                # Handle base64 string
                if "," in image_data:
                    image_base64 = image_data.split(",")[1]
                else:
                    image_base64 = image_data
                image_bytes = base64.b64decode(image_base64)
            else:
                # Already bytes
                image_bytes = image_data

            # Save image
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            return file_path, len(image_bytes)
        except Exception as e:
            print(f"Error saving image output: {e}")
            return None, 0

    @staticmethod
    def save_flow_results(results, file_path, exclude_binary=True):
        """Save flow results to a JSON file, optionally excluding binary data"""
        try:
            serializable_results = {}
            for key, value in results.items():
                if exclude_binary and value.get("type") in ["audio", "image"]:
                    # Skip binary data
                    serializable_results[key] = {
                        "type": value["type"],
                        "output": "[BINARY DATA]",
                    }
                else:
                    serializable_results[key] = value

            with open(file_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            return file_path
        except Exception as e:
            print(f"Error saving flow results: {e}")
            return None

    @staticmethod
    def create_sample_outputs(output_map):
        """
        Create sample files for testing if the actual ones weren't generated

        Args:
            output_map: Dict mapping file paths to their expected content type
                       ('text', 'audio', 'image')
        """
        for file_path, content_type in output_map.items():
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                print(f"Creating sample {content_type} file for testing: {file_path}")
                if content_type == 'text':
                    with open(file_path, "w") as f:
                        f.write(f"Sample {os.path.basename(file_path)} for testing")
                elif content_type in ('audio', 'image'):
                    with open(file_path, "wb") as f:
                        f.write(f"dummy {content_type} file".encode())
