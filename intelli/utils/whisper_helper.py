import numpy as np
import librosa
import tensorflow as tf
import keras_hub as hub

class WhisperHelper:
    """Helper class for Whisper speech-to-text inference."""
    
    def __init__(self, model_name="whisper_tiny_en", backbone=None):
        self.model_name = model_name
        self.backbone = backbone if backbone else hub.models.WhisperBackbone.from_preset(model_name)
        self.tokenizer = hub.tokenizers.WhisperTokenizer.from_preset(model_name)
        self.converter = hub.layers.WhisperAudioConverter.from_preset(model_name)
        self.end_token_id = self.tokenizer.token_to_id("<|endoftext|>")

    def _prepare_audio(self, audio_data, sr, target_sr=16000):
        """Resample audio and convert to float32 numpy array."""
        if sr != target_sr:
            audio_data = librosa.resample(np.array(audio_data), orig_sr=sr, target_sr=target_sr)
        return audio_data.astype(np.float32)

    def transcribe(self, audio_data, sample_rate=16000, language="<|en|>", max_steps=100):
        """Convert audio to text using greedy decoding."""
        # Preprocess audio
        audio_data = self._prepare_audio(audio_data, sample_rate)
        audio_tensor = tf.convert_to_tensor(audio_data)[tf.newaxis, ...]
        
        # Extract features
        encoder_features = self.converter(audio_tensor)
        
        # Initialize decoder with special tokens
        start_ids = [
            self.tokenizer.token_to_id("<|startoftranscript|>"),
            self.tokenizer.token_to_id(language),
            self.tokenizer.token_to_id("<|transcribe|>"),
            self.tokenizer.token_to_id("<|notimestamps|>"),
        ]
        decoder_ids = tf.constant([start_ids], dtype=tf.int32)

        # Greedy decoding loop
        for _ in range(max_steps):
            outputs = self.backbone({
                "encoder_features": encoder_features,
                "decoder_token_ids": decoder_ids,
                "decoder_padding_mask": tf.ones_like(decoder_ids),
            })
            logits = self.backbone.token_embedding(outputs["decoder_sequence_output"], reverse=True)
            next_id = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
            decoder_ids = tf.concat([decoder_ids, tf.expand_dims(next_id, 1)], axis=1)
            
            if tf.reduce_any(tf.equal(next_id, self.end_token_id)):
                break

        # Convert tokens to text
        final_ids = decoder_ids[0, 4:]  # Skip initial special tokens
        return self.tokenizer.detokenize(final_ids).numpy().decode("utf-8").strip()
    