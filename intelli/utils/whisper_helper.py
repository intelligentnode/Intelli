import numpy as np
import librosa
import tensorflow as tf
import keras_hub as hub


class WhisperHelper:
    def __init__(self, model_name="whisper_tiny_en", backbone=None):
        self.model_name = model_name
        self.backbone = (
            backbone if backbone else hub.models.WhisperBackbone.from_preset(model_name)
        )
        self.tokenizer = hub.tokenizers.WhisperTokenizer.from_preset(model_name)
        self.converter = hub.layers.WhisperAudioConverter.from_preset(model_name)
        self.end_token_id = self.tokenizer.token_to_id("<|endoftext|>")

    def _prepare_audio(self, audio_data, sr, target_sr=16000):
        """Downmix stereo and resample if needed to target_sr."""
        # If stereo: average channels to produce mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=-1)
        # Resample if sampling rate != 16k
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        return audio_data.astype(np.float32)

    def transcribe(
        self,
        audio_data,
        sample_rate=16000,
        language=None,
        max_steps=100,
        max_chunk_sec=30,
    ):
        """
        Top-level method to handle *any length* audio by splitting into <=30s chunks.

        Args:
            audio_data: 1D (or 2D) numpy array of raw audio samples.
            sample_rate: Sample rate of the input audio_data.
            language: Optional language token (e.g. "<|en|>") if model is multilingual.
            max_steps: Max decoder steps for each chunk.
            max_chunk_sec: The maximum chunk length in seconds (default 30s).
        Returns:
            A concatenated string of all chunk transcriptions.
        """
        # 1) Downmix/resample
        audio_data = self._prepare_audio(audio_data, sr=sample_rate, target_sr=16000)
        sample_rate = 16000  # now we’re certain about the sample rate

        # 2) If audio is short enough, transcribe it directly
        total_length = len(audio_data)
        max_chunk_samples = int(max_chunk_sec * sample_rate)
        if total_length <= max_chunk_samples:
            return self._transcribe_single_chunk(
                audio_data, sample_rate, language, max_steps
            )

        # 3) Otherwise, chunk the audio in 30s blocks
        #    (you can make this more advanced with overlap/stride).
        results = []
        start = 0
        while start < total_length:
            end = min(start + max_chunk_samples, total_length)
            chunk_data = audio_data[start:end]

            chunk_text = self._transcribe_single_chunk(
                chunk_data, sample_rate, language=language, max_steps=max_steps
            )
            results.append(chunk_text)
            start = end

        # 4) Concatenate all partial transcriptions
        return " ".join(results)

    def _transcribe_single_chunk(
        self, chunk_audio_data, sample_rate=16000, language=None, max_steps=100
    ):
        """
        Transcribe a single chunk of audio (<=30s).
        """
        # Convert to tensor shape: (1, samples)
        audio_tensor = tf.convert_to_tensor(chunk_audio_data, dtype=tf.float32)[
            tf.newaxis, ...
        ]

        # Convert to log-mel spectrogram
        encoder_features = self.converter(audio_tensor)

        # Build initial decoder tokens
        # Always: <|startoftranscript|>, [optionally <|en|>], <|transcribe|>, <|notimestamps|>
        start_ids = [self.tokenizer.token_to_id("<|startoftranscript|>")]

        if language:
            try:
                start_ids.append(self.tokenizer.token_to_id(language))
            except KeyError:
                pass  # skip if the token doesn't exist (e.g. English-only model)

        start_ids.append(self.tokenizer.token_to_id("<|transcribe|>"))
        start_ids.append(self.tokenizer.token_to_id("<|notimestamps|>"))

        decoder_ids = tf.constant([start_ids], dtype=tf.int32)

        # Greedy decode
        for _ in range(max_steps):
            outputs = self.backbone(
                {
                    "encoder_features": encoder_features,
                    "decoder_token_ids": decoder_ids,
                    "decoder_padding_mask": tf.ones_like(decoder_ids),
                }
            )

            # Next-token logits
            logits = self.backbone.token_embedding(
                outputs["decoder_sequence_output"], reverse=True
            )
            next_id = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)

            # Append
            decoder_ids = tf.concat([decoder_ids, next_id[:, tf.newaxis]], axis=1)

            # Stop if we see <|endoftext|>
            if tf.reduce_any(tf.equal(next_id, self.end_token_id)):
                break

        # Remove the initial special tokens
        final_ids = decoder_ids[0, len(start_ids) :]

        # Detokenize => python string
        text = self.tokenizer.detokenize(final_ids)

        # Remove leftover <|endoftext|>
        text = text.replace("<|endoftext|>", "").strip()
        return text
