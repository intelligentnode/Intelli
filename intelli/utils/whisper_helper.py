class WhisperHelper:
    def __init__(self, model_name="whisper_tiny_en", backbone=None):
        """
        Initialize once, store imports as instance attributes for optional usage.
        """
        try:
            import numpy as np
            import tensorflow as tf
            import librosa
            import keras_hub as hub
        except ImportError as e:
            raise ImportError(
                "Missing optional libraries"
                "Install via:\n\n  pip install intelli[offline]\n"
            ) from e

        self.np = np
        self.tf = tf
        self.librosa = librosa
        self.hub = hub

        self.model_name = model_name
        self.backbone = (
            backbone
            if backbone
            else self.hub.models.WhisperBackbone.from_preset(model_name)
        )
        self.tokenizer = self.hub.tokenizers.WhisperTokenizer.from_preset(model_name)
        self.converter = self.hub.layers.WhisperAudioConverter.from_preset(model_name)
        self.end_token_id = self.tokenizer.token_to_id("<|endoftext|>")

    def _prepare_audio(self, audio_data, sr, target_sr=16000):
        """
        Downmix stereo and resample if needed to target_sr.
        """
        if audio_data.ndim > 1:
            audio_data = self.np.mean(audio_data, axis=-1)
        if sr != target_sr:
            audio_data = self.librosa.resample(
                audio_data, orig_sr=sr, target_sr=target_sr
            )
        return audio_data.astype("float32")

    def transcribe(
        self,
        audio_data,
        sample_rate=16000,
        language=None,
        max_steps=100,
        min_chunk_sec=20,
        max_chunk_sec=30,
        silence_top_db=40,
    ):
        """
        Transcribe using:
          1) Silence-based segmentation (librosa.effects.split).
          2) Merge segments so each chunk is ideally [20s, 30s] in length.
          3) Transcribe each chunk individually.

        Args:
            audio_data: 1D (or 2D) np.array of raw audio.
            sample_rate: Input sample rate.
            language: Optional language token for multilingual models (e.g. "<|en|>").
            max_steps: Max decoding steps per chunk
            min_chunk_sec: Minimum chunk length (20s).
            max_chunk_sec: Maximum chunk length (30s)
            silence_top_db: Threshold (in dB) below reference to consider silence. Larger = more silence recognized.
        Returns:
            String containing the concatenated transcription of all chunks.
        """
        audio_data = self._prepare_audio(audio_data, sr=sample_rate, target_sr=16000)
        sr = 16000
        segments = self.librosa.effects.split(y=audio_data, top_db=silence_top_db)
        if len(segments) == 0:
            return ""
        min_chunk_samples = int(min_chunk_sec * sr)
        max_chunk_samples = int(max_chunk_sec * sr)
        final_chunks = self._merge_segments(
            segments, audio_data, sr, min_chunk_samples, max_chunk_samples
        )
        results = []
        for start, end in final_chunks:
            chunk_data = audio_data[start:end]
            text = self._transcribe_single_chunk(chunk_data, sr, language, max_steps)
            results.append(text)
        return " ".join(results).strip()

    def _merge_segments(
        self, segments, audio_data, sr, min_chunk_samples, max_chunk_samples
    ):
        """
        Merge consecutive non-silent segments into final chunks of length in [min_chunk_samples, max_chunk_samples].
        """
        final_chunks = []
        current_start = None
        current_end = None
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len > max_chunk_samples:
                if current_start is not None and current_end is not None:
                    final_len = current_end - current_start
                    if final_len > 0:
                        final_chunks.append((current_start, current_end))
                start_pos = seg_start
                while start_pos < seg_end:
                    end_pos = min(start_pos + max_chunk_samples, seg_end)
                    final_chunks.append((start_pos, end_pos))
                    start_pos = end_pos
                current_start = None
                current_end = None
                continue
            if current_start is None:
                current_start = seg_start
                current_end = seg_end
            else:
                extended_end = seg_end
                extended_len = extended_end - current_start
                if extended_len <= max_chunk_samples:
                    current_end = extended_end
                else:
                    chunk_len = current_end - current_start
                    if chunk_len < min_chunk_samples:
                        final_chunks.append((current_start, current_end))
                        current_start = seg_start
                        current_end = seg_end
                    else:
                        final_chunks.append((current_start, current_end))
                        current_start = seg_start
                        current_end = seg_end
        if current_start is not None and current_end is not None:
            final_chunks.append((current_start, current_end))
        return final_chunks

    def _transcribe_single_chunk(
        self, chunk_audio_data, sample_rate=16000, language=None, max_steps=100
    ):
        """
        Transcribe a single chunk (<=30s).
        """
        audio_tensor = self.tf.convert_to_tensor(
            chunk_audio_data, dtype=self.tf.float32
        )[self.tf.newaxis, ...]
        encoder_features = self.converter(audio_tensor)
        start_ids = [self.tokenizer.token_to_id("<|startoftranscript|>")]
        if language:
            try:
                start_ids.append(self.tokenizer.token_to_id(language))
            except KeyError:
                pass
        start_ids.append(self.tokenizer.token_to_id("<|transcribe|>"))
        start_ids.append(self.tokenizer.token_to_id("<|notimestamps|>"))
        decoder_ids = self.tf.constant([start_ids], dtype=self.tf.int32)
        for _ in range(max_steps):
            outputs = self.backbone(
                {
                    "encoder_features": encoder_features,
                    "decoder_token_ids": decoder_ids,
                    "decoder_padding_mask": self.tf.ones_like(decoder_ids),
                }
            )
            logits = self.backbone.token_embedding(
                outputs["decoder_sequence_output"], reverse=True
            )
            next_id = self.tf.argmax(
                logits[:, -1, :], axis=-1, output_type=self.tf.int32
            )
            decoder_ids = self.tf.concat(
                [decoder_ids, next_id[:, self.tf.newaxis]], axis=1
            )
            if self.tf.reduce_any(self.tf.equal(next_id, self.end_token_id)):
                break
        final_ids = decoder_ids[0, len(start_ids) :]
        text = self.tokenizer.detokenize(final_ids)
        return text.replace("<|endoftext|>", "").strip()
