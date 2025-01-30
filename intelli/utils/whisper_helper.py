class WhisperHelper:
    def __init__(self, model_name="whisper_tiny_en", backbone=None):
        """
        Initialize once, store imports as instance attributes for optional usage.
        """
        try:
            import numpy as np
            import tensorflow as tf
            tf.config.optimizer.set_jit(True)
            import librosa
            import keras_hub as hub
        except ImportError as e:
            raise ImportError(
                "Missing optional libraries. "
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

    def _merge_segments(
        self, segments, audio_data, sr, min_chunk_samples, max_chunk_samples
    ):
        """
        Merge consecutive non-silent segments into final chunks of length
        in [min_chunk_samples, max_chunk_samples].
        """
        final_chunks = []
        current_start = None
        current_end = None

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start

            # split if single segment is larger than max_chunk_samples
            if seg_len > max_chunk_samples:
                # in progress, finalize it
                if current_start is not None and current_end is not None:
                    chunk_len = current_end - current_start
                    if chunk_len > 0:
                        final_chunks.append((current_start, current_end))
                # split the big segment
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
                extended_len = seg_end - current_start
                if extended_len <= max_chunk_samples:
                    current_end = seg_end
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

        # leftover chunk
        if current_start is not None and current_end is not None:
            final_chunks.append((current_start, current_end))

        return final_chunks

    def transcribe(
        self,
        audio_data,
        sample_rate=16000,
        language=None,
        max_steps=80,
        min_chunk_sec=20,
        max_chunk_sec=30,
        silence_top_db=40,
        keep_last_n_tokens=80,
        user_prompt=None,
        condition_on_previous_text=False,
    ):
        """
        Transcribe entire audio by:
          1) Splitting on silence
          2) Merging short segments ~[min_chunk_sec, max_chunk_sec]
          3) Decoding each chunk with `_transcribe_single_chunk()`
          4) Optionally carrying prompt context from one chunk to the next.

        Args:
            audio_data: 1D or 2D NumPy array (audio).
            sample_rate: Original sample rate of `audio_data`.
            language: E.g. "<|en|>". If None, you can skip or auto-detect.
            max_steps: Maximum decoding steps per chunk (usually up to ~448 tokens).
            min_chunk_sec, max_chunk_sec: chunk sizes from segments.
            silence_top_db: threshold for silence detection (dB).
            keep_last_n_tokens: prompt context window (if condition_on_previous_text=True).
            user_prompt: (str) optional user-provided prompt for custom vocab.
            condition_on_previous_text: carry context forward across chunks.

        Returns:
            Full transcription as a string.
        """
        audio_data = self._prepare_audio(audio_data, sr=sample_rate, target_sr=16000)
        sr = 16000

        # identify non-silent segments
        segments = self.librosa.effects.split(y=audio_data, top_db=silence_top_db)
        if len(segments) == 0:
            return ""

        # merge small segments
        min_chunk_samples = int(min_chunk_sec * sr)
        max_chunk_samples = int(max_chunk_sec * sr)
        final_chunks = self._merge_segments(
            segments, audio_data, sr, min_chunk_samples, max_chunk_samples
        )

        running_prompt = user_prompt or ""
        results = []

        for start, end in final_chunks:
            chunk_data = audio_data[start:end]

            text = self._transcribe_single_chunk(
                chunk_audio_data=chunk_data,
                sample_rate=sr,
                language=language,
                max_steps=max_steps,
                user_prompt=running_prompt,
            )
            results.append(text)

            # optionally carry forward the newly decoded text
            if condition_on_previous_text and text.strip():
                running_prompt = (running_prompt + " " + text).strip()

                # limit prompt tokens
                if keep_last_n_tokens > 0:
                    last_tokens = self.tokenizer.tokenize(running_prompt)
                    last_tokens = last_tokens[-keep_last_n_tokens:]
                    running_prompt = self.tokenizer.detokenize(last_tokens)

        return " ".join(results).strip()

    
    def _transcribe_single_chunk(
        self,
        chunk_audio_data,
        sample_rate=16000,
        language=None,
        max_steps=80,
        user_prompt=None,
    ):
        """
        Decode a single chunk (<= 30s).
        Inject user_prompt tokens right after the standard start IDs if provided.
        """
        audio_tensor = self.tf.convert_to_tensor(
            chunk_audio_data, dtype=self.tf.float32
        )[self.tf.newaxis, ...]
        encoder_features = self.converter(audio_tensor)

        # basic start tokens
        start_ids = [self.tokenizer.token_to_id("<|startoftranscript|>")]
        if language:
            try:
                # append language token if recognized
                lang_id = self.tokenizer.token_to_id(language)
                if isinstance(lang_id, int):
                    start_ids.append(lang_id)
            except KeyError:
                pass

        start_ids.append(self.tokenizer.token_to_id("<|transcribe|>"))
        start_ids.append(self.tokenizer.token_to_id("<|notimestamps|>"))

        # if the user provided a prompt or previous context
        if user_prompt:
            prompt_ids = self.tokenizer.tokenize(" " + user_prompt.strip())
            # convert to integers
            prompt_ids = [int(pid) for pid in prompt_ids if isinstance(pid, int)]
            start_ids.extend(prompt_ids)

        # final check - everything is an integer
        if any(not isinstance(x, int) for x in start_ids):
            raise ValueError(f"start_ids contains a non-integer. start_ids={start_ids}")

        # convert to TF tensor
        decoder_ids = self.tf.constant([start_ids], dtype=self.tf.int32)

        # autoregressive decoding loop
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
            # append next token
            decoder_ids = self.tf.concat(
                [decoder_ids, next_id[:, self.tf.newaxis]], axis=1
            )

            # break on <|endoftext|>
            if self.tf.reduce_any(self.tf.equal(next_id, self.end_token_id)):
                break

        # slice out generated tokens - ignore the "start_ids"
        final_ids = decoder_ids[0, len(start_ids) :]
        text = self.tokenizer.detokenize(final_ids)
        return text.replace("<|endoftext|>", "").strip()
