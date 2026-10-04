class WhisperHelper:
    # whisper keeps at most half of the 448 decoder positions for the prompt
    MAX_PROMPT_TOKENS = 223

    def __init__(self, model_name="whisper_tiny_en", backbone=None):
        """
        Initialize once, store imports as instance attributes for optional usage.
        """
        try:
            import numpy as np
            import keras
            import librosa

            try:
                import keras_hub as hub
            except ImportError:
                # older releases were published as keras-nlp
                import keras_nlp as hub
        except ImportError as e:
            raise ImportError(
                "Missing optional libraries. "
                "Install via:\n\n  pip install intelli[offline]\n"
            ) from e

        self.np = np
        self.keras = keras
        self.ops = keras.ops
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

        self.start_token_id = int(self.tokenizer.token_to_id("<|startoftranscript|>"))
        self.end_token_id = int(self.tokenizer.token_to_id("<|endoftext|>"))
        self.transcribe_token_id = int(self.tokenizer.token_to_id("<|transcribe|>"))
        self.no_timestamps_token_id = int(
            self.tokenizer.token_to_id("<|notimestamps|>")
        )
        # <|startofprev|> is not exposed by the tokenizer, it sits two ids before <|notimestamps|>
        prev_token_id = self._special_token_id("<|startofprev|>")
        self.prev_token_id = (
            prev_token_id
            if prev_token_id is not None
            else self.no_timestamps_token_id - 2
        )
        # english only models have no language tokens and no task token in the start sequence
        self.is_multilingual = bool(getattr(self.tokenizer, "language_tokens", None))
        self.max_decoder_length = int(
            getattr(self.backbone, "max_decoder_sequence_length", 448)
        )
        self.encoder = self._build_encoder()

    def _to_numpy(self, value):
        """
        Convert a backend tensor to a NumPy array for any keras backend.
        """
        try:
            return self.np.asarray(self.ops.convert_to_numpy(value))
        except Exception:
            return self.np.asarray(value)

    def _special_token_id(self, token):
        """
        Return the id of a special token, or None when the tokenizer does not have it.
        """
        try:
            return int(self.tokenizer.token_to_id(token))
        except (KeyError, ValueError, TypeError):
            return None

    def _language_token_id(self, language):
        """
        Map "en" or "<|en|>" to the language token id, None when not applicable.
        """
        if not language or not self.is_multilingual:
            return None
        token = str(language).strip().lower()
        if not token.startswith("<|"):
            token = f"<|{token}|>"
        language_tokens = getattr(self.tokenizer, "language_tokens", None)
        if isinstance(language_tokens, dict):
            token_id = language_tokens.get(token)
            return int(token_id) if token_id is not None else None
        return self._special_token_id(token)

    def _encode_text(self, text):
        """
        Tokenize text into a list of python integers.
        """
        if not text or not text.strip():
            return []
        token_ids = self._to_numpy(self.tokenizer.tokenize(" " + text.strip()))
        return [int(token_id) for token_id in token_ids.reshape(-1)]

    def _build_encoder(self):
        """
        Encoder only model, so the audio is encoded once per chunk instead of once per token.
        """
        try:
            # the decoding loop calls these backbone layers directly
            self.backbone.decoder_embeddings
            self.backbone.decoder_transformer_layers
            self.backbone.decoder_layer_norm
            return self.keras.Model(
                self.backbone.input["encoder_features"],
                self.backbone.output["encoder_sequence_output"],
            )
        except Exception:
            return None

    def _prepare_audio(self, audio_data, sr, target_sr=16000):
        """
        Downmix stereo and resample if needed to target_sr.
        """
        audio_data = self.np.asarray(audio_data)
        if self.np.issubdtype(audio_data.dtype, self.np.integer):
            # integer pcm to [-1, 1]
            scale = float(self.np.iinfo(audio_data.dtype).max) + 1.0
            audio_data = audio_data.astype("float32") / scale
        else:
            audio_data = audio_data.astype("float32")

        if audio_data.ndim == 2:
            # channels are the short axis: (samples, channels) or (channels, samples)
            channel_axis = 0 if audio_data.shape[0] < audio_data.shape[1] else 1
            audio_data = self.np.mean(audio_data, axis=channel_axis)
        elif audio_data.ndim > 2:
            raise ValueError("audio_data must be a mono or stereo array.")

        if sr != target_sr and audio_data.size > 0:
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
          3) Decoding each chunk with greedy search
          4) Optionally carrying prompt context from one chunk to the next.

        Args:
            audio_data: 1D or 2D NumPy array (audio).
            sample_rate: Original sample rate of `audio_data`.
            language: E.g. "en" or "<|en|>". Used by the multilingual models only.
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

        # nothing to transcribe for empty or silent audio
        if audio_data.size == 0 or float(self.np.max(self.np.abs(audio_data))) < 1e-4:
            return ""

        # identify non-silent segments
        segments = self.librosa.effects.split(y=audio_data, top_db=silence_top_db)
        if len(segments) == 0:
            return ""

        # the audio converter pads or cuts every chunk to its window (30 seconds)
        max_window_sec = getattr(self.converter, "max_audio_length", 30) or 30
        max_chunk_sec = min(max_chunk_sec, max_window_sec)

        # merge small segments
        min_chunk_samples = int(min_chunk_sec * sr)
        max_chunk_samples = int(max_chunk_sec * sr)
        final_chunks = self._merge_segments(
            segments, audio_data, sr, min_chunk_samples, max_chunk_samples
        )

        prompt_ids = self._encode_text(user_prompt)[-self.MAX_PROMPT_TOKENS :]
        context_ids = []
        results = []

        for start, end in final_chunks:
            chunk_data = audio_data[start:end]

            # keep the user prompt and fill the rest of the window with the previous text
            free_slots = self.MAX_PROMPT_TOKENS - len(prompt_ids)
            chunk_prompt_ids = prompt_ids + (
                context_ids[-free_slots:] if free_slots > 0 else []
            )

            token_ids = self._decode_chunk(
                chunk_audio_data=chunk_data,
                language=language,
                max_steps=max_steps,
                prompt_ids=chunk_prompt_ids,
            )
            text = self._detokenize(token_ids)
            if text:
                results.append(text)

            # optionally carry forward the newly decoded tokens
            if condition_on_previous_text and token_ids:
                context_ids = context_ids + token_ids
                if keep_last_n_tokens > 0:
                    context_ids = context_ids[-keep_last_n_tokens:]

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
        The user_prompt is given to the model as the previous context of the transcript.
        """
        token_ids = self._decode_chunk(
            chunk_audio_data=chunk_audio_data,
            language=language,
            max_steps=max_steps,
            prompt_ids=self._encode_text(user_prompt)[-self.MAX_PROMPT_TOKENS :],
        )
        return self._detokenize(token_ids)

    def _build_start_ids(self, language=None, prompt_ids=None):
        """
        Whisper decoder prefix: optional previous context, then the start sequence.
        """
        start_ids = []
        if prompt_ids:
            start_ids.append(self.prev_token_id)
            start_ids.extend(prompt_ids)

        start_ids.append(self.start_token_id)
        if self.is_multilingual:
            lang_id = self._language_token_id(language)
            if lang_id is not None:
                start_ids.append(lang_id)
            start_ids.append(self.transcribe_token_id)
        start_ids.append(self.no_timestamps_token_id)
        return start_ids

    def _next_token_logits(self, encoder_features, encoder_output, token_ids):
        """
        Logits of the next token for the current decoder sequence.
        """
        decoder_ids = self.np.asarray([token_ids], dtype="int32")
        if encoder_output is not None:
            hidden = self.backbone.decoder_embeddings(decoder_ids)
            for layer in self.backbone.decoder_transformer_layers:
                hidden = layer(decoder_sequence=hidden, encoder_sequence=encoder_output)
            hidden = self.backbone.decoder_layer_norm(hidden)
        else:
            hidden = self.backbone(
                {
                    "encoder_features": encoder_features,
                    "decoder_token_ids": decoder_ids,
                    "decoder_padding_mask": self.np.ones_like(decoder_ids),
                }
            )["decoder_sequence_output"]

        logits = self.backbone.token_embedding(hidden[:, -1:, :], reverse=True)
        return self._to_numpy(logits)[0, -1]

    def _decode_chunk(self, chunk_audio_data, language=None, max_steps=80, prompt_ids=None):
        """
        Greedy decoding of one chunk, returns the generated token ids.
        """
        chunk_audio_data = self.np.asarray(chunk_audio_data, dtype="float32")
        encoder_features = self._to_numpy(
            self.converter(chunk_audio_data[self.np.newaxis, ...])
        )
        encoder_output = (
            self.encoder(encoder_features) if self.encoder is not None else None
        )

        start_ids = self._build_start_ids(language=language, prompt_ids=prompt_ids)
        decoder_ids = list(start_ids)
        generated_ids = []

        # stay inside the decoder positions of the model
        max_steps = min(max_steps or 80, self.max_decoder_length - len(start_ids))

        # autoregressive decoding loop
        for _ in range(max_steps):
            logits = self._next_token_logits(
                encoder_features, encoder_output, decoder_ids
            )
            next_id = int(self.np.argmax(logits))

            # break on <|endoftext|>
            if next_id == self.end_token_id:
                break

            decoder_ids.append(next_id)
            generated_ids.append(next_id)

        return generated_ids

    def _detokenize(self, token_ids):
        """
        Convert generated ids to text, the special tokens are dropped.
        """
        # every special token id is above <|endoftext|>
        text_ids = [token_id for token_id in token_ids if token_id < self.end_token_id]
        if not text_ids:
            return ""
        text = self.tokenizer.detokenize(self.np.asarray(text_ids, dtype="int32"))
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return str(text).strip()
