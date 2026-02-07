from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

from src.utils.config import AppConfig
from src.models.real_tts import RealTTSEngine


class TextToSpeech:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.word_to_id = {word: idx for idx, word in enumerate(config.toy_vocab)}
        self.model = self._build_model()
        self.real_tts = RealTTSEngine(rate=150, volume=0.9)

    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.config.tts_max_len,), name="token_ids")
        x = tf.keras.layers.Embedding(self.config.vocab_size, self.config.embedding_dim)(inputs)
        x = tf.keras.layers.LSTM(self.config.hidden_dim)(x)
        outputs = tf.keras.layers.Dense(self.config.tts_output_len)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="tts_toy")

    def synthesize(self, text: str) -> np.ndarray:
        # Use real TTS engine for human voice
        waveform = self.real_tts.synthesize_to_wav(text)
        return waveform.astype(np.float32)

    def train_on_dummy(self) -> None:
        x_train, y_train = self._build_dummy_dataset()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        self.model.fit(x_train, y_train, epochs=self.config.train_epochs, verbose=0)
        self._save_weights()

    def load_or_train(self) -> None:
        if self._load_weights():
            return
        self.train_on_dummy()

    def _build_dummy_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        samples = [
            "hello kids",
            "read and learn",
            "write to learn",
            "speech to text",
            "visual math",
        ]
        x_train: list[np.ndarray] = []
        y_train: list[np.ndarray] = []
        for text in samples:
            token_ids = self._pad_sequence(self._tokenize(text))
            waveform = self._encode_text(text)
            x_train.append(token_ids)
            y_train.append(waveform)
        return np.stack(x_train, axis=0), np.stack(y_train, axis=0)

    def _weights_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "tts_toy.weights.h5")

    def _save_weights(self) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.model.save_weights(self._weights_path())

    def _load_weights(self) -> bool:
        path = self._weights_path()
        if not os.path.exists(path):
            return False
        self.model.load_weights(path)
        return True

    def _encode_text(self, text: str) -> np.ndarray:
        t = np.linspace(0, 1, self.config.tts_output_len, endpoint=False)
        signal = np.zeros_like(t)
        for word in text.lower().split():
            freq = 180 + (sum(ord(c) for c in word) % 50) * 4
            signal += 0.2 * np.sin(2 * np.pi * freq * t)
        return signal.astype(np.float32)

    def _tokenize(self, text: str) -> list[int]:
        tokens = []
        unk = self.word_to_id.get("<unk>", 1)
        for word in text.lower().split():
            tokens.append(self.word_to_id.get(word, unk))
        return tokens

    def _pad_sequence(self, token_ids: list[int]) -> np.ndarray:
        pad_id = self.word_to_id.get("<pad>", 0)
        padded = np.full(self.config.tts_max_len, pad_id, dtype=np.int32)
        length = min(len(token_ids), self.config.tts_max_len)
        if length:
            padded[:length] = np.array(token_ids[:length], dtype=np.int32)
        return padded
