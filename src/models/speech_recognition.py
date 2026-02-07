from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

from src.utils.config import AppConfig


class SpeechRecognizer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.phrases = list(config.toy_phrases)
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.config.sr_audio_len, 1), name="audio")
        x = tf.keras.layers.Conv1D(32, 5, activation="relu")(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(len(self.phrases), activation="softmax")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="speech_recognizer_toy")

    def transcribe(self, audio_pcm: np.ndarray) -> str:
        audio = self._pad_or_trim(audio_pcm)
        audio = np.expand_dims(audio, axis=(0, 2))
        probs = self.model.predict(audio, verbose=0)[0]
        phrase_idx = int(np.argmax(probs))
        return self.phrases[phrase_idx]

    def train_on_dummy(self) -> None:
        x_train, y_train = self._build_dummy_dataset(samples_per_phrase=6)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(x_train, y_train, epochs=self.config.train_epochs, verbose=0)
        self._save_weights()

    def load_or_train(self) -> None:
        if self._load_weights():
            return
        self.train_on_dummy()

    def dummy_audio_for_phrase(self, phrase: str) -> np.ndarray:
        return self._encode_phrase(phrase)

    def _build_dummy_dataset(self, samples_per_phrase: int) -> tuple[np.ndarray, np.ndarray]:
        audio_samples: list[np.ndarray] = []
        labels: list[int] = []
        for idx, phrase in enumerate(self.phrases):
            base = self._encode_phrase(phrase)
            for _ in range(samples_per_phrase):
                noise = np.random.normal(0.0, 0.02, size=base.shape)
                audio_samples.append(self._pad_or_trim(base + noise))
                labels.append(idx)
        x = np.stack(audio_samples, axis=0)[..., np.newaxis]
        y = np.array(labels, dtype=np.int32)
        return x, y

    def _weights_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "speech_recognizer.weights.h5")

    def _save_weights(self) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.model.save_weights(self._weights_path())

    def _load_weights(self) -> bool:
        path = self._weights_path()
        if not os.path.exists(path):
            return False
        self.model.load_weights(path)
        return True

    def _encode_phrase(self, phrase: str) -> np.ndarray:
        t = np.linspace(0, 1, self.config.sr_audio_len, endpoint=False)
        signal = np.zeros_like(t)
        for char in phrase:
            freq = 220 + (ord(char) % 40) * 5
            signal += 0.2 * np.sin(2 * np.pi * freq * t)
        return signal.astype(np.float32)

    def _pad_or_trim(self, audio_pcm: np.ndarray) -> np.ndarray:
        audio = audio_pcm.astype(np.float32)
        if audio.shape[0] < self.config.sr_audio_len:
            pad = self.config.sr_audio_len - audio.shape[0]
            audio = np.pad(audio, (0, pad))
        return audio[: self.config.sr_audio_len]
