from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

from src.utils.config import AppConfig


class PredictiveText:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.word_to_id = {word: idx for idx, word in enumerate(config.toy_vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.config.max_seq_len,), name="token_ids")
        x = tf.keras.layers.Embedding(self.config.vocab_size, self.config.embedding_dim)(inputs)
        x = tf.keras.layers.LSTM(self.config.hidden_dim)(x)
        outputs = tf.keras.layers.Dense(self.config.vocab_size, activation="softmax")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="predictive_text_toy")

    def suggest_next(self, token_ids: list[int]) -> list[int]:
        padded = self._pad_sequence(token_ids)
        probs = self.model.predict(padded[np.newaxis, :], verbose=0)[0]
        top_ids = np.argsort(probs)[-3:][::-1]
        return [int(idx) for idx in top_ids]

    def train_on_dummy(self) -> None:
        inputs, labels = self._build_dummy_dataset()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(inputs, labels, epochs=self.config.train_epochs, verbose=0)
        self._save_weights()

    def load_or_train(self) -> None:
        if self._load_weights():
            return
        self.train_on_dummy()

    def decode_ids(self, token_ids: list[int]) -> str:
        words = [self.id_to_word.get(idx, "<unk>") for idx in token_ids]
        return " ".join(words)

    def _build_dummy_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        sentences = [
            "hello kids",
            "read and learn",
            "write to learn",
            "speech to text",
            "visual math",
            "support kids to read",
        ]
        inputs: list[np.ndarray] = []
        labels: list[int] = []
        for sentence in sentences:
            token_ids = self._tokenize(sentence)
            for i in range(1, len(token_ids)):
                context = token_ids[:i]
                inputs.append(self._pad_sequence(context))
                labels.append(token_ids[i])
        return np.stack(inputs, axis=0), np.array(labels, dtype=np.int32)

    def _weights_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "predictive_text.weights.h5")

    def _save_weights(self) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.model.save_weights(self._weights_path())

    def _load_weights(self) -> bool:
        path = self._weights_path()
        if not os.path.exists(path):
            return False
        self.model.load_weights(path)
        return True

    def _tokenize(self, text: str) -> list[int]:
        tokens = []
        for word in text.lower().split():
            tokens.append(self.word_to_id.get(word, self.word_to_id["<unk>"]))
        return tokens

    def _pad_sequence(self, token_ids: list[int]) -> np.ndarray:
        padded = np.full(self.config.max_seq_len, self.word_to_id["<pad>"], dtype=np.int32)
        length = min(len(token_ids), self.config.max_seq_len)
        if length:
            padded[:length] = np.array(token_ids[:length], dtype=np.int32)
        return padded
