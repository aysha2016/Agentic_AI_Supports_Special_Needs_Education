from __future__ import annotations

from difflib import get_close_matches
from typing import Iterable


class SpellCorrector:
    def __init__(self) -> None:
        self.lexicon: set[str] = set()

    def load_lexicon(self, words: Iterable[str]) -> None:
        self.lexicon = set(w.lower() for w in words)

    def correct(self, text: str) -> str:
        if not self.lexicon:
            return text

        corrected_words: list[str] = []
        for word in text.split():
            normalized = "".join(ch for ch in word.lower() if ch.isalnum())
            if normalized in self.lexicon:
                corrected_words.append(word)
                continue
            match = get_close_matches(normalized, self.lexicon, n=1, cutoff=0.8)
            corrected_words.append(match[0] if match else word)
        return " ".join(corrected_words)
