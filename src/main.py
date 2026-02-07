from __future__ import annotations

from src.agentic_ai import LearningAgent
from src.utils.config import AppConfig


def main() -> None:
    config = AppConfig()
    agent = LearningAgent(config)

    agent.spell.load_lexicon(config.toy_vocab)
    agent.speech.load_or_train()
    agent.predictive.load_or_train()
    agent.tts.load_or_train()

    audio_pcm = agent.speech.dummy_audio_for_phrase("speech to text")
    result = agent.support_session(audio_pcm)

    decoded_suggestions = agent.predictive.decode_ids(result.suggestions)

    print("Transcript:", result.transcript)
    print("Suggestions:", result.suggestions)
    print("Suggestions decoded:", decoded_suggestions)
    print("Corrected:", result.corrected)
    print("TTS length:", result.audio_len)
    print("Visual scene:", result.scene_title)


if __name__ == "__main__":
    main()
