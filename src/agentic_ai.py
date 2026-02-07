from __future__ import annotations

from dataclasses import dataclass

from src.models.predictive_text import PredictiveText
from src.models.spell_correction import SpellCorrector
from src.models.speech_recognition import SpeechRecognizer
from src.models.text_to_speech import TextToSpeech
from src.utils.config import AppConfig
from src.visuals.simulations import build_math_concept_scene


@dataclass
class AgentResult:
    transcript: str
    suggestions: list[int]
    corrected: str
    audio_len: int
    scene_title: str


class LearningAgent:
    def __init__(self, config: AppConfig) -> None:
        self.speech = SpeechRecognizer(config)
        self.tts = TextToSpeech(config)
        self.predictive = PredictiveText(config)
        self.spell = SpellCorrector()

    def support_session(self, audio_pcm) -> AgentResult:
        transcript = self.speech.transcribe(audio_pcm)
        token_suggestions = self.predictive.suggest_next([0, 1, 2])
        corrected = self.spell.correct(transcript)
        tts_audio = self.tts.synthesize(corrected)
        scene = build_math_concept_scene()
        return AgentResult(
            transcript=transcript,
            suggestions=token_suggestions,
            corrected=corrected,
            audio_len=len(tts_audio),
            scene_title=scene.title,
        )
