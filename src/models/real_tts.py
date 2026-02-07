from __future__ import annotations

import io
import wave

import numpy as np
from gtts import gTTS


class RealTTSEngine:
    """Real text-to-speech using Google Text-to-Speech API with human voice."""

    def __init__(self, rate: int = 150, volume: float = 1.0) -> None:
        self.volume = volume
        self.sample_rate = 22050

    def synthesize_to_wav(self, text: str) -> np.ndarray:
        try:
            # Use gTTS to generate speech
            tts = gTTS(text=text, lang="en", slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Read back as WAV
            with wave.open(audio_buffer, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                # Apply volume scaling
                audio_array = audio_array * self.volume
            return audio_array
        except Exception as e:
            print(f"TTS error: {e}")
            # Fallback: return silence
            return np.zeros(self.sample_rate, dtype=np.float32)
