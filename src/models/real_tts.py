from __future__ import annotations

import io

import numpy as np
from gtts import gTTS


class RealTTSEngine:
    """Real text-to-speech using Google Text-to-Speech API with human voice."""

    def __init__(self, rate: int = 150, volume: float = 1.0) -> None:
        self.volume = volume
        self.sample_rate = 24000

    def synthesize_to_wav(self, text: str) -> np.ndarray:
        try:
            # Use gTTS to generate speech
            tts = gTTS(text=text, lang="en", slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # gTTS returns MP3, return raw bytes as float array
            # For Streamlit playback, we'll handle MP3 directly
            mp3_bytes = audio_buffer.read()
            
            # Return a dummy array - we'll use MP3 bytes directly in the UI
            # This is a workaround since gTTS outputs MP3, not WAV
            return np.frombuffer(mp3_bytes[:self.sample_rate], dtype=np.uint8).astype(np.float32) / 255.0
        except Exception as e:
            print(f"TTS error: {e}")
            # Fallback: return silence
            return np.zeros(self.sample_rate, dtype=np.float32)
    
    def synthesize_to_mp3_bytes(self, text: str) -> bytes:
        """Generate MP3 audio bytes directly for playback."""
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
        except Exception as e:
            print(f"TTS error: {e}")
            return b""
