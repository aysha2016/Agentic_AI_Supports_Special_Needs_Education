import io
import time
import wave

import streamlit as st

from src.agentic_ai import LearningAgent
from src.utils.config import AppConfig


@st.cache_resource
def load_agent() -> LearningAgent:
    config = AppConfig()
    agent = LearningAgent(config)
    agent.spell.load_lexicon(config.toy_vocab)
    agent.speech.load_or_train()
    agent.predictive.load_or_train()
    agent.tts.load_or_train()
    return agent


def waveform_to_wav_bytes(
    waveform,
    sample_rate_hz: int,
    normalize: bool,
    target_peak: float,
) -> bytes:
    clipped = waveform
    if normalize and clipped.size:
        max_val = float(max(abs(clipped.max()), abs(clipped.min())))
        if max_val > 0:
            clipped = (clipped / max_val) * target_peak
    pcm16 = (clipped * 32767.0).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate_hz)
            wf.writeframes(pcm16.tobytes())
        return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Assistive AI Demo", page_icon="🤖")
    st.title("Assistive AI Demo")
    st.write("Run the toy agentic AI pipeline and view outputs.")

    st.subheader("Audio Settings")
    normalize_audio = st.checkbox("Normalize volume", value=True)
    target_peak = st.slider("Target peak", min_value=0.2, max_value=1.0, value=0.9)

    if "last_transcript" not in st.session_state:
        st.session_state["last_transcript"] = ""

    if st.button("Run demo"):
        agent = load_agent()
        config = AppConfig()
        audio_pcm = agent.speech.dummy_audio_for_phrase("speech to text")
        result = agent.support_session(audio_pcm)
        suggestions_text = agent.predictive.decode_ids(result.suggestions)
        tts_waveform = agent.tts.synthesize(result.corrected)
        tts_wav = waveform_to_wav_bytes(
            tts_waveform,
            config.sample_rate_hz,
            normalize_audio,
            target_peak,
        )
        st.session_state["last_transcript"] = result.transcript

        st.subheader("Results")
        st.write("Transcript:", result.transcript)
        st.write("Suggestions:", result.suggestions)
        st.write("Suggestions decoded:", suggestions_text)
        st.write("Corrected:", result.corrected)
        st.write("TTS length:", result.audio_len)
        st.audio(tts_wav, format="audio/wav")
        st.write("Visual scene:", result.scene_title)

    st.subheader("Word Highlighting")
    playback_speed = st.slider("Words per second", min_value=1, max_value=6, value=2)
    transcript = st.session_state.get("last_transcript", "")
    if not transcript:
        st.info("Run the demo to generate a transcript for highlighting.")
        return

    words = transcript.split()
    highlight_area = st.empty()

    def render_highlight(active_index: int) -> None:
        rendered = []
        for idx, word in enumerate(words):
            if idx == active_index:
                rendered.append(f"<span style='background:#ffe08a;padding:2px 6px;border-radius:6px;'>{word}</span>")
            else:
                rendered.append(word)
        highlight_area.markdown(" ".join(rendered), unsafe_allow_html=True)

    col_play, col_replay = st.columns(2)
    play = col_play.button("Play highlight")
    replay = col_replay.button("Replay highlight")
    if play or replay:
        delay = 1.0 / playback_speed
        for idx in range(len(words)):
            render_highlight(idx)
            time.sleep(delay)


if __name__ == "__main__":
    main()
