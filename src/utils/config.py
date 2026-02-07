from dataclasses import dataclass, field


@dataclass
class AppConfig:
    sample_rate_hz: int = 16000
    max_seq_len: int = 128
    embedding_dim: int = 128
    hidden_dim: int = 256
    tts_voice: str = "default"
    tts_output_len: int = 4000
    tts_max_len: int = 16
    sr_audio_len: int = 8000
    train_epochs: int = 6
    checkpoint_dir: str = "checkpoints"
    toy_vocab: list[str] = field(
        default_factory=lambda: [
            "<pad>",
            "<unk>",
            "hello",
            "read",
            "write",
            "help",
            "learn",
            "support",
            "math",
            "visual",
            "speech",
            "text",
            "kids",
            "audio",
        ]
    )
    toy_phrases: list[str] = field(
        default_factory=lambda: [
            "hello kids",
            "read and learn",
            "write to learn",
            "speech to text",
            "visual math",
        ]
    )
    vocab_size: int = 0

    def __post_init__(self) -> None:
        if not self.vocab_size:
            self.vocab_size = len(self.toy_vocab)
