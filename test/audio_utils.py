"""Deterministic audio inputs shared by metric tests."""

import wave

import numpy as np


def generate_fixed_wav(
    filename, duration=1.0, sample_rate=16000, base_freq=150, envelope_func=None
):
    """
    Generate a deterministic WAV file with a modulated sine wave.

    Parameters:
      - filename: Path (str or Path) to write the WAV file.
      - duration: Duration of the audio in seconds.
      - sample_rate: Number of samples per second.
      - base_freq: Frequency (in Hz) of the sine wave.
      - envelope_func: Optional function to generate a custom amplitude envelope.
                       If None, a default sine-based envelope is used.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Use default envelope if none is provided.
    if envelope_func is None:
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    else:
        envelope = envelope_func(t)
    audio = envelope * np.sin(2 * np.pi * base_freq * t)

    # Scale to 16-bit PCM.
    amplitude = np.iinfo(np.int16).max
    data = (audio * amplitude).astype(np.int16)

    # Write the WAV file.
    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(1)  # Mono audio.
        wf.setsampwidth(2)  # 16 bits per sample.
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
