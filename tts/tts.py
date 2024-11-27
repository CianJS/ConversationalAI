import os
import torch
import sounddevice as sd
from TTS.api import TTS


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
# print(TTS().list_models())

# Init
# tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
base_path = os.path.dirname(os.path.abspath(__file__))
wav_file_path = os.path.join(base_path, 'data/wavs/bae_korean2.wav')
print(wav_file_path)
output_file_path = os.path.join(base_path, '../data/output.wav')
print(output_file_path)

def make_tts_file(answer):
    # tts.voice_conversion_to_file(source_wav="tts/data/wavs/bae_korean2.wav", target_wav="tts/data/wavs/product_explain_bae.wav", file_path="output.wav")
    
    # tts.tts(
    #     text=answer,
    #     speaker_wav=wav_file_path,
    #     language="ko",
    #     file_path=output_file_path,
    #     speed=2.0
    # )
    return tts.tts(
        text=answer,
        speaker_wav=wav_file_path,
        language="ko",
        speed=2.0,
    )


def play_audio(audio_array):
    sample_rate = tts.synthesizer.output_sample_rate
    sd.play(audio_array, samplerate=sample_rate)
    sd.wait()
    