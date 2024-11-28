import os
import torch
import torch.nn.functional as F
import sounddevice as sd

# from TTS.api import TTS
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from transformers import SpeechT5HifiGan


# Get device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
# print(TTS().list_models())

# Microsoft HiFi-GAN vocoder 초기화
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# 모델 경로 설정
base_path = os.path.dirname(os.path.abspath(__file__))
glowtts_model_path = os.path.join(base_path, "model/glowtts_model.pth")
glowtts_config_path = os.path.join(base_path, "model/glowtts_config.json")
dataset_path = os.path.join(base_path, "../train_data/")


# Glow-TTS 모델 로드 함수
def load_glowtts(model_path, config_path):
    config = GlowTTSConfig()
    config.load_json(config_path)
    
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    model = GlowTTS(config, ap, tokenizer)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, tokenizer


# 텍스트를 mel-spectrogram으로 변환
def text_to_mel(text, model, tokenizer):
    # 텍스트를 토큰 ID로 변환
    token_ids = tokenizer.text_to_ids(text)
    
    # 토큰 ID를 텐서로 변환
    tokens = torch.tensor([token_ids], dtype=torch.long)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long)

    # 입력 길이를 384로 맞추기 (Padding or Truncation)
    max_length = 384
    if tokens.size(1) < max_length:
        tokens = F.pad(tokens, (0, max_length - tokens.size(1)), "constant", 0)
    else:
        tokens = tokens[:, :max_length]
    
    # Conv1D와 호환되도록 텐서 차원 조정
    tokens = tokens.unsqueeze(1).permute(0, 2, 1)  # [batch_size, length, channels]

    # Glow-TTS 모델을 사용해 mel-spectrogram 생성
    with torch.no_grad():
        outputs = model.inference(x=tokens, aux_input={"d_vectors": None, "speaker_ids": None, "x_lengths": lengths})
    
    # 결과에서 mel-spectrogram 추출
    mel_spectrogram = outputs["mel_post"].squeeze(0).cpu().numpy()
    return mel_spectrogram


# 텍스트를 음성으로 변환
def synthesize_speech(text):
    # Glow-TTS 모델과 토크나이저 로드
    model, tokenizer = load_glowtts(glowtts_model_path, glowtts_config_path)

    # 텍스트를 mel-spectrogram으로 변환
    mel_spectrogram = text_to_mel(text, model, tokenizer)

    # vocoder를 사용해 오디오 신호 생성
    audio_signal = vocoder(mel_spectrogram)
    return audio_signal


# def make_tts_file(answer):
#     # tts.voice_conversion_to_file(source_wav="tts/data/wavs/bae_korean2.wav", target_wav="tts/data/wavs/product_explain_bae.wav", file_path="output.wav")
    
#     # tts.tts(
#     #     text=answer,
#     #     speaker_wav=wav_file_path,
#     #     language="ko",
#     #     file_path=output_file_path,
#     #     speed=2.0
#     # )
#     return tts.tts(
#         text=answer,
#         speaker_wav=wav_file_path,
#         language="ko",
#         speed=2.0,
#     )


def play_audio(audio, sample_rate=22050):
    # sample_rate = tts.synthesizer.output_sample_rate
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

