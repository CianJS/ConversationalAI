import os
import torch
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


# Glow-TTS 모델 로드 함수
def load_glowtts(model_path, config_path):
    config = GlowTTSConfig()
    config.load_json(config_path)
    
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    model = GlowTTS(config, ap, tokenizer)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, tokenizer

# def load_glowtts_model():
#     config = GlowTTSConfig()
#     config.load_json(file_name=glowtts_config_path)
#     ap = AudioProcessor.init_from_config(config)
    
#     # Tokenizer 초기화
#     tokenizer, config = TTSTokenizer.init_from_config(config)

#     model = GlowTTS(config, ap)
#     model.load_state_dict(torch.load(glowtts_model_path, map_location="cpu", weights_only=True))
#     model.eval()
#     return model, ap, tokenizer


# 텍스트를 음성으로 변환하는 함수
def synthesize_speech(text):
    # Glow-TTS 모델 및 설정 로드
    glowtts_model, tokenizer = load_glowtts(glowtts_model_path, glowtts_config_path)
    print(glowtts_model.embedded_speaker_dim)

    # 1. 텍스트를 토큰화 및 음소화
    text_sequence = tokenizer.encode(text)
    text_tensor = torch.tensor([text_sequence], dtype=torch.long)
    text_lengths = torch.tensor([text_tensor.size(1)], dtype=torch.long)

    print(text_tensor.shape)
    print(text_lengths)

    # 2. Glow-TTS를 사용하여 멜 스펙트로그램 생성
    with torch.no_grad():
        mel_outputs = glowtts_model.inference(x=text_tensor)
    print('mel_outputs:', mel_outputs)

    # 3. HiFi-GAN vocoder를 사용하여 wav 변환
    with torch.no_grad():
        wav = vocoder(mel_outputs)

    print(wav)
    return wav


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


# 텍스트를 멜 스펙트로그램으로 변환하고 음성 신호를 생성하는 함수
# def synthesize_speech(text):
#     # GlowTTS 모델, AudioProcessor, Tokenizer 로드
#     model, ap, tokenizer = load_glowtts_model()

#     # 텍스트를 토크나이저로 처리
#     tokens = tokenizer.text_to_ids(text)
#     tokens = torch.LongTensor(tokens).unsqueeze(0)  # 배치 차원 추가
#     print("Tokens shape:", tokens.shape)
#     print("First token length:", len(tokens[0]))

#     # 멜 스펙트로그램 생성
#     with torch.no_grad():
#         outputs = model.inference(tokens)
#         print(outputs)
#         print("GlowTTS inference completed successfully.")
#         mel_post = outputs["mel_post"]  # 멜 스펙트로그램 가져오기
#         print("Mel spectrogram shape:", mel_post.shape)

#     try:
#         # SpeechT5 HiFi-GAN을 사용해 음성 신호로 변환
#         with torch.no_grad():
#             wav = vocoder(mel_post)
#             print("Vocoder synthesis completed successfully.")
#     except Exception as e:
#         print(f"Error during vocoder synthesis: {e}")
#         return None, None

#     return wav.cpu().numpy(), ap.sample_rate


def play_audio(audio, sample_rate=22050):
    # sample_rate = tts.synthesizer.output_sample_rate
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

