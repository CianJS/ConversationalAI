import os
import json
import pandas as pd
from TTS.config import load_config
from TTS.trainer import Trainer  # 수정된 임포트 경로
from TTS.utils.synthesizer import Synthesizer

def create_metadata(data_dir, wav_dir, output_csv, texts):
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    if len(wav_files) != len(texts):
        raise ValueError("WAV 파일 수와 텍스트 수가 일치하지 않습니다.")

    metadata = []
    for wav, text in zip(wav_files, texts):
        file_path = os.path.join('wavs', wav)
        metadata.append({'file_path': file_path, 'transcription': text})

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(data_dir, output_csv), index=False, encoding='utf-8')
    print(f"metadata.csv가 {output_csv}로 저장되었습니다.")

def preprocess_data(config_path):
    config = load_config(config_path)
    # 전처리 작업 (필요 시 추가)
    print("데이터 전처리가 완료되었습니다.")

def train_model(config_path):
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.fit()

def evaluate_model(config_path, model_path, output_dir, test_texts):
    config = load_config(config_path)
    synthesizer = Synthesizer(model_path, config)
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, text in enumerate(test_texts):
        wav = synthesizer.tts(text)
        synthesizer.save_wav(wav, os.path.join(output_dir, f"test_{idx}.wav"))
        print(f"test_{idx}.wav 생성 완료.")

def synthesize_text(config_path, model_path, text, output_wav):
    config = load_config(config_path)
    synthesizer = Synthesizer(model_path, config)
    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, output_wav)
    print(f"{output_wav} 생성 완료.")

if __name__ == "__main__":
    DATA_DIR = 'data'
    WAV_DIR = os.path.join(DATA_DIR, 'wavs')
    OUTPUT_CSV = 'metadata.csv'
    CONFIG_PATH = 'config.json'
    MODEL_PATH = 'out/models/best_model.pth'
    EVAL_OUTPUT_DIR = 'evaluation_outputs'
    
    # 1. metadata.csv 생성
    TEXTS = [
        "안녕하세요. bae의 한국어 파일입니다.",
        "안녕하세요. bae의 한국어 파일2입니다.",
        "bae의 상품 설명 한국어입니다.",
    ]
    create_metadata(DATA_DIR, WAV_DIR, OUTPUT_CSV, TEXTS)
    
    # 2. 데이터 전처리
    preprocess_data(CONFIG_PATH)
    
    # 3. 모델 학습
    train_model(CONFIG_PATH)
    
    # 4. 모델 평가
    TEST_TEXTS = [
        "평가를 위한 첫 번째 테스트 문장입니다.",
        "두 번째 테스트 문장을 생성합니다."
    ]
    evaluate_model(CONFIG_PATH, MODEL_PATH, EVAL_OUTPUT_DIR, TEST_TEXTS)
    
    # 5. 음성 합성
    TEXT_TO_SYNTH = "여기에 변환할 텍스트를 입력하세요."
    OUTPUT_WAV = 'output.wav'
    synthesize_text(CONFIG_PATH, MODEL_PATH, TEXT_TO_SYNTH, OUTPUT_WAV)