import os
import json
import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import kagglehub


# Experiment paths
output_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(output_path, "../train_data/")

# Download dataset if not exists
if not os.path.exists(dataset_path):
    dataset_path = kagglehub.dataset_download("bryanpark/korean-single-speaker-speech-dataset")
    print("Dataset path:", dataset_path)


def kss_formatter(root_path, meta_file):
    """Custom formatter for KSS dataset."""
    meta_path = os.path.join(root_path, meta_file)
    samples = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2 or not parts[1].strip():  # Skip invalid or empty text lines
                continue

            wav_file = os.path.join(root_path, "kss", parts[0])
            samples.append({
                "audio_file": wav_file,
                "text": parts[1],
                "speaker_name": "kss",  # Single-speaker dataset
                "audio_unique_name": os.path.splitext(os.path.basename(wav_file))[0],
                "language": "ko-kr",  # Korean language
            })
    return samples


def extract_characters_from_texts(samples):
    """Extract unique characters from the dataset's text."""
    characters = set()
    for sample in samples:
        characters.update(sample["text"])
    return sorted(characters)


def split_samples(samples, eval_ratio=0.1):
    """Split dataset into training and evaluation sets."""
    eval_size = int(len(samples) * eval_ratio)
    return samples[eval_size:], samples[:eval_size]


def validate_dataset(samples):
        valid_samples = []
        for sample in samples:
            try:
                # 오디오 파일 확인
                wav = ap.load_wav(sample["audio_file"])
                mel = ap.melspectrogram(wav)
                if mel.ndim != 2:  # 멜 스펙트로그램이 2차원인지 확인
                    print(f"Invalid mel shape for {sample['audio_file']}")
                    continue
                valid_samples.append(sample)
            except Exception as e:
                print(f"Error processing {sample['audio_file']}: {e}")
        return valid_samples


if __name__ == "__main__":
    # Metadata file
    meta_file = "transcript.v.1.4.txt"

    # Dataset configuration
    dataset_config = BaseDatasetConfig(formatter="kss", meta_file_train="", path=dataset_path)

    # Audio configuration
    audio_config = BaseAudioConfig(sample_rate=22050, resample=True, do_trim_silence=True, trim_db=23.0)

    # Load dataset
    samples = kss_formatter(dataset_path, meta_file)

    punctuations = [",", ".", "?", "!", ":", ";", "-", "…", "'"]
    characters = extract_characters_from_texts(samples)
    characters = [char for char in characters if char not in punctuations]
    
    # Save characters to file for debugging
    char_path = os.path.join(output_path, "characters.json")
    with open(char_path, "w", encoding="utf-8") as f:
        json.dump(characters, f, ensure_ascii=False, indent=4)

    # Convert characters to CharactersConfig
    characters_config = CharactersConfig(
        characters="".join(characters),
        punctuations="".join(punctuations),
        pad='_',
        eos='~',
        bos='^',
    )

    # Split dataset into train and eval
    train_samples, eval_samples = split_samples(samples, eval_ratio=0.1)

    # Model configuration
    # 모델 구성 수정
    config = GlowTTSConfig(
        batch_size=64,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        epochs=5,
        lr=1e-3,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        output_path=output_path,
        datasets=[dataset_config],
        use_speaker_embedding=True,
        min_text_len=1,
        max_text_len=300,
        min_audio_len=1,
        max_audio_len=500000,
        characters=characters_config,
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # 검증된 데이터셋으로 필터링
    train_samples = validate_dataset(train_samples)
    eval_samples = validate_dataset(eval_samples)

    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.num_speakers = speaker_manager.num_speakers

    # Initialize the model
    model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

    # Start trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    
    # Save the model
    model_save_path = os.path.join(output_path, "glowtts_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")
