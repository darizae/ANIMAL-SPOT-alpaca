from pathlib import Path
import os


def analyze_all_datasets(base_dir):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}")
        return

    dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("dataset_")]
    if not dataset_dirs:
        print("No dataset_* folders found.")
        return

    print(f"\n🔍 Analyzing datasets in: {base_dir}\n")

    for ds in sorted(dataset_dirs):
        wav_files = list(ds.glob("*.wav"))
        total = len(wav_files)
        noise_count = sum(1 for f in wav_files if f.name.startswith("noise-"))
        target_count = sum(1 for f in wav_files if f.name.startswith("target-"))
        noise_to_target_ratio = (noise_count / target_count) if target_count else 0
        print(f"📁 {ds.name}")
        print(f"  Total .wav files:     {total}")
        print(f"  Target clips:         {target_count}")
        print(f"  Noise clips:          {noise_count}")
        print(f"  Noise/Target ratio:   {noise_to_target_ratio:.2f}\n")


if __name__ == "__main__":
    base = os.getenv("TRAINING_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "training_corpus_v1"))
    analyze_all_datasets(base)
