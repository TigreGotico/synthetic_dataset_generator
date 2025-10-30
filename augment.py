#!/usr/bin/env python3
"""
augmentation.py — Optional dataset augmentation stage for wake-word detection.

This script applies noise mixing, music mixing, background speech mixing, reverb,
pitch shifting, and speed perturbation to existing preprocessed audio.

It is designed to run *after* `preprocess.py` and *before* `train.py`.

Advantages of training on augmented files while evaluating on the clean source data:
- **Realism:** Synthetic TTS samples or clean lab recordings often lack natural acoustic variability. Augmentations mimic
  real-world conditions such as background chatter, reverb, and microphone inconsistencies.
- **Robustness:** By training the model on noisy and reverberant versions, it becomes more resilient to deployment
  environments where SNR and acoustics vary widely.
- **Fair Evaluation:** Evaluating on clean (non-augmented) data ensures that model quality improvements are not just due
  to fitting the augmentation artifacts but generalize to unseen, unmodified samples.

Usage examples:

  python augmentation.py \
    --input processed_data \
    --out augmented_data \
    --bg-noise assets/noise \
    --rir assets/rir \
    --snr-min 0 --snr-max 20 \
    --pitch-min -1 --pitch-max 1 \
    --speed-min 0.95 --speed-max 1.05

Outputs:
  augmented_data/
    wakes_aug/*.wav
    negatives_aug/*.wav
    metadata.csv

The new metadata file points to the augmented files, preserving labels.

You can then train with:
  python train.py train --metadata augmented_data/metadata.csv
and test against the clean dataset:
  python train.py test --ckpt model.pt --dataset processed_data/metadata.csv

"""

import random
from pathlib import Path

import click
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def load_audio_mono(path, sr=16000):
    wav, orig_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def save_wav(path, wav, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)


def mix_audio(clean, bg, snr_db):
    """
    Mixes a background audio track (noise, music, or speech) into the clean audio.
    The background audio is cropped/tiled to match the length of the clean audio.
    """
    clean_len = len(clean)

    if len(bg) < clean_len:
        # Tile the background audio if it's too short
        nrep = int(np.ceil(clean_len / len(bg)))
        bg = np.tile(bg, nrep)

    # Randomly select a segment of the background audio that is the length of the clean audio
    start = random.randint(0, len(bg) - clean_len)
    bg = bg[start:start + clean_len]

    # Calculate RMS for scaling
    rms_clean = np.sqrt(np.mean(clean ** 2) + 1e-9)
    rms_bg = np.sqrt(np.mean(bg ** 2) + 1e-9)

    # Calculate desired background RMS based on target SNR
    desired_bg_rms = rms_clean / (10 ** (snr_db / 20.0))

    if rms_bg > 0:
        bg = bg * (desired_bg_rms / rms_bg)

    mixed = clean + bg

    # Apply peak normalization to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def apply_reverb(wav, rir):
    # The original normalization made the convolved signal's RMS match the input wav's RMS.
    # To make it less pronounced, we apply a final scaling factor (e.g., 0.5) to attenuate the reverb level.
    REVERB_ATTENUATION_FACTOR = 0.5

    out = np.convolve(wav, rir)[:len(wav)]

    # The original RMS ratio logic: out = out * (RMS_wav / RMS_out)
    rms_wav = np.sqrt(np.mean(wav ** 2) + 1e-9)
    rms_out = np.sqrt(np.mean(out ** 2) + 1e-9)

    # Apply the RMS ratio AND the attenuation factor
    out = out * (rms_wav / rms_out) * REVERB_ATTENUATION_FACTOR

    return out.astype(np.float32)

def pitch_shift(wav, sr, n_steps):
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps).astype(np.float32)


def speed_perturb(wav, factor):
    return librosa.effects.time_stretch(wav, rate=factor).astype(np.float32)


def collect_audio_files(base_folder):
    exts = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]
    files = []
    if not base_folder:
        return files
    p = Path(base_folder)
    if not p.exists():
        click.echo(f"Warning: Augmentation folder not found: {base_folder}")
        return files
    for ext in exts:
        files.extend(p.rglob(f"*{ext}"))
    return sorted(files)


@click.command()
@click.option('--input', 'input_dir', required=True, help='Path to preprocessed dataset folder (with metadata.csv)')
@click.option('--out', 'out_dir', required=True, help='Output folder for augmented dataset')
@click.option('--bg-noise', 'bg_folder', default=None, help='Folder with general background noises (e.g., static, nature)')
@click.option('--music-folder', default=None, help='Folder with music files to mix in background')
@click.option('--bg-speech-folder', default=None, help='Folder with non-wake-word speech to simulate busy rooms')
@click.option('--mic-noise', 'mic_folder', default=None, help='Folder with microphone specific background silence')
@click.option('--rir', 'rir_folder', default=None, help='Folder with room impulse responses (RIRs)')
@click.option('--snr-min', default=0.0, type=float, help='Minimum SNR for general background mixing (bg-noise, mic-noise)')
@click.option('--snr-max', default=20.0, type=float, help='Maximum SNR for general background mixing (bg-noise, mic-noise)')
@click.option('--pitch-min', default=-1.0, type=float, help='Minimum pitch shift (semitones)')
@click.option('--pitch-max', default=1.0, type=float, help='Maximum pitch shift (semitones)')
@click.option('--speed-min', default=0.95, type=float, help='Minimum speed factor')
@click.option('--speed-max', default=1.05, type=float, help='Maximum speed factor')
@click.option('--prob', default=0.9, type=float, help='Probability of applying augmentation to each file')
def main(input_dir, out_dir, bg_folder, music_folder, bg_speech_folder, mic_folder, rir_folder, snr_min, snr_max, pitch_min, pitch_max, speed_min, speed_max, prob):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_in = input_dir / 'metadata.csv'
    metadata_out = out_dir / 'metadata.csv'

    if not metadata_in.exists():
        raise FileNotFoundError(f"metadata.csv not found in {input_dir}")

    bg_files = collect_audio_files(bg_folder)
    music_files = collect_audio_files(music_folder)
    bg_speech_files = collect_audio_files(bg_speech_folder)
    rir_files = collect_audio_files(rir_folder)
    mic_files = collect_audio_files(mic_folder)

    # Read all lines from the input metadata
    with open(metadata_in, 'r', encoding='utf-8') as fin:
        source_data = [line.strip().split(',') for line in fin]

    total_samples = len(source_data)

    click.echo("--- Starting Augmentation ---")
    click.echo(f"Processing {total_samples} samples from {metadata_in}")
    click.echo(f"Augmentation probability: {prob * 100:.1f}%")

    aug_summary = {
        'noise_mixing': len(bg_files) > 0,
        'music_mixing': len(music_files) > 0,
        'bg_speech_mixing': len(bg_speech_files) > 0,
        'reverb': len(rir_files) > 0,
        'pitch_shift': True,
        'speed_perturb': True
    }

    aug_list = [name for name, available in aug_summary.items() if available]
    click.echo(f"Available augmentations: {', '.join(aug_list) if aug_list else 'None'}")

    augmented = []
    augmented_count = 0

    click.echo("Applying transformations...")

    for path, label in tqdm(source_data, desc="Augmenting Data", unit="sample"):
        src_path = Path(path)

        # Determine relative path from input_dir to ensure correct loading
        # We assume they are relative to the input_dir if they don't start with /
        if not src_path.is_absolute():
            # Adjust based on how preprocess.py wrote paths
            src_path = input_dir.parent / src_path

        try:
            wav = load_audio_mono(src_path)
        except Exception as e:
            tqdm.write(f"Warning: Could not load file {src_path}. Skipping. Error: {e}")
            continue

        was_augmented = False

        if random.random() < prob:

            # 1. Background Noise Mixing (General and Mic)
            if bg_files and random.random() < 0.5:
                bg = load_audio_mono(random.choice(bg_files))
                snr = random.uniform(snr_min, snr_max)
                wav = mix_audio(wav, bg, snr)
                was_augmented = True

            # Apply mic silence with 80% probability if files are available
            if mic_files and random.random() < 0.8:
                mic = load_audio_mono(random.choice(mic_files))
                snr = random.uniform(snr_min, snr_max)
                wav = mix_audio(wav, mic, snr)
                was_augmented = True

            # 2. Music Mixing (Higher volume, 50% probability)
            if music_files and random.random() < 0.5:
                music = load_audio_mono(random.choice(music_files))
                # Music is usually louder, so we use a lower SNR range (louder background)
                snr = random.uniform(0.0, 10.0)
                wav = mix_audio(wav, music, snr)
                was_augmented = True

            # 3. Background Speech Mixing (Lower volume, 60% probability)
            if bg_speech_files and random.random() < 0.6:
                speech = load_audio_mono(random.choice(bg_speech_files))
                # Background speech is quieter, so we use a higher SNR range (quieter background)
                snr = random.uniform(10.0, 25.0)
                wav = mix_audio(wav, speech, snr)
                was_augmented = True

            # 4. Reverb (30% probability)
            if rir_files and random.random() < 0.3:
                rir = load_audio_mono(random.choice(rir_files))
                wav = apply_reverb(wav, rir)
                was_augmented = True

            # 5. Pitch Shift (30% probability)
            if random.random() < 0.3:
                n_steps = random.uniform(pitch_min, pitch_max)
                wav = pitch_shift(wav, 16000, n_steps)
                was_augmented = True

            # 6. Speed Perturb (30% probability)
            if random.random() < 0.3:
                f = random.uniform(speed_min, speed_max)
                wav = speed_perturb(wav, f)
                was_augmented = True

        if was_augmented:
            augmented_count += 1

        # normalize and save
        wav = wav / (np.max(np.abs(wav)) + 1e-9)

        rel_path = Path('wakes_aug' if int(label) == 1 else 'negatives_aug') / (src_path.stem + '_aug.wav')
        dst_path = out_dir / rel_path
        save_wav(dst_path, wav)
        augmented.append((str(dst_path), label))

    with metadata_out.open('w', encoding='utf-8') as fout:
        for path, label in augmented:
            fout.write(f"{path},{label}\n")

    click.echo("\n--- Augmentation Complete ---")
    click.echo(f"Source samples processed: {total_samples}")
    click.echo(f"Successfully generated {len(augmented)} augmented samples.")
    click.echo(f"{augmented_count} samples received at least one transformation.")
    click.echo(f"Metadata written to {metadata_out}")


if __name__ == '__main__':
    main()
