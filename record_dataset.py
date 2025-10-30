import os
import time
import csv
import uuid
import click
import torch
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
from colorama import Fore, Style, init as colorama_init

colorama_init()

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.3  # seconds
BAR_WIDTH = 30


def float32_to_int16(samples):
    return np.clip(samples * 32768, -32768, 32767).astype(np.int16)


def load_silero_vad():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    return model, get_speech_timestamps


def save_audio_file(audio, folder, category, metadata_writer, phrase=None):
    os.makedirs(folder, exist_ok=True)
    uid = str(uuid.uuid4())
    path = os.path.join(folder, f"{uid}.wav")
    write(path, SAMPLE_RATE, float32_to_int16(audio))
    metadata_writer.writerow({
        "uuid": uid,
        "category": category,
        "phrase": phrase or "",
        "duration_sec": len(audio) / SAMPLE_RATE,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"\n💾 Saved: {path}")


def record_category(category, phrase, output_dir, metadata_writer,
                    vad_model, get_speech_timestamps,
                    threshold=0.01, speech_prob_threshold=0.35):
    click.echo(f"\n🎙️ Recording {category} samples — speak naturally.")
    click.echo("Press Ctrl+C to stop recording.\n")
    click.echo(Fore.YELLOW +
               "🕓 Wait until VAD speech probability drops (GREEN) before saying the next phrase "
               "to ensure clean segmentation." + Style.RESET_ALL)
    click.echo(Fore.CYAN +
               "🎧 Adjust your mic gain between recordings to simulate different volume levels.\n"
               + Style.RESET_ALL)

    folder = os.path.join(output_dir, category)
    buffer = []
    chunk_samples = int(SAMPLE_RATE * 0.3)  # 300 ms chunks
    smoothing_window = 5
    recent_probs = []

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            while True:
                chunk, _ = stream.read(chunk_samples)
                audio_chunk = chunk[:, 0]
                buffer.extend(audio_chunk)

                # Noise / loudness level
                noise_level = np.sqrt(np.mean(audio_chunk ** 2))
                noise_bar = "█" * int(min(noise_level * 2000, BAR_WIDTH))

                # Run Silero VAD on 512-sample frames
                tensor = torch.from_numpy(audio_chunk).float()
                n_frames = len(tensor) // 512
                probs = []
                for i in range(n_frames):
                    frame = tensor[i * 512:(i + 1) * 512]
                    if len(frame) == 512:
                        p = vad_model(frame, SAMPLE_RATE).item()
                        probs.append(p)
                speech_prob = float(np.mean(probs)) if probs else 0.0

                # Smooth VAD output for stability
                recent_probs.append(speech_prob)
                if len(recent_probs) > smoothing_window:
                    recent_probs.pop(0)
                smooth_prob = np.mean(recent_probs)

                speech_bar = "█" * int(smooth_prob * BAR_WIDTH)

                # Display two-line meter
                color = Fore.RED if smooth_prob > speech_prob_threshold else Fore.GREEN
                print(Style.RESET_ALL + "\r" +
                      f"🔊 Noise:  {noise_level:.3f} {noise_bar}\n" +
                      color + f"🗣️  Speech Prob: {smooth_prob:.3f} {speech_bar}" + Style.RESET_ALL,
                      end="\033[F")  # move cursor up one line to overwrite next iteration

                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n⏹️ Stopped recording. Processing segments...")

        audio = np.array(buffer, dtype=np.float32)
        tensor_audio = torch.from_numpy(audio)

        if category == "background":
            total_len = len(audio)
            num_segments = max(1, total_len // (SAMPLE_RATE * 3))
            for _ in range(num_segments):
                start = np.random.randint(0, max(0, total_len - SAMPLE_RATE * 3))
                segment = audio[start:start + SAMPLE_RATE * 3]
                save_audio_file(segment, folder, category, metadata_writer)
        else:
            speech_timestamps = get_speech_timestamps(
                tensor_audio,
                vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=speech_prob_threshold,  # more sensitive
            )
            if not speech_timestamps:
                click.echo(Fore.YELLOW + "⚠️ No speech detected in this session." + Style.RESET_ALL)
                return

            for segment in speech_timestamps:
                start, end = segment["start"], segment["end"]
                clip = audio[start:end]
                save_audio_file(clip, folder, category, metadata_writer, phrase)


@click.command()
@click.option('--output', '-o', default='wakeword_dataset', help='Output directory')
@click.option('--threshold', '-t', default=0.01, help='Energy threshold for display bar')
def main(output, threshold):
    click.echo("🎧 Personal Wake Word Dataset Collector")
    click.echo("======================================\n")

    os.makedirs(output, exist_ok=True)
    metadata_path = os.path.join(output, "metadata.csv")
    with open(metadata_path, mode='w', newline='') as csvfile:
        fieldnames = ["uuid", "category", "phrase", "duration_sec", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        instructions = f"""
📋 INSTRUCTIONS FOR HIGH-QUALITY DATA
-------------------------------------
- Reflect real-world conditions where your wake-word will be deployed.
- Use a quiet room or minimal background noise. Noise can be simulated later.
- Speak naturally and clearly, leaving pauses between samples.
- Wait for the VAD (RED -> GREEN) to go quiet before saying the next phrase.
- Vary your voice — pitch, tone, distance, emotion.
- Adjust microphone gain between samples to simulate different volumes.
- For background noise, record several seconds of typical room sound.
- Re-run this script on different microphones or devices to diversify your dataset.
- Speak at various distances from the microphone and loudness to mimic real world conditions.

During recording:
- GREEN = silence/background
- RED = voice detected
- Ctrl+C stops each recording session.

"""
        click.echo(instructions)


        wake_word = click.prompt("Enter your wake word (e.g., 'Hey Nova')")

        click.echo("\n📦 Loading Silero VAD model (first time may take a moment)...")
        vad_model, get_speech_timestamps = load_silero_vad()
        vad_model.eval()

        categories = [
            ("wake_word", f"Say your wake word: '{wake_word}' repeatedly, naturally."),
            ("not_wake_random", "Say random phrases NOT containing your wake word. Include similar/rhyming/partial words."),
            ("background", "Record silence or ambient noise."),
        ]

        for label, prompt_text in categories:
            click.echo(f"\n--- {label.upper()} ---")
            click.echo(prompt_text)
            record_category(label, wake_word if 'wake' in label else None,
                            output, writer, vad_model, get_speech_timestamps, threshold)

    click.echo("\n✅ Dataset collection complete!")
    click.echo(f"All data and metadata saved in: {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
