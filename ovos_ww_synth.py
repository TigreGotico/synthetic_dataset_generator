import os
import random
import sys
import time
import warnings
from os.path import join
from typing import List, Dict
from uuid import uuid4

import click
import torch
import torchaudio as ta
from chatterbox_onnx import ChatterboxOnnx
from ovos_tts_plugin_edge_tts import EdgeTTSPlugin, VOICES
from ovos_tts_plugin_google_tx import GoogleTranslateTTS
from ovos_tts_plugin_piper import PiperTTSPlugin
from ovos_tts_plugin_piper.voice_models import get_available_voices
from ovos_utils.lang import standardize_lang_tag
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

console = Console()


def load_model() -> ChatterboxOnnx:
    with console.status("Loading VC model...", spinner="dots"):
        model = ChatterboxOnnx()
    console.log("[green]VC model loaded successfully[/green]")
    return model


def _collect_tts_metadata(lang: str, edge: bool = True, google: bool = True, piper: bool = False) -> List[Dict]:
    """
    Collects metadata for all available TTS plugins (no instantiation yet).
    Each dict contains: plugin_type, name, config
    """
    console.log(f"Collecting TTS metadata for [bold]{lang}[/bold]")
    lang_prefix = lang.split("-")[0].lower()
    metadata = []

    # Edge voices
    if edge:
        rate_variations = [f"+{r}%" for r in range(35) if r > 0] + [f"-{r}%" for r in range(30) if r > 0]
        for l, voices in VOICES.items():
            if not l.startswith(lang_prefix):
                continue
            for v in voices:
                rate = random.choice(rate_variations)
                metadata.append({"type": "edge", "name": f"edge_{v}", "config": {"voice": v, "rate": rate}})

    # Google
    if google:
        metadata.append({"type": "google", "name": f"google_{lang}",
                         "config": {"lang": lang, "slow": random.choice([True, False])}})

    # Piper
    if piper:
        for voice, data in get_available_voices(update_voices=False).items():
            l = standardize_lang_tag(data["language"]["code"])
            if not l.startswith(lang_prefix):
                continue
            n = len(data["speaker_id_map"])
            voices_list = [f"{voice}#{i}" for i in range(n)] if n > 0 else [voice]
            for v in voices_list:
                metadata.append({"type": "piper", "name": f"piper_{l}_{v}", "config": {"lang": l, "voice": v}})

    console.log(Panel(f"Total TTS configurations: [bold green]{len(metadata)}[/bold green]", title="TTS Scan"))
    return metadata


def _instantiate_plugin(plugin_meta: Dict):
    """Instantiate a plugin from its metadata."""
    if plugin_meta["type"] == "edge":
        return EdgeTTSPlugin(config=plugin_meta["config"])
    if plugin_meta["type"] == "google":
        return GoogleTranslateTTS(config=plugin_meta["config"])
    if plugin_meta["type"] == "piper":  # TODO replace with phoonnx, supports piper and more
        return PiperTTSPlugin(config=plugin_meta["config"])
    # TODO - chatterbox
    raise ValueError(f"Unknown TTS type: {plugin_meta['type']}")


def _ensure_wav_tensor(wave_obj, sr: int = 22050):
    import numpy as np
    if isinstance(wave_obj, torch.Tensor):
        return (wave_obj.unsqueeze(0) if wave_obj.dim() == 1 else wave_obj, sr)
    if isinstance(wave_obj, np.ndarray):
        wav = torch.from_numpy(wave_obj)
        return (wav.unsqueeze(0) if wav.dim() == 1 else wav, sr)
    if isinstance(wave_obj, str) and os.path.exists(wave_obj):
        return ta.load(wave_obj)
    raise ValueError("Unsupported waveform object")


def synthesize_and_convert(
        wake_word: str,
        lang: str,
        output_dir: str,
        reference_voices_dir: str,
        n: int,
        edge: bool = True, google: bool = True, piper: bool = False
):
    """Generate and convert N samples, each with a unique (TTS plugin, reference voice) combination."""
    all_meta = _collect_tts_metadata(lang, edge=edge, google=google, piper=piper)
    if not all_meta:
        console.log("[red]No TTS metadata found[/red]")
        return

    try:
        from pathlib import Path
        base_path = Path(reference_voices_dir)
        # The '**/*.wav' pattern finds all .wav files, recursively (**)
        ref_voices = [str(p) for p in base_path.glob("**/*.wav")]
        if not ref_voices:
            console.log(f"[yellow]No reference voices (.wav) found recursively in {reference_voices_dir}[/yellow]")
            return

        console.log(f"[green]Found {len(ref_voices)} reference voices.[/green]")

    except FileNotFoundError:
        console.log(f"[red]Reference voice directory not found: {reference_voices_dir}[/red]")
        return

    os.makedirs(output_dir, exist_ok=True)
    vc_model = load_model()

    success, fail = 0, 0
    start = time.time()

    for i in tqdm(range(n), total=n, desc="Synth + VC", unit="sample", file=sys.stdout):
        plugin_meta = random.choice(all_meta)
        ref_file = random.choice(ref_voices)

        # print(plugin_meta, ref_file)

        # Instantiate the TTS plugin on-the-fly
        try:
            plugin = _instantiate_plugin(plugin_meta)
        except Exception as e:
            fail += 1
            console.log(f"[red]Failed to instantiate plugin {plugin_meta['name']}: {e}[/red]")
            continue

        base_name = str(uuid4())[10:]
        tts_path = join(output_dir, f"{base_name}_tts.wav")
        vc_path = join(output_dir, f"{base_name}.wav")
        ref_path = join(reference_voices_dir, ref_file)

        try:
            # 1️⃣ Generate TTS
            plugin.get_tts(wake_word.replace("_", " ").replace("-", " "),
                           tts_path,
                           lang=lang,
                           voice=plugin_meta["config"].get("voice"))

            # 2️⃣ Voice conversion
            vc_model.voice_convert(source_audio_path=tts_path,
                                   target_voice_path=ref_path,
                                   output_file_name=vc_path)
            success += 1
        except Exception as e:
            fail += 1
            console.log(f"[red]Failed sample {i + 1}/{n} ({plugin_meta['name']} → {ref_file}): {e}[/red]")

        os.remove(tts_path)

    elapsed = time.time() - start
    console.print(Panel(f"Completed [green]{success}[/green] / [red]{fail}[/red] samples "
                        f"in {elapsed:.1f}s ({success}/{n} total)", title="Synth + VC Summary"))


@click.command()
@click.option("--wakeword", "-w", required=True, help="Wake word phrase (e.g. 'Hey Mycroft').")
@click.option("--lang", "-l", required=True, help="Language code (e.g. en-US, pt-PT).")
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False, writable=True),
              help="Output folder for generated samples.")
@click.option("--vc-refs", required=True, type=click.Path(file_okay=False, exists=True),
              help="Folder with reference voices (wav).")
@click.option("--n", "-n", default=1000, show_default=True, help="Number of random samples to generate.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--edge", is_flag=True, default=False, help="Include Edge TTS engine.")
@click.option("--google", is_flag=True, default=False, help="Include Google TTS engine.")
@click.option("--piper", is_flag=True, default=False, help="Include Piper TTS engine.")
@click.option("--suppress-warnings", is_flag=True, help="Suppress Python warnings.")
def cli(
        wakeword: str, lang: str, output: str, vc_refs: str, n: int, seed: int,
        edge: bool, google: bool, piper: bool, suppress_warnings: bool
):
    if not edge and not google and not piper:
        raise ValueError("must pass at least one of --edge --google or --piper")
    if suppress_warnings:
        warnings.filterwarnings("ignore")
    if seed is not None:
        random.seed(seed)
        console.log(f"[blue]Using random seed:[/blue] {seed}")

    synthesize_and_convert(
        wake_word=wakeword.replace("_", " ").replace("-", " "),
        lang=lang,
        output_dir=output,
        reference_voices_dir=vc_refs,
        n=n,
        edge=edge,
        google=google,
        piper=piper
    )


if __name__ == "__main__":
    cli()
