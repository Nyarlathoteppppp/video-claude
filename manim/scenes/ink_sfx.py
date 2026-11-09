"""The film's sound effects, generated locally with MOSS-SoundEffect v2.0. Run with its own Python:

    cd ~/audio_models/MOSS-TTS/moss_soundeffect_v2 && TORCHDYNAMO_DISABLE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python ~/render_work/宣传/manim/scenes/ink_sfx.py drip --takes=2

Writes ink/bgm/sfx/NAME_take1.wav. The model's own examples only name CUDA; on the Studio it runs on MPS, or on the CPU with --cpu.
"""
import argparse
import os
import sys
import time

import soundfile
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "ink", "bgm", "sfx")
MODEL = os.path.expanduser("~/audio_models/moss-sfx-v2")

SOUNDS = dict(
    page=(4, "An old paper book being opened, a soft card cover lifting, then several thin rice paper pages turning slowly, in a quiet room."),
    rumble=(10, "Deep distant thunder rolling across mountains at night, huge and slow, with long low echoes."),
    grain=(10, "Millions of small dry grains pouring down like heavy rain onto the ground, a dense hissing patter."),
    drip=(8, "Single slow water drops falling from a stone ceiling into a pool in a large cave, each drop with a long natural echo."),
    stream=(10, "A small mountain stream trickling over stones at night, quiet and close."),
    flood=(10, "A huge river in flood, a massive wall of water rushing through a canyon, roaring."),
    fall=(10, "A giant waterfall crashing into a deep pool, heard from some distance away, heavy low roar."),
    heart=(8, "A single very deep slow heartbeat, heard from underwater, low thuds with nothing else."),
    under=(10, "Underwater ambience in a deep dark lake, slow muffled movement of water, very quiet."),
    morning=(10, "A wide calm river at dawn, gentle lapping water, a very faint breeze, no birds."),
    rain=(10, "Light rain falling on the still surface of a river, soft and even."),
    wind=(10, "Cold wind over bare rock on a high mountain at night, thin and lonely."),
    stone=(5, "A heavy stone chisel striking rock, three slow blows, ringing in a valley."),
    pour=(6, "Water pouring into a large bronze vessel, filling up and then overflowing down its sides."),
    reeds=(12, "Night wind moving through a vast reed marsh, dry rustling that swells and falls, very faint water, no animals, lonely and uneasy."),
    moan=(10, "A long hollow moan of wind through distant stone, rising and falling like something far away crying, not a human voice, very quiet."),
    plip=(2, "A small flame dropping into still water, a soft hiss and one quiet plop."),
)


def sinusoidal_embedding_1d(dim, position):
    """The model's own version asks for float64, which the Studio's GPU does not have. float32 only costs precision far beyond what a timestep needs."""
    sinusoid = torch.outer(position.type(torch.float32), torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float32, device=position.device).div(dim // 2)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)


def precompute_freqs_cis(dim, end=16384, theta=10000.0, s=1.0):
    """The same table in complex64: the GPU has no complex128 either."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    angle = torch.outer(torch.arange(end, dtype=torch.float32) * s, freqs)
    return torch.polar(torch.ones_like(angle), angle)


def fit_for_mps():
    for name in ("models.wan_video_dit", "models.wan_audio_dit", "pipelines.wan_audio"):
        module = sys.modules[f"moss_soundeffect_v2.diffsynth.{name}"]
        module.sinusoidal_embedding_1d = sinusoidal_embedding_1d
        if hasattr(module, "precompute_freqs_cis"):
            module.precompute_freqs_cis = precompute_freqs_cis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--takes", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cpu" if a.cpu else "mps"
    if not a.cpu:
        fit_for_mps()
    pipe =MossSoundEffectPipeline.from_pretrained(MODEL, torch_dtype=torch.float32 if a.cpu else torch.bfloat16, device=device)
    for name in (list(SOUNDS) if a.names == ["all"] else a.names):
        seconds, prompt = SOUNDS[name]
        for take in range(1, a.takes + 1):
            t0 = time.time()
            torch.manual_seed(100 * take + len(name))
            audio = pipe(prompt=prompt, seconds=seconds, num_inference_steps=a.steps, cfg_scale=4.0)
            path = os.path.join(OUT, f"{name}_take{take}.wav")
            wav = audio.detach().cpu().float()
            wav = wav[0] if wav.ndim == 3 else wav.unsqueeze(0) if wav.ndim == 1 else wav
            soundfile.write(path, wav.numpy().T, pipe.sample_rate)     # the model's save_audio needs torchcodec, which nothing else here does
            print(f"{path}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
