"""The style reel's music, generated locally with ACE-Step 1.5 XL. No vocals. Run with ACE-Step's own Python, from its folder:

    cd ~/audio_models/ACE-Step-1.5 && .venv/bin/python ~/render_work/宣传/manim/scenes/styles_score.py --takes=3

Writes styles/film_out/bgm/take1.flac ... The reel cuts every 138 frames at 30 fps, 4.6 s, which is 8 beats at 104.35 BPM.
styles_mix.py measures each take's real beat and stretches it onto that grid.
"""
import argparse
import os
import time

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "styles", "film_out", "bgm")
ACE = os.path.expanduser("~/audio_models/ACE-Step-1.5")
CAPTION = ("instrumental showreel music, cinematic electronic, analog synth arpeggio, punchy tight drums on a steady four-on-the-floor pulse, "
           "deep sidechained bass, a clear one-bar pickup into a big confident drop, bright and modern, keeps building, wide stereo, no vocals, no choir")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--dur", type=int, default=84)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--model", default="acestep-v15-xl-sft")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dit = AceStepHandler()
    print(dit.initialize_service(project_root=ACE, config_path=a.model, device="auto", use_mlx_dit=True), flush=True)
    llm = LLMHandler()
    for take in range(1, a.takes + 1):
        t0 = time.time()
        seed = 7100 + take
        params = GenerationParams(caption=CAPTION, lyrics="[Instrumental]", instrumental=True, bpm=104, keyscale="A minor",
                                  duration=a.dur, inference_steps=a.steps, seed=seed, thinking=False)
        config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False, seeds=[seed])
        r = generate_music(dit, llm, params, config, save_dir=OUT)
        if not r.success:
            print(take, "FAILED", r.error, flush=True)
            continue
        dst = os.path.join(OUT, f"take{take}.flac")
        os.replace(r.audios[0]["path"], dst)
        print(f"{dst}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
