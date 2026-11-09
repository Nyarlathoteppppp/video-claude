"""The score of the ink film, generated locally with ACE-Step 1.5 XL. No vocals anywhere. Run with ACE-Step's own Python, from its folder:

    cd ~/audio_models/ACE-Step-1.5 && .venv/bin/python ~/render_work/宣传/manim/scenes/ink_score.py read --takes=3

Writes ink/bgm/CUE_take1.flac ... next to a line of timing per take. I cannot hear these, so every cue comes in several takes for the user to choose from.
"""
import argparse
import os
import time

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "ink", "bgm")
ACE = os.path.expanduser("~/audio_models/ACE-Step-1.5")
COMMON = "instrumental, no vocals, no choir, no drum kit, cinematic, Chinese traditional instruments, recorded in a large quiet hall, natural reverb, pentatonic"

# one cue per stretch of the film, see SCORE in ink_mix.py; durations leave room to be cut to the picture
CUES = dict(
    myth=dict(dur=26, bpm=50, key="D minor", cap="mythic and terrifying ancient ritual music: huge bronze bianzhong bells struck slowly, enormous low war drums far away, a deep drone, the weight of heaven and earth, sparse, slow, awe"),
    ages=dict(dur=16, bpm=96, key="D minor", cap="a journey through three thousand years, guqin and pipa ostinato slowly accelerating, low strings rising underneath, frame drums joining, gathering like a river in flood, ends at its loudest"),
    awake=dict(dur=20, bpm=54, key="D minor", cap="out of deep silence, a very low bowed string drone grows into an overwhelming swell of erhu and low strings with a single great bronze bell, a revelation, something enormous becoming aware"),
    read=dict(dur=72, bpm=66, key="D minor", cap="solo guqin, intimate and unhurried, a quiet melody flowing like a river in morning mist, a bamboo xiao answering from far away, tender, lonely, very sparse"),
    bed=dict(dur=28, bpm=60, key="D major", cap="slow rising swell of low strings and guqin harmonics, sunrise over wet stone, warm, deeply moving, climaxing gently and then settling"),
    windows=dict(dur=40, bpm=52, key="A minor", cap="very sparse solo felt piano with soft guqin harmonics, single notes like distant lights coming on one by one in the dark, wide empty space, then fewer and fewer notes, ending almost in silence"),
    channel=dict(dur=28, bpm=60, key="D major", cap="moonlit river, solo cello and guqin playing a calm, steady, consoling melody, not sad, quietly sure of itself"),
    grief=dict(dur=60, bpm=44, key="D minor", cap="solo bamboo xiao, breathy and hesitant, long silences between phrases, distant rain, restrained grief, honest, never sentimental"),
    names=dict(dur=36, bpm=58, key="D minor", cap="stone and bronze: slow single bianzhong bells and a low drum, four heavy statements with silence between, restrained, ancient, ending unresolved"),
    new=dict(dur=26, bpm=50, key="D major", cap="a single sustained guqin harmonic and very high soft strings, open, weightless, an unknown new thing, nothing resolves"),
    poem=dict(dur=42, bpm=60, key="D minor", cap="solo guqin plays the main quiet melody one last time, unhurried, each note left to die away, ending on one low open string and silence"),
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cue", nargs="+")
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--model", default="acestep-v15-xl-sft")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--dur", type=int, default=0)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dit = AceStepHandler()
    print(dit.initialize_service(project_root=ACE, config_path=a.model, device="auto", use_mlx_dit=True), flush=True)
    llm = LLMHandler()
    for name in (list(CUES) if a.cue == ["all"] else a.cue):
        c = CUES[name]
        for take in range(1, a.takes + 1):
            t0 = time.time()
            params = GenerationParams(caption=f"{c['cap']}. {COMMON}", lyrics="[Instrumental]", instrumental=True, bpm=c["bpm"], keyscale=c["key"],
                                      duration=a.dur or c["dur"], inference_steps=a.steps, seed=1000 * take + len(name), thinking=False)
            config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False, seeds=[1000 * take + len(name)])
            r = generate_music(dit, llm, params, config, save_dir=OUT)
            if not r.success:
                print(name, take, "FAILED", r.error, flush=True)
                continue
            src = r.audios[0]["path"]
            dst = os.path.join(OUT, f"{name}_take{take}.flac")
            os.replace(src, dst)
            print(f"{dst}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
