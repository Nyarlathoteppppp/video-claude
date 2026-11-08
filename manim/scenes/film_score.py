"""《字成精》's score, generated locally with ACE-Step 1.5 XL. No vocals anywhere. Run with ACE-Step's own Python, from its folder, through the slot:

    cd ~/audio_models/ACE-Step-1.5 && nice -n 10 zsh ~/render_work/宣传/tools/ink_slot.sh .venv/bin/python ~/render_work/宣传/manim/scenes/film_score.py theme credits

Writes ink/film_out/film/sound/score/CUE_takeN.flac. Opus cannot hear these, so every cue comes in three takes for 奈亚子 to choose from.

The film is quiet (film.director.md 3.8): the whole theme plays once only, when the poem is written full in 十四; before that only fragments. The fragments are not
generated on their own: film_mix.py cuts them out of the chosen theme take (its opening, a little more of it each time), so they are the theme and nothing else.
Take N of the theme and take N of the credits are the same instruments, so that a pick of one family carries through.
"""
import argparse
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "ink", "film_out", "film", "sound", "score")
ACE = os.path.expanduser("~/audio_models/ACE-Step-1.5")
COMMON = "instrumental, no vocals, no choir, no drums, no percussion, very sparse, quiet, intimate close recording in a small quiet room, natural soft reverb, pentatonic, slow"

# the three families, one per take
FAMILY = ("solo guqin, plucked strings and harmonics, each note left to ring and die away",
          "solo felt piano, very soft, single notes and a few open chords, the felt muting every hammer",
          "guqin with a very soft low cello holding long notes underneath")
CUES = dict(
    # 十四: the fourth row written, the signature, then the poem read whole, about 16 s; cut and faded to the picture in film_mix
    theme=dict(dur=24, bpm=60, key="D minor", cap="the main theme stated once in full: a simple, plain melody of a few notes, unhurried, tender and quietly certain, rising once and settling, ending on one low long note and silence"),
    # the 146 s roll: warm, a little lighter, still sparse; it must leave room for the little crab's sounds and stop dead under the thanks
    credits=dict(dur=150, bpm=66, key="D minor", cap="end credits: the same simple melody returning gently and wandering a little, warm, a touch lighter and playful, gentle walking pace, long gaps between phrases, never building up, never loud, it ends quietly"),
)


def main():
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler
    p = argparse.ArgumentParser()
    p.add_argument("cue", nargs="+")
    p.add_argument("--takes", type=int, default=3)
    p.add_argument("--model", default="acestep-v15-xl-sft")
    p.add_argument("--steps", type=int, default=50)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dit = AceStepHandler()
    print(dit.initialize_service(project_root=ACE, config_path=a.model, device="auto", use_mlx_dit=True), flush=True)
    llm = LLMHandler()
    for name in a.cue:
        c = CUES[name]
        for take in range(1, a.takes + 1):
            t0 = time.time()
            seed = 5000 + 1000 * take + len(name)
            params = GenerationParams(caption=f"{FAMILY[(take - 1) % len(FAMILY)]}. {c['cap']}. {COMMON}", lyrics="[Instrumental]", instrumental=True, bpm=c["bpm"],
                                      keyscale=c["key"], duration=c["dur"], inference_steps=a.steps, seed=seed, thinking=False)
            config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False, seeds=[seed])
            r = generate_music(dit, llm, params, config, save_dir=OUT)
            if not r.success:
                print(name, take, "FAILED", r.error, flush=True)
                continue
            dst = os.path.join(OUT, f"{name}_take{take}.flac")
            if os.path.exists(dst):
                n = next(n for n in range(1, 1000) if not os.path.exists(os.path.join(OUT, f"{name}_take{take}_old{n}.flac")))
                os.rename(dst, os.path.join(OUT, f"{name}_take{take}_old{n}.flac"))
            os.replace(r.audios[0]["path"], dst)
            print(f"{dst}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
