"""The new film's sound effects, from the same local MOSS-SoundEffect model as ink_sfx.py. With its own Python, through the slot:

    cd ~/audio_models/MOSS-TTS/moss_soundeffect_v2 && TORCHDYNAMO_DISABLE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python ~/render_work/宣传/manim/scenes/film_sfx.py press lift room typing --takes=3

Writes ink/film_out/film/sound/NAME_takeN.wav.

    python film_sfx.py mix          # with the render env's numpy and scipy: 三's press, rough, on T3 from 0 to the end of the dawn: sound/press_rough.wav

The press, as 三 has it (film.director.md 3.8): every strike its own stuff and a little heavier than the one before, the first dull, the fifth hard, the sixth the return
key and a burst of keys, a mouse click before the seventh, the seventh heaviest. Each strike's tail runs on into the next shot, the first six only. After the seventh nothing
but one quiet breath out, 1.3 to 2.1 s after it lands, then silence through the dawn. Each sound is put where its own onset lands on the frame of the contact.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "ink", "film_out", "film", "sound")

SOUNDS = dict(
    press=(4, "One single heavy impression of an old cast iron hand printing press: the iron screw turns, the heavy platen comes down hard on thick paper with a deep dull thud of metal and wood, then stillness, inside a quiet workshop. No music, no voices."),
    lift=(5, "A heavy iron printing press platen being lifted very slowly off thick paper: a low creak of the iron screw and a soft dry peel of paper, getting quieter, then silence. No music, no voices."),
    room=(12, "The quiet of an ordinary apartment room on a calm afternoon with the window a little open: very faint distant street traffic far away, the soft air of the room, almost nothing else. No music, no voices, no birds close by."),
    typing=(10, "A person typing slowly on a modern laptop keyboard with soft low travel keys, close up, dry, in a quiet room, individual key clicks clearly separated, a short pause now and then. No music, no voices, no mouse."),
    enter=(3, "A single press of the return key on a modern laptop keyboard, one soft firm click, close up and dry, in a quiet room, and nothing else. No music, no voices."),
    flywheel=(10, "A giant heavy cast iron flywheel spinning down after the motor is switched off. It starts fast with a loud low whirring hum, then the pitch falls steadily and the sound gets quieter and quieter, whoosh, whoosh, each turn slower than the last, until it stops dead and the last second is complete silence. No music, no voices."),
    # 三's seven strikes, each its die's own stuff going into the same wet clay
    strike_clay=(3, "A heavy hand-made pottery stamp pressed down hard into a thick slab of soft wet clay on a wooden table: one deep, dull, muffled thud with a short damp squelch, then silence. Close and dry, in a quiet room at night. No music, no voices, no metal."),
    strike_wood=(3, "A heavy carved wooden printing block pressed hard into a thick slab of wet clay on a wooden table: one dull wooden knock with a damp squelch, then silence. Close and dry, in a quiet room at night. No music, no voices."),
    strike_bronze=(3, "A heavy old bronze seal pressed hard into a thick slab of dense wet clay: a dull thud with a faint low ring of metal and a short squelch, then silence. Close and dry, in a quiet room at night. No music, no voices."),
    strike_lead=(3, "A heavy block of lead printing type pressed hard into dense wet clay: a dense dead thud with a tiny clatter of loose metal type, then silence. Close and dry, in a quiet room at night. No music, no voices."),
    strike_hard=(3, "A heavy steel die driven hard and fast into stiff clay: one sharp, hard, tight knock of metal into clay, short, then silence. Close and dry, in a quiet room at night. No music, no voices."),
    keys_fast=(3, "Fast confident typing on a modern laptop keyboard with soft low keys, close up, crisp, a quick burst of key clicks, then the return key struck firmly once, then silence. Quiet room at night. No music, no voices."),
    mouse=(2, "A single click of a computer mouse button, close up and dry, in a quiet room at night, and nothing else. No music, no voices."),
    strike_last=(4, "The heaviest blow of all: a huge dark steel die slammed down into a deep bed of wet clay on a wooden table, a deep low crushing boom, the table shudders once, wet clay squeezes out, then complete silence. Close up, at night. No music, no voices."),
    exhale=(3, "One very soft, tired human breath out through the nose, late at night, close, in a silent room, and nothing else at all. No music, no words."),
    # the rest of the film (film_mix.py lays them): 3 takes each, 2026-11-07
    brush=(8, "A soft wet Chinese calligraphy brush writing slowly on thin absorbent xuan rice paper laid on a felt pad, close up: the gentle dry hiss and whisper of the hair dragging across the paper, one stroke after another with short pauses between, in a silent room. No music, no voices, no tapping."),
    seal_heavy=(3, "A heavy carved stone seal pressed down firmly onto a sheet of xuan paper on a thick felt pad: one soft, deep, muffled thump, then stillness, close and dry in a quiet room. No music, no voices, no metal, no ringing."),
    seal_light=(3, "A small carved stone seal set down gently onto a sheet of paper on a felt pad: one very soft, light, muffled tap, almost nothing, close and dry in a silent room. No music, no voices."),
    whoosh=(2, "A small heavy stone falling a short way through still air: a very faint, soft swish of air, close, in a silent room, and nothing else. No impact, no music, no voices."),
    peel=(3, "A stone seal slowly lifted off a sheet of slightly damp paper: a faint soft sticky peel as the paper lets go, then silence, close and dry in a quiet room. No music, no voices."),
    page_air=(5, "In a quiet old room at night a soft draft of air comes through and lifts one page of an old open paper book: the page rises with a faint rustle and falls back softly onto the others, then stillness. No music, no voices."),
    rain_frame=(12, "Light rain at dusk on an old wooden window frame and its sill, close outside the window: soft, sparse, slow drops tapping the wood, a gentle drizzle beyond, heard from inside a quiet room. No thunder, no music, no voices."),
    drop=(2, "One single drop of rain water falling from the edge of a wooden window frame onto a stone sill below: one small, clear, soft plip, close, and nothing else. No music, no voices."),
    lamp=(12, "A small oil lamp burning in a silent old room at night: the faint soft flutter of its flame and an almost silent hush of air, very quiet, nothing else. No crackling fire, no music, no voices."),
    room_night=(12, "The quiet of a room at night with the windows shut, very late: an almost silent room tone, the faint low hum of a computer and the soft air of the room, nothing moving. No music, no voices, no traffic, no clock ticking."),
    city_night=(12, "A residential neighbourhood late at night heard from an open window high up: very faint distant traffic, the low hum of air conditioner units on nearby balconies, a calm wide quiet. No sirens, no music, no voices, no dogs."),
    chew=(4, "A tiny creature chewing a small mouthful of soft clay: little soft wet squishy munches, gentle and cute, close up, in a quiet room. No human voice, no music, no crunching."),
    slurp=(4, "A small creature slurping one long thin noodle into its mouth: one soft, wet, quick slurp, gentle and cute, close up, in a quiet room, then a tiny pop at the end. No human voice, no music."),
    gulp=(2, "A small creature swallowing a mouthful with one soft wet gulp, cute, close up, in a quiet room, and nothing else. No human voice, no music."),
    riverbed=(12, "A vast dry riverbed of cracked clay seen from high above in the morning: a thin, dry, steady wind over bare earth, very wide and empty, very quiet, no water at all. No music, no voices, no birds."),
    scratch=(6, "A thin wooden stick slowly drawing a line into soft wet clay, close up: a fine, quiet, slightly gritty scraping of the point through the clay, going, stopping, then going again, in a silent room. No music, no voices, no chalk."),
    # second goes at the two above, which came out as hiss and as rumble (2026-11-07): new names, so the first takes stay as they are
    furrow=(6, "A single thin stick or reed drawn slowly through soft wet clay, very close: a soft, low, muffled dragging furrow, the damp clay pushed aside, smooth and slow, going, pausing, then going on again, in a quiet room. Soft low-mid sound, no hiss, no scraping, no scratching, no chalk. No music, no voices."),
    # a third go in other words, since MOSS heard "clay" as hiss and "wind" as rumble both times
    mud_drag=(6, "Slow wet mud squelch: a blunt wooden tool pushed slowly through thick wet clay, a soft muffled squishing drag, low and wet, close and quiet, going and pausing. No hiss, no high frequencies, no scraping. No music, no voices."),
    air_dry=(12, "Soft high airy hiss of a light breeze through dry grass and over dry ground, thin and delicate, very quiet and steady, no low frequencies, no rumble, no gusts. No music, no voices."),
    dry_wind=(12, "A thin, dry, airy wind moving over a wide plain of cracked dry earth, very quiet and still, only a faint breath of air and a little dust, steady, no gusts. No low rumble, no thunder, no water, no birds, no insects. No music, no voices."),
    clay_morph=(5, "Soft wet clay slowly squeezing and folding by itself on an old wooden board: gentle wet squelches and soft sticky creaks of clay, a few tiny dry crumbs pattering onto the wood, close up, in a quiet room. No music, no voices."),
    clay_yawn=(4, "A tiny soft creature made of clay giving one small sleepy yawn: a faint, soft, stretching squeak of moist clay rising and falling, breathy and cute, not a human voice, close up, in a quiet room. No music, no words."),
    clay_step=(4, "Small soft clay feet walking a few steps across an old wooden table: tiny, muffled, soft pats, light and cute, close up, in a quiet room. No music, no voices."),
    clay_thud=(2, "A small lump of soft clay set down gently on an old wooden board: one soft muffled pat, close and dry, in a quiet room, and nothing else. No music, no voices."),
    cry=(10, "Far away across dark mountains at night, one long, hollow, mournful cry, like wind through a hollow stone rising into a wail and slowly dying away, echoing across the valley, then fainter cries answering from distant ridges. Not a human voice, no music, no thunder."),
    grain_mud=(6, "Dry grains of millet falling from the sky onto soft wet mud: at first a thick, soft pattering rain of grains, then thinning out to a few last single grains, then silence. No music, no voices, no thunder."),
)
# 三's press on T3: (sound, take, where its onset lands, gain, how long after its onset it is kept). Contacts as film_imprint_clay.BLOWS has them; the keys run up to the
# return key, which is the sixth strike. Takes chosen by their envelopes (one clean hit each; strike_hard take 2 and the quiet exhale takes are noise): not by ear.
CONTACTS = (0.9, 5.5, 8.9, 11.5, 13.5, 15.0, 16.1)
MIX = (("strike_clay", 3, CONTACTS[0], .5, 1.2), ("strike_wood", 3, CONTACTS[1], .56, 1.2), ("strike_bronze", 1, CONTACTS[2], .62, 1.2), ("strike_lead", 3, CONTACTS[3], .68, 1.2),
       ("strike_hard", 3, CONTACTS[4], .78, 1.0), ("keys_fast", 1, None, .32, None), ("strike_hard", 1, CONTACTS[5], .7, .6), ("mouse", 1, 15.9, .42, .2),
       ("strike_last", 1, CONTACTS[6], 1.0, 1.15), ("exhale", 2, CONTACTS[6] + 1.6, .3, 2.0))
MIX_END = 21.2          # the end of the dawn: from the breath to here, nothing
KEYS_FROM = 13.7        # the keys only under the return key's shot (s4_key), not over the fourth and fifth strikes


def onset(a, part=.2):
    """Where a sound starts: its first sample over part of its peak."""
    import numpy as np
    return int(np.argmax(np.abs(a) > part * np.abs(a).max()))


def mix():
    """三's press, rough, mono at the takes' rate. Each take scaled to its peak, then by its gain; the keys placed so that their last click, the return key, lands on the sixth strike."""
    import numpy as np
    from scipy.io import wavfile
    rate, out = None, None
    for name, take, at, gain, keep in MIX:
        r, a = wavfile.read(os.path.join(OUT, f"{name}_take{take}.wav"))
        a = (a.astype(np.float32) / (32768.0 if a.dtype == np.int16 else 1.0))
        a = a.mean(1) if a.ndim == 2 else a
        if rate is None:
            rate, out = r, np.zeros(int(MIX_END * r), np.float32)
        assert r == rate, (name, r, rate)
        a = a / (np.abs(a).max() + 1e-9) * gain
        if keep:            # the first hit only, its tail let go over the last third
            on, n = onset(a), int(keep * r)
            a = a[:on + n].copy()
            f = int(n / 3)
            a[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        if at is None:          # the keys: the last loud click is the return key
            loud = np.flatnonzero(np.abs(a) > .5 * np.abs(a).max())
            start = int(CONTACTS[5] * rate) - int(loud[-1])
            cut = max(int(KEYS_FROM * rate) - start, 0)
            a = a.copy()
            a[:cut] = 0
            a[cut:cut + int(.04 * rate)] *= np.linspace(0, 1, len(a[cut:cut + int(.04 * rate)]), dtype=np.float32)
        else:
            start = int(at * rate) - onset(a)
        a = a[max(-start, 0):]
        start = max(start, 0)
        end = min(start + len(a), len(out))
        out[start:end] += a[:end - start]
        print(f"{name} take {take} from {start / rate:.3f} s", flush=True)
    # after the seventh, only the breath: anything else still sounding goes before it
    hush = int((CONTACTS[6] + 1.2) * rate)
    last = [m for m in MIX if m[0] == "exhale"][0]
    r, b = wavfile.read(os.path.join(OUT, f"{last[0]}_take{last[1]}.wav"))     # the breath again, as it was put in, to take it out from under the hush
    fade = np.clip(1 - (np.arange(len(out)) - hush) / (.25 * rate), 0, 1).astype(np.float32)
    breath = np.zeros_like(out)
    b = (b.astype(np.float32) / 32768.0) if b.dtype == np.int16 else b.astype(np.float32)
    b = (b.mean(1) if b.ndim == 2 else b) / (np.abs(b).max() + 1e-9) * last[3]
    on, n = onset(b), int(last[4] * r)
    b = b[:on + n].copy()
    b[-int(n / 3):] *= np.linspace(1, 0, int(n / 3), dtype=np.float32)
    s0 = int(last[2] * rate) - onset(b)
    breath[s0:s0 + len(b)] = b[:len(breath) - s0]
    out = (out - breath) * fade + breath
    peak = np.abs(out).max()
    out = out / max(peak, 1.0) * .9
    path = os.path.join(OUT, "press_rough.wav")
    wavfile.write(path, rate, (out * 32767).astype(np.int16))
    print(path, f"{len(out) / rate:.2f} s at {rate} Hz, T3 0 to {MIX_END}", flush=True)


def main():
    if sys.argv[1:2] == ["mix"]:
        return mix()
    import ink_sfx
    import soundfile
    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--takes", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ink_sfx.fit_for_mps()
    pipe = MossSoundEffectPipeline.from_pretrained(ink_sfx.MODEL, torch_dtype=torch.bfloat16, device="mps")
    for name in a.names:
        seconds, prompt = SOUNDS[name]
        for take in range(1, a.takes + 1):
            t0 = time.time()
            audio = pipe(prompt=prompt, seconds=seconds, num_inference_steps=a.steps, cfg_scale=4.0, seed=100 * take + len(name))     # the pipeline seeds its own noise, torch.manual_seed changes nothing
            path = os.path.join(OUT, f"{name}_take{take}.wav")
            wav = audio.detach().cpu().float()
            wav = wav[0] if wav.ndim == 3 else wav.unsqueeze(0) if wav.ndim == 1 else wav
            soundfile.write(path, wav.numpy().T, pipe.sample_rate)
            print(f"{path}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
