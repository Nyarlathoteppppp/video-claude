"""The weather card clip: Cycles plate, a lens's halation round whatever clipped, a trace of film grain."""
import numpy as np
from scipy.ndimage import gaussian_filter

from styles_common import seq, surface


def glass(t):
    rgb = seq("glass", t)
    small = rgb[::2, ::2]
    hot = np.clip(small.max(axis=2, keepdims=True) - 0.985, 0, None) / 0.015
    light = small * hot                                         # whatever clipped glows in its own colour: a gold sun gold, a red one red
    halo = gaussian_filter(light, (16, 16, 0)) * 0.22 + gaussian_filter(light, (52, 52, 0)) * 0.3
    halo = np.repeat(np.repeat(halo, 2, 0), 2, 1)[:rgb.shape[0], :rgb.shape[1]] * np.array((1.0, 0.72, 0.42), np.float32)
    out = 1 - (1 - rgb) * (1 - np.clip(halo, 0, 1))
    grain = np.random.default_rng(int(t * 30) + 11).normal(0, 0.006, out.shape[:2])[..., None].astype(np.float32)
    return surface(np.clip(out + grain, 0, 1))
