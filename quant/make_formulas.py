r"""Typeset the film's formulas with LaTeX into black-on-white PNGs (marked terms in pure colours) (quant/formulas/<name>.png).

The film renderer turns each PNG into an alpha mask and paints it in any colour, so one set of images serves every scene.
Terms marked \T1..\T6 print in pure red, green, blue, magenta, cyan and yellow, so the renderer can cut each into its own
mask and light it up while the picture explains that part (quant_film.py, formula_masks).
Needs pdflatex (TeX Live) and the PDF rasteriser tools/pdfpage.swift (macOS); run it on the Mac that has TeX.

    python3 quant/make_formulas.py                       # all formulas
    PDFPAGE=/path/to/compiled/pdfpage python3 quant/make_formulas.py
"""
import os
import re
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "formulas")
PX_PER_PT = 12                         # 10 pt maths -> glyphs about 100 px tall: crisp at any size the film uses
PDFPAGE = shlex.split(os.environ.get("PDFPAGE", f"swift {os.path.join(ROOT, 'tools', 'pdfpage.swift')}"))

FORMULAS = {   # \T1{..} .. \T6{..} mark the terms the film lights up one by one; each prints in its own pure colour
    "bachelier": r"S_t = \T1{S_0} + \T2{\sigma W_t}",
    "markowitz": r"\min_{w}\; \T1{w^{\top}\Sigma\, w} \quad \text{s.t.}\quad \T2{w^{\top}\mu = R}",
    "kelly": r"\T4{f^{*}} = \T1{p} - \dfrac{\T2{1-p}}{\T3{b}}",
    "sharpe": r"S = \dfrac{\T1{R_p - R_f}}{\T2{\sigma_p}}",
    "black_scholes": r"C = \T1{S_0\,N(d_1)} - \T2{K e^{-rT} N(d_2)}",
    "grinold": r"\T1{\mathrm{IR}} = \T2{\mathrm{IC}} \times \T3{\sqrt{N}}",
    "fama_french": r"\T1{r_i - r_f} = \alpha_i + \T2{\beta_i (r_m - r_f)} + \T3{s_i\,\mathrm{SMB}} + \T4{h_i\,\mathrm{HML}} + \varepsilon_i",
    "factor": r"r_i = \alpha_i + \T1{\sum_{k=1}^{K} \beta_{ik}\, f_k} + \T2{\varepsilon_i}",
    "ou": r"dX_t = \theta\,(\mu - X_t)\,dt + \sigma\, dW_t",
    "edge": r"\mathbb{E}[\,\Pi_N\,] = \T1{N\varepsilon} \qquad \sigma(\Pi_N) = \T2{\sigma\sqrt{N}}",
    "macd": r"\T1{\mathrm{DIF}} = \T2{\mathrm{EMA}_{12}} - \T3{\mathrm{EMA}_{26}}",
    "chern_simons": r"\mathrm{CS}(A) = \frac{k}{4\pi}\int_M \mathrm{tr}\Big(\T1{A \wedge dA} + \T2{\tfrac{2}{3}\, A \wedge A \wedge A}\Big)",
}

TEMPLATE = r"""\pdfcompresslevel=0 \pdfobjcompresslevel=0
\documentclass[preview,border=3pt]{standalone}
\usepackage{amsmath,amssymb,xcolor}
\newcommand{\T}[2]{\textcolor[RGB]{#1}{#2}}
\newcommand{\Ta}[1]{\T{255,0,0}{#1}}\newcommand{\Tb}[1]{\T{0,255,0}{#1}}\newcommand{\Tc}[1]{\T{0,0,255}{#1}}
\newcommand{\Td}[1]{\T{255,0,255}{#1}}\newcommand{\Te}[1]{\T{0,255,255}{#1}}\newcommand{\Tf}[1]{\T{255,255,0}{#1}}
\begin{document}
$\displaystyle %s$
\end{document}
"""


def build(name, tex, tmp):
    src = os.path.join(tmp, name + ".tex")
    tex = re.sub(r"\\T([1-6])\{", lambda m: "\\T" + "abcdef"[int(m[1]) - 1] + "{", tex)   # \T1{ -> \Ta{
    with open(src, "w") as fh:
        fh.write(TEMPLATE % tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=tmp, check=True,
                   stdout=subprocess.DEVNULL)
    pdf = os.path.join(tmp, name + ".pdf")
    box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", open(pdf, "rb").read())
    assert box, f"{name}: no MediaBox in {pdf}"           # the page is left uncompressed so it can be read here
    width_pt = float(box[3]) - float(box[1])
    png = os.path.join(OUT, name + ".png")
    subprocess.run(PDFPAGE + [pdf, "0", png, str(round(width_pt * PX_PER_PT))], check=True, stdout=subprocess.DEVNULL)
    print(f"{name:14s} {width_pt:6.1f} pt -> {png}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(FORMULAS)
    with tempfile.TemporaryDirectory() as tmp:
        for n in names:
            build(n, FORMULAS[n], tmp)
