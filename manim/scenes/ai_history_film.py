"""AI 发展路线图 — the film renderer (vertical 1080x1920, 30 fps). Storyboard: ai_history.scenes.md (newest revision first).

v3 (2026-10-18): a 3-second opening; the window never goes black (drawings stay, cards keep the last picture); GPT,
Claude, DeepSeek, image/video generation and "beat the best humans" nodes with their numbers; flashes for the smaller
breakthroughs; foreshadowing lines (此时，离……还有 N 年) and echoes (the old seed lights up, a spark travels to the new node);
bursts sized by impact; faster lift after ChatGPT; the music loops past the end of the cycle.
v2 (user notes on v1, 2026-10-18): one line (no school strands), the song in its original order, an opening that says what
the line means, every node explained in the top-left window (title, plain one-liner, Chinese for English names),
厚积薄发 as "the window shrinks and flies into the head", a bottom time axis whose unit shrinks (years -> months -> days),
the stop "时间，开始加速了。" before ChatGPT, a feed of well-known breakthroughs only, and a final pull-back overview with
every important date labelled.

Not Manim: one continuous, data-driven camera move is easier as a pure function of time. Every frame is
render_frame(t) on a cairo surface (vectors, text) plus numpy (glow, plates).

    PY=python3        # a Python with pycairo, numpy, scipy, Pillow (and ffmpeg on PATH)
    $PY ai_history_film.py snap 3 20 45 70          # PNGs + contact sheet -> ai_history/film_out/snaps/
    $PY ai_history_film.py render 0 328.0 --jobs 8 --name ai_history_v7   # mp4 with the BGM -> ai_history/film_out/
    $PY ai_history_film.py check                     # data and timeline sanity checks

Inputs: ai_history/milestones.csv (dates, weights; Pi-verified), ai_history/bgm/bgm_v3.wav (the song cycle, original order,
four quiet bars cut in the first winter) and bgm_v3.beats.csv (every beat, from manim/sfx/analyze_bgm.py), ai_history/images.
The releases (cold open + film, on the beat) are put together by assemble_v7.sh.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import unicodedata
from functools import lru_cache

import cairo
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.normpath(os.path.join(HERE, "..", "..", "ai_history"))
IMG = os.path.join(PROJ, "images")
BGM = os.path.join(PROJ, "bgm")
OUT = os.path.join(PROJ, "film_out")
ORIENT = os.environ.get("AIH_ORIENT", "portrait")          # landscape: the Bilibili 16:9 cut, same story and timing
W, H, FPS = (1920, 1080, 30) if ORIENT == "landscape" else (1080, 1920, 30)
LAYOUT = {
    # portrait (Douyin): the line in the lower middle, window top left, clock and pictures top right
    "portrait": dict(x_left=70.0, x_left_ov=70.0, x_head=0.64, y_top=0.43, y_base=0.78, y_top_ov=0.26, y_base_ov=0.72,
                     clock_y=250, photo=(26, 300, 470, 380), faction_y=(712, 760), closing=(60, 330, 64), closing_panel=False,
                     formula_slots=[(70, 1738, 52), (430, 1822, 48), (110, 1898, 44)]),
    # landscape (Bilibili): window in the left column, the line in the middle, clock / pictures / flashes on the right
    "landscape": dict(x_left=600.0, x_left_ov=90.0, x_head=0.72, y_top=0.53, y_base=0.83, y_top_ov=0.22, y_base_ov=0.68,
                      clock_y=150, photo=(26, 220, 460, 330), faction_y=(582, 630), closing=(560, 200, 48), closing_panel=True,
                      formula_slots=[(640, 220, 44), (900, 350, 40), (660, 480, 38)]),
}[ORIENT]
SERIF = "Songti SC"
LATIN = "Times New Roman"


# ======================================================================================
# colour
# ======================================================================================
def hx(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


PAL = {
    "mono": dict(top="#0a0a0a", bot="#141414", glow="#ddd8ce", cream="#dcd7cd", core="#f3f0ea", plate="#cfcac0"),
    "warm": dict(top="#0b0907", bot="#15110c", glow="#f6d99c", cream="#efe3c6", core="#fff4d8", plate="#efe3c6"),
    "cold": dict(top="#07090c", bot="#0d1117", glow="#9fb6d0", cream="#c9d3df", core="#e6edf6", plate="#c9d3df"),
}
SCHOOL = {
    "rule": dict(name="规则派", color="#c9d3df"),
    "learn": dict(name="学习派", color="#f2c98a"),
    "trial": dict(name="试错派", color="#d9826b"),
}
COLD_HEX = "#9fb3c8"


def mix(a, b, k):
    return a * (1 - k) + b * k


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_in(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x


# ======================================================================================
# data: verified milestones -> one momentum curve (setbacks weighted up so the winters read on a single line)
# ======================================================================================
def date_to_year(d):
    p = [int(x) for x in d.split("-")]
    if len(p) == 1:
        return p[0] + 0.5
    if len(p) == 2:
        return p[0] + (p[1] - 0.5) / 12
    return p[0] + (p[1] - 1) / 12 + (p[2] - 0.5) / 365


GRID = np.arange(1940.0, 2027.5, 0.005)
SETBACK_GAIN = 3.0          # a winter is the loss of money and belief, felt by the whole field, not one school
STORY_END = 2026.7951       # 2026-10-18


@lru_cache(None)
def data():
    rows = []
    with open(os.path.join(PROJ, "milestones.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(t=date_to_year(r["date"]), w=float(r["weight"]), label=r["cn_label"], tier=r["tier"],
                             original=r["original"], span=float(r["span_years"] or 0), date=r["date"]))
    idx = np.zeros_like(GRID)
    for r in rows:
        t, w = r["t"], r["w"]
        if w < 0:
            a, b, w = t, t + min(r["span"] or 3.0, 6.0), w * SETBACK_GAIN
        else:
            w *= 1 + 1.5 * float(smooth((t - 2022.9) / 0.6))   # after ChatGPT every step lifts the line harder
            tau = 3.2 if t < 1990 else 1.8 if t < 2005 else 1.0 if t < 2016 else 0.6 if t < 2024 else 0.35
            a, b = t - tau, t
        idx += w * smooth((GRID - a) / (b - a))
    for bt in BEATS:                                   # big breakthroughs: an extra, sharp step that lands on the burst
        if bt.get("major") or bt.get("lift"):
            idx += (bt.get("major", 0) + bt.get("lift", 0)) * smooth((GRID - (bt["year"] - 0.12)) / 0.12)
    for d_, _, _, tag_ in FLASH:                       # after ChatGPT every flash is one more visible step
        y_ = date_to_year(d_)
        if y_ > 2022.95 and tag_ != "新闻":
            idx += 22 * smooth((GRID - (y_ - 0.02)) / 0.02)
    idx = np.maximum(idx, 0.0) * smooth((GRID - 1943.0) / 2.0)
    return dict(rows=rows, idx=idx, run_max=np.maximum.accumulate(idx))


def I(t):
    return np.interp(t, GRID, data()["idx"])


def winter_amount(t):
    v = 0.0
    for a, b in ((1974.0, 1980.0), (1987.0, 1993.0)):
        v = max(v, float(smooth((t - a) / 1.2) * (1 - smooth((t - b) / 1.5))))
    return v


# ======================================================================================
# music (the song in its original order) and the story clock
# ======================================================================================
BGM_FILE, BGM_JSON = "bgm_v3.wav", "bgm_v3.json"   # v3: four quiet bars cut in the first winter


@lru_cache(None)
def downbeats():
    cyc = []
    with open(os.path.join(BGM, "bgm_v3.beats.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["beat_in_bar"] == "1":
                cyc.append(float(r["t"]))
    return cyc


BAR = 3.866
BEAT = BAR / 4


@lru_cache(None)
def beat_times():
    """Every beat of the song (bgm_v3), extended a little past both ends at the bar's pulse."""
    with open(os.path.join(BGM, "bgm_v3.beats.csv")) as fh:
        b = np.array([float(r["t"]) for r in csv.DictReader(fh)])
    cycle = 294.558                                     # bgm_v3 length: the song starts over (the render loops it)
    return np.concatenate([b[0] - BEAT * np.arange(2, 0, -1), b, b + cycle])


def snap(t, half=True):
    """Nearest beat (half=True: nearest half-beat) — everything that pops lands on the music."""
    b = beat_times()
    g = np.sort(np.concatenate([b, (b[:-1] + b[1:]) / 2])) if half else b
    return float(g[int(np.argmin(np.abs(g - t)))])
KEYS = [
    (0, 1943.0), (1.37, 1943.95), (3, 1944.3), (5.248, 1945.49), (6.9, 1945.8), (9.102, 1950.8), (10.8, 1951.1),
    (16.834, 1956.6), (18.5, 1956.9), (20.689, 1958.55), (22.4, 1958.9), (28.421, 1966.05), (30.1, 1966.4),
    (32.276, 1969.05), (34, 1969.6), (36.153, 1973.4), (37.9, 1973.7), (43.886, 1975.0),
    (55.472, 1980.0), (59.349, 1980.45),
    (61.058, 1980.8), (67.082, 1983.5), (70.936, 1986.78), (72.558, 1986.95), (78.668, 1987.2), (90.255, 1990.5),
    (94.11, 1992.3), (95.758, 1992.6), (101.842, 1995.5),
    (105.72, 1997.37), (107.358, 1997.55), (109.597, 1997.9), (111.158, 1998.1), (113.452, 1998.9), (114.958, 1999.3),
    (117.306, 2006.5), (118.858, 2006.8), (121.161, 2009.4), (122.658, 2009.7), (125.039, 2012.75), (126.758, 2012.95),
    (128.893, 2013.2), (132.771, 2014.45), (134.467, 2014.6), (136.625, 2015.9), (138.267, 2016.0), (144.358, 2016.2),
    (148.212, 2016.55), (152.078, 2017.08), (153.758, 2017.22), (155.944, 2017.45), (157.667, 2017.6), (159.799, 2017.79),
    (161.458, 2017.85), (163.677, 2018.82), (167.531, 2019.28), (169.267, 2019.6), (171.398, 2020.41), (173.058, 2020.55),
    (175.263, 2020.91), (176.967, 2020.96), (179.128, 2021.01), (180.758, 2021.12), (182.996, 2022.6), (184.667, 2022.62),
    (186.85, 2022.66), (188.558, 2022.7),
    (194.582, 2022.86), (197.958, 2022.865), (198.437, 2022.915),          # the stop before ChatGPT
    (200.267, 2022.98), (202.315, 2023.2), (203.958, 2023.45), (205.058, 2023.85), (206.169, 2024.124), (207.858, 2024.2), (210.047, 2024.36),
    (211.658, 2024.42), (213.901, 2024.56), (215.667, 2024.6), (217.779, 2024.70), (219.467, 2024.73), (221.634, 2024.77),
    (223.558, 2024.79), (225.498, 2024.806), (227.158, 2024.9), (229.366, 2025.05), (231.067, 2025.1), (233.22, 2025.195),
    (234.958, 2025.22), (237.098, 2025.25), (238.858, 2025.3), (240.953, 2025.38), (242.558, 2025.42), (244.807, 2025.55),
    (246.558, 2025.57), (248.685, 2025.595), (252.548, 2025.68), (256.417, 2025.88), (260.258, 2026.079), (264.126, 2026.118),
    (268.004, 2026.173),
    (271.858, 2026.265), (273.158, 2026.28), (275.736, 2026.385), (276.867, 2026.40), (279.591, 2026.585), (280.667, 2026.6),
    (283.445, 2026.69), (284.867, 2026.71), (287.323, 2026.765), (289.867, STORY_END), (314.558, STORY_END + 1e-4),
]
_story = PchipInterpolator([k[0] for k in KEYS], [k[1] for k in KEYS], extrapolate=True)


def story(t):
    return min(float(_story(t)), STORY_END)


# Every named node gets a window: an algorithm drawing (viz), a big number (readout) or a text card. Title and a
# plain one-line explanation always; English names carry their Chinese meaning in brackets. Facts: milestones.csv,
# quotes.csv (Pi-verified). Short explanations where the music gives the node only one bar.
B = lambda at, year, label, title, desc, **kw: dict(at=at, year=year, label=label, title=title, desc=desc, **kw)
BEATS = [
    B(1.37, 1943.95, "神经元模型", "神经元模型 · 1943", "第一次用数学描述脑细胞：信号加起来，超过阈值就「点亮」", viz="neuron", image="neuron_p1.png",
      story="神经元「全或无」，所以能用逻辑来描述"),
    B(5.248, 1945.49, "冯·诺依曼结构", "冯·诺依曼结构 · 1945", "程序和数据放进同一个存储器：现代电脑的基本结构", viz="vonneumann",
      image="vonneumann_lanl.png", lead_fixed=3.0),
    B(9.102, 1950.8, "图灵测试", "图灵测试 · 1950", "图灵问「机器能思考吗」：隔着屏幕聊天，分不出是人是机器", viz="turing",
      image="turing_aged16.jpg|vonneumann_lanl.png", faction=True, story="图灵预言：到世纪末，说机器在思考，大概不会再有人反驳", fore="75 年后，这个测试会被 AI 通过"),
    B(16.834, 1956.6, "达特茅斯会议", "达特茅斯会议 · 1956", "一群科学家开会，讨论怎样让机器像人一样学习和思考", viz="tree", image="dartmouth_p1.png", echo=[(1943.95, "神经元模型"), (1950.8, "图灵测试")], first="第一次有了「人工智能」这个名字",
      story="提案原话：挑一批科学家，一个夏天就能做出重大进展"),
    B(20.689, 1958.55, "感知机", "感知机 · 1958", "最早能从例子里自己学习的机器之一", viz="perceptron", image="mark1_perceptron_fig2.png", echo=[(1943.95, "神经元模型")],
      story="《纽约时报》：海军期待它能走、说、看、写，并自觉存在"),
    B(28.421, 1966.05, "ELIZA 聊天程序", "ELIZA 聊天程序 · 1966", "抓关键词、套模板回复，看起来像在聊天", viz="chat", image="eliza_doctor.png", echo=[(1950.8, "图灵测试")], first="第一次，人能和电脑「对话」",
      story="作者的秘书和它聊了几句，就请他先离开房间"),
    B(32.276, 1969.05, "神经网络遭质疑", "感知机遭质疑 · 1969", "指出单层网络的硬伤，研究降温", viz="xor", setback=True,
      story="书中指出：单层感知机连「异或」都算不了", fore="神经网络，要等 43 年才翻身"),
    B(36.153, 1973.4, "莱特希尔报告", "莱特希尔报告 · 1973", "英国政府的报告批评 AI 没兑现承诺，经费随后被砍", card=True, setback=True,
      story="报告原话：没有任何发现，兑现了当初许诺的重大影响"),
    B(43.886, 1975.0, "第一次寒冬", "第一次 AI 寒冬 · 1974–1980", "承诺的成果迟迟做不出来，经费被大幅削减", card=True, setback=True,
      story="西蒙曾预言：二十年内，机器能做任何人能做的工作"),
    B(59.349, 1980.45, "专家系统", "专家系统 · 1980 年代", "把专家经验写成成千上万条「如果……就……」规则", viz="rules", image="symbolics_lisp_chm.jpg", echo=[(1956.6, "达特茅斯会议")],
      story="DEC 的配置系统，每年净回报估计超过四千万美元"),
    B(70.936, 1986.78, "反向传播", "反向传播 · 1986", "把误差从后往前一层层传回去修正，多层神经网络终于能训练", viz="backprop", image="backprop_p1.png", echo=[(1958.55, "感知机"), (1969.05, "异或难题"), (1974.62, "Werbos 1974")],
      story="方法有了，可当年的算力和数据都撑不起大网络", fore="这个方法还要等 26 年，等来显卡"),
    B(78.668, 1987.2, "第二次寒冬", "第二次 AI 寒冬 · 1987–1993", "专家系统又贵又难维护，专用电脑市场崩溃，AI 再次失宠", card=True, setback=True,
      image="symbolics_lisp_chm.jpg"),
    B(94.11, 1992.3, "TD-Gammon 自学双陆棋", "TD-Gammon 自学双陆棋 · 1992", "自己和自己反复对弈，下到接近人类高手", viz="selfplay", echo=[(1986.78, "反向传播")],
      story="从零自学到中级，加上特征后接近顶尖"),
    B(105.72, 1997.37, "深蓝", "深蓝 · 1997", "IBM 超级计算机，靠海量搜索下棋", viz="tree", image="deep_blue_chm.jpg", first="第一次，电脑战胜国际象棋世界冠军", cat="超越人类",
      story="六局 3½ 比 2½。卡斯帕罗夫：我失去了斗志", fore="围棋，还要再守 19 年"),
    B(109.597, 1997.9, "LSTM 长短期记忆网络", "LSTM 长短期记忆网络 · 1997", "给网络加上「记忆格」", viz="lstm", story="后来撑起了语音识别和机器翻译", echo=[(1986.78, "反向传播")]),
    B(113.452, 1998.9, "卷积网络", "卷积网络 · 1998", "小窗口滑过图片找特征", viz="conv", image="lenet5_fig.png", echo=[(1980.29, "新认知机"), (1986.78, "反向传播"), (1989.92, "LeNet 1989")], story="已商用，每天读几百万张支票"),
    B(117.306, 2006.5, "深度学习", "深度学习 · 2006", "多层网络重新被训练起来", viz="layers", image="dl2006_p1.png", story="1969 年被冷落的神经网络，又回来了", echo=[(1969.05, "神经网络遭质疑"), (1986.78, "反向传播")]),
    B(121.161, 2009.4, "ImageNet 图像数据集", "ImageNet 图像数据集 · 2009", "几百万张标注好的图片", viz="mosaic", image="imagenet_p1.png", story="320 万张图、5247 类：为深度学习的爆发备好了数据"),
    B(125.039, 2012.75, "AlexNet 深度卷积网络", "AlexNet 深度卷积网络 · 2012", "认图错误率 15.3%，第二名 26.2%", viz="cnn", image="alexnet_p1.png", major=30,
      story="1986 年的反向传播，终于等来了显卡算力", echo=[(1969.05, "神经网络遭质疑"), (1986.78, "反向传播"), (1998.9, "卷积网络"), (2009.4, "ImageNet")]),
    B(132.771, 2014.45, "GAN 生成对抗网络", "GAN 生成对抗网络 · 2014", "一个网络造假，一个网络识假，越练越逼真", viz="gan", image="gan_p1.png|gan_samples.png", echo=[(2012.75, "AlexNet")],
      story="论文里的人脸和数字，都是它自己画的", cat="生成"),
    B(136.625, 2015.9, "ResNet 残差网络", "ResNet 残差网络 · 2015", "网络能叠到 152 层深", viz="resnet", image="resnet_p1.png",
      story="认图错误率降到 3.57%，人类约 5%", echo=[(1958.55, "感知机"), (2012.75, "AlexNet")]),
    B(144.358, 2016.2, "AlphaGo", "AlphaGo · 2016", "人类下了两千多年的围棋，它 4:1 赢了李世石", viz="mcts", image="press/alphago_lee_sedol.png", first="第一次，AI 下围棋赢了顶尖棋手", major=45, cat="超越人类", echo=[(1992.3, "TD-Gammon"), (1997.37, "深蓝"), (2012.75, "AlexNet")],
      story="李世石：只是我输了，不是人类输了"),
    B(152.078, 2017.08, "Libratus 德州扑克", "Libratus 德州扑克 · 2017.01", "一对一德州扑克，对战四位顶尖职业选手", readout=("12 万手", "20 天对局，AI 赢了"),
      first="第一次，AI 在德州扑克里赢了顶尖职业选手", story="扑克要猜对手的牌、会诈唬，它都学会了", major=15, cat="超越人类"),
    B(155.944, 2017.45, "Transformer 注意力机制", "Transformer 注意力机制 · 2017", "谷歌提出：每个词都「关注」句子里的其他词；ChatGPT 里的 T 就是它", viz="attention", image="attention_p1.png", first="后来的大模型，几乎都建在它上面",
      story="论文标题 Attention Is All You Need\n意思是：注意力就是你所需要的一切", fore="仅仅 5 年后，它撑起了 ChatGPT", major=120, echo=[(1997.9, "LSTM")]),
    B(159.799, 2017.79, "AlphaGo Zero", "AlphaGo Zero · 2017", "不看人类棋谱，从零开始自己和自己下", viz="elo", image="agz_p1.png",
      story="100 比 0，战胜击败李世石的那一版", major=40, echo=[(1992.3, "TD-Gammon"), (2015.9, "ResNet"), (2016.2, "AlphaGo")]),
    B(163.677, 2018.82, "BERT 语言理解模型", "BERT 语言理解模型 · 2018", "谷歌：先读海量文本，做「完形填空」", viz="cloze", image="bert_p1.png", echo=[(2017.45, "Transformer")]),
    B(167.531, 2019.28, "OpenAI Five 电竞", "OpenAI Five 电竞 · 2019.04", "5 个 AI 组队打电竞游戏 Dota 2", readout=("2 : 0", "击败世界冠军战队 OG"), image="openai_five_p1.png", echo=[(1997.9, "LSTM"), (2017.79, "AlphaGo Zero")],
      first="第一次，AI 在电竞比赛里击败世界冠军", story="训练时自己和自己打了相当于 4.5 万年的游戏", major=20, cat="超越人类"),
    B(171.398, 2020.41, "GPT-3 大语言模型", "GPT-3 大语言模型 · 2020", "1750 亿个参数，只做一件事：猜下一个词", viz="params", image="gpt3_p1.png",
      story="它为两年后的 ChatGPT 打下了基础", major=80, echo=[(2017.45, "Transformer"), (2017.92, "百度 · 规模定律"), (2018.43, "GPT-1"), (2019.12, "GPT-2")]),
    B(175.263, 2020.91, "AlphaFold 2 蛋白质结构", "AlphaFold 2 · 2020", "从氨基酸序列算出蛋白质的三维结构", viz="protein", image="press/alphafold_structure.jpg", echo=[(2017.45, "Transformer")],
      story="蛋白质评测创始人：这个问题我们卡了将近五十年", fore="这项工作，后来拿了 2024 年诺贝尔化学奖"),
    B(179.128, 2021.01, "DALL·E 文字生成图像", "DALL·E 文字生成图像 · 2021", "输入一句话，画出从没存在过的图", card=True, image="press/dalle_radish.jpg", echo=[(2017.45, "Transformer"), (2020.41, "GPT-3")],
      story="官方示例：一只穿芭蕾舞裙、遛狗的萝卜宝宝", cat="生成"),
    B(182.996, 2022.6, "Stable Diffusion 开源", "Stable Diffusion 开源 · 2022.08", "从一团噪声开始一步步去噪，还原出图像；模型开源，普通显卡就能跑",
      viz="diffusion", image="ldm_samples.png", story="2015 年提出的扩散模型，冷门多年后成了主角", major=40, echo=[(2014.45, "GAN"), (2015.2, "扩散模型 · 2015"), (2017.45, "Transformer"), (2021.01, "DALL·E")], cat="生成"),
    B(186.85, 2022.66, "AI 画作夺冠", "AI 画作夺冠 · 2022.08", "Midjourney 生成的画，拿下美国州博览会美术比赛数字艺术类第一名", card=True,
      image="press/theatre_opera_spatial.jpg", story="两位评委事后说，当时不知道 Midjourney 是 AI", cat="生成"),
    B(198.437, 2022.915, "ChatGPT", "ChatGPT · 2022.11", "基于 GPT-3.5 的聊天助手；两个月，月活用户估计破亿", readout=("GPT-3.5 上线", "5 天，100 万用户"), image="press/altman_tweet_card.png",
      story="1966 年的 ELIZA 只会套模板；56 年后，AI 真的会聊天了", major=150, cat="对话",
      echo=[(1966.05, "ELIZA"), (2017.45, "Transformer"), (2020.41, "GPT-3")]),
    B(202.315, 2023.2, "GPT-4", "GPT-4 · 2023.03", "能看懂图片，推理更强", viz="bars:gpt4", image="gpt4_exams.png",
      story="模拟律师资格考试：从后 10% 到前 10%", major=90, cat="大模型", echo=[(2017.45, "Transformer"), (2022.915, "ChatGPT")]),
    B(206.169, 2024.124, "Sora 视频生成", "Sora 视频生成 · 2024.02", "研究预览：一句话生成最长约一分钟的视频", viz="video",
      story="技术报告的标题：视频生成模型是世界模拟器", lift=40, cat="视频",
      echo=[(2015.2, "扩散模型"), (2017.45, "Transformer"), (2022.965, "DiT"), (2023.24, "吃面 · 2023")]),
    B(210.047, 2024.36, "GPT-4o", "GPT-4o · 2024.05", "文字、声音、图像，一个模型实时对话", viz="voice", image="gpt4o_card_p1.png",
      story="平均 320 毫秒回应，和人与人对话差不多", major=60, cat="语音", echo=[(2016.8, "语音识别"), (2017.45, "Transformer"), (2023.2, "GPT-4")]),
    B(213.901, 2024.56, "奥数银牌水平", "AlphaProof · 2024.07", "国际数学奥赛，银牌水平", readout=("28/42", "国际数学奥赛 · 银牌水平"),
      story="差 1 分到金牌线", lift=40, cat="数学", echo=[(2017.79, "AlphaGo Zero"), (2024.04, "AlphaGeometry")]),
    B(217.779, 2024.70, "o1 推理模型", "o1 推理模型 · 2024.09", "先一步步想，再回答", viz="bars:o1", image="o1_card_p1.png", echo=[(2023.2, "GPT-4")], story="想得越久，答得越准", lift=60, cat="推理"),
    B(221.634, 2024.77, "诺贝尔奖", "2024 诺贝尔奖", "物理学奖、化学奖都颁给了 AI 相关研究", card=True, image="press/nobel_hinton.jpg|press/nobel_hassabis.jpg",
      story="1986 年反向传播论文的作者之一辛顿，38 年后拿了物理学奖",
      echo=[(1943.95, "神经元模型"), (1982.29, "Hopfield 网络"), (1985.17, "玻尔兹曼机"), (1986.78, "反向传播"), (2020.91, "AlphaFold 2")], lift=30, cat="科学"),
    B(225.498, 2024.806, "AI 操作电脑", "AI 操作电脑 · 2024.10", "Claude 能看屏幕、动鼠标、敲键盘，像人一样用电脑", viz="computer", echo=[(2024.47, "Claude 3.5 Sonnet")],
      story="一个月后，Anthropic 又开源了 MCP：让 AI 接上各种工具", lift=30, cat="智能体"),
    B(229.366, 2025.05, "DeepSeek-R1", "DeepSeek-R1 · 2025.01", "国产开源推理模型，登顶美国 App Store 免费榜", viz="bars:r1", image="deepseek_r1_p1.png",
      story="一周后，英伟达单日跌 17%，市值蒸发约 5890 亿美元", major=80, cat="开源", echo=[(2017.79, "AlphaGo Zero"), (2024.70, "o1"), (2024.98, "DeepSeek-V3")]),
    B(233.22, 2025.195, "具身智能", "Gemini Robotics · 2025.03", "机器人听懂一句话就动手，没见过的任务也能做", viz="robot",
      story="1958 年的报纸就写过：它将来能走、能说、能看", echo=[(1958.55, "感知机"), (2017.45, "Transformer"), (2023.57, "RT-2"), (2024.83, "π0")], lift=40, cat="具身"),
    B(237.098, 2025.25, "图灵测试被通过", "图灵测试被通过 · 2025.03", "三方对话测试中，GPT-4.5 比真人更像真人", first="第一次，AI 通过了标准的三方图灵测试", major=120,
      viz="bars:turing", image="turing2025_fig2.png", story="图灵 1950 年提出的测试，75 年后被 AI 通过",
      echo=[(1950.8, "图灵测试"), (1966.05, "ELIZA"), (2022.915, "ChatGPT")], cat="对话"),
    B(240.953, 2025.38, "AI 视频", "AI 生成视频 · 2025.05", "同一个题目「威尔·史密斯吃面」：两年，从扭曲变成以假乱真", card=True,
      image="press/willsmith_2023_2025.png", story="2025 年的 Veo 3，还能同时生成声音", echo=[(2015.2, "扩散模型"), (2023.24, "吃面 · 2023"), (2024.124, "Sora")], lift=50, cat="视频"),
    B(244.807, 2025.55, "奥数金牌水平", "国际数学奥赛 · 2025.07", "金牌水平", viz="bars:imo", first="第一次，AI 拿到官方认证的奥数金牌", major=150,
      story="一年前还是银牌，差 1 分", echo=[(2024.04, "AlphaGeometry"), (2024.56, "银牌 · 2024"), (2024.70, "o1")], cat="数学"),
    B(248.685, 2025.595, "世界模型", "Genie 3 世界模型 · 2025.08", "一句话生成一个世界，还能实时走进去、和它互动", viz="world", image="world_models_p1.png",
      story="2018 年的论文《World Models》：让 AI 在自己的「梦」里学习", lift=50, cat="世界模型",
      echo=[(2018.24, "World Models"), (2019.88, "MuZero"), (2024.124, "Sora"), (2024.93, "Genie 2")]),
    B(252.548, 2025.68, "ICPC 编程竞赛全对", "ICPC 编程竞赛 · 2025.09", "世界总决赛：12 道题全部做对", viz="bars:icpc", echo=[(2021.49, "GitHub Copilot"), (2024.70, "o1")], lift=50, cat="超越人类"),
    B(256.417, 2025.88, "AI 编程智能体", "AI 编程智能体 · 2025", "Claude Code、Codex：自己读代码、改代码、跑测试",
      readout=("90%", "很多团队的代码由 AI 写"), story="Anthropic CEO 2025.10：很多团队如此，但不是每个团队",
      echo=[(1945.49, "冯·诺依曼结构"), (2021.49, "GitHub Copilot"), (2024.806, "AI 操作电脑"), (2025.146, "Claude Code")], lift=80, cat="编程"),
    B(260.258, 2026.079, "任务长度翻倍", "AI 能独立完成的任务 · 2026.01", "AI 有一半把握完成的任务长度：2019 年起约 7 个月翻一倍，2023 年起约 4 个月",
      viz="metr", image="metr_fig1.png", story="GPT-4：3.5 分钟\nClaude Opus 4.5：5 小时 20 分", lift=60, cat="智能体"),
    B(264.126, 2026.118, "字节 Seedance 2.0", "字节跳动 Seedance 2.0 · 2026.02", "画面和声音一起生成，还能多镜头叙事", viz="video_av",
      story="片段全网疯传，迪士尼发出停止侵权函", lift=50, cat="视频",
      echo=[(2024.124, "Sora"), (2025.38, "AI 视频"), (2025.44, "Seedance 1.0")]),
    B(268.004, 2026.173, "GPT-5.4", "GPT-5.4 · 2026.03", "OpenAI 第一个原生会操作电脑的通用模型", viz="bars:gpt54",
      story="超过了人类基线 72.4%", echo=[(2024.806, "AI 操作电脑")], lift=50, cat="超越人类"),
    B(271.858, 2026.265, "Claude Mythos 突破沙盒", "Claude Mythos · 2026.04", "测试中被要求逃出沙盒，它做到了，还给研究员发了邮件", viz="sandbox",
      story="同一个模型，在各大操作系统和浏览器里\n找出了数千个零日漏洞", lift=60, cat="安全"),
    B(275.736, 2026.385, "推翻 Erdős 猜想", "Erdős 单位距离猜想 · 2026.05", "1946 年的数学猜想，被 AI 找到反例", viz="unitdist", image="erdos_p1.png", echo=[(2023.95, "FunSearch"), (2025.365, "AlphaEvolve")],
      story="高尔斯：这无疑是人工智能数学的一座里程碑", lift=80, cat="数学"),
    B(279.591, 2026.585, "十项数学成果", "OpenAI 十项数学成果 · 2026.08", "一次公布十项研究级数学进展，每项都附机器可检验的证明", viz="math10", image="ten_advances_p1.png", echo=[(2026.385, "Erdős")], lift=80, cat="数学"),
    B(283.445, 2026.69, "千禧年难题", "Navier–Stokes 方程 · 2026.09", "千禧年难题：AI 给出了一个解答", viz="ns",
      story="克雷研究所：「看来已经被解决」，评审仍在进行", major=250, cat="数学",
      echo=[(1956.6, "达特茅斯会议"), (2025.55, "奥数金牌"), (2026.385, "Erdős")]),
    B(287.323, 2026.765, "数学手稿", "OpenAI · 2026.10", "公开的数学手稿，部分仍在验证", readout=("719 篇", "数学手稿"), echo=[(2026.585, "十项数学成果"), (2026.69, "千禧年难题")], lift=100, cat="数学"),
]
INJECT = BEAT             # the window shrinks and flies into the head during the last beat before the node
for _i, _b in enumerate(BEATS):
    gap_prev = _b["at"] - BEATS[_i - 1]["at"] if _i else 8.0
    _b["lead"] = _b.get("lead_fixed") or max(2.0, min(5.5, 0.6 * gap_prev))
    _bt = beat_times()                                  # windows open on a beat, at least two beats before the node
    _o = snap(_b["at"] - _b["lead"], half=False)
    while _b["at"] - _o < 1.5 * BEAT:
        _o = float(_bt[max(0, int(np.argmin(np.abs(_bt - _o))) - 1)])
    _b["lead"] = _b["at"] - _o
    _b["open"] = _b["at"] - _b["lead"]
def has_img(name):
    return bool(name) and all(os.path.exists(os.path.join(IMG, n)) for n in name.split("|"))


for _i, _b in enumerate(BEATS):                       # what fills the window: a drawing, the card's picture, or the last one
    if _b.get("viz"):
        _b["win"] = ("viz", _b["viz"], _b, False)
    elif _b.get("card") and has_img(_b.get("image")):
        _b["win"] = ("img", _b["image"], _b, False)
    elif _b.get("card") and not _b.get("readout"):
        _prev = next((q["win"] for q in reversed(BEATS[:_i]) if q.get("win")), None)
        if _prev:
            _b["win"] = (_prev[0], _prev[1], _prev[2], True)
STOP = (194.582, 198.437)        # the stop before ChatGPT: freeze, dim, one line, the music drops out
WALL = (288.758, 291.178)           # manuscript pages pile up, then the head breaks through
OVERVIEW = (snap(293.558, half=False), 0.0)                       # pull back to the whole line, on a beat, two beats long
OVERVIEW = (OVERVIEW[0], snap(OVERVIEW[0] + 2 * BEAT, half=False))
EXCUSE = (291.178, OVERVIEW[0] + 0.3)        # 借过一下
CLOSING = [                      # from can't to can: the gaps shrink
    ("从只会套模板，到真正会聊天", "56 年"), ("从连「异或」都算不了，到下赢围棋世界冠军", "47 年"),
    ("从会聊天，到拿奥数金牌", "不到 3 年"), ("从吃面都吃不明白，到视频以假乱真", "2 年"),
    ("从奥数银牌，到金牌", "1 年"), ("从推翻数学猜想，到千禧年难题", "111 天"),
]
CLOSING_T0, CLOSING_DT = 297.158, 1.7
CLOSING_T0 = snap(CLOSING_T0, half=False)                          # one closing line every two beats
CLOSING_TS = [snap(CLOSING_T0 + 2 * BEAT * k, half=False) for k in range(len(CLOSING))]
QUESTION_T = snap(CLOSING_TS[-1] + 2 * BEAT, half=False)
FADE = (QUESTION_T + 3.4, QUESTION_T + 4.6)
DURATION = round(FADE[1], 2)
for _b in BEATS:                                       # nothing of the next node shows during the stop
    if _b["open"] < STOP[1] and _b["at"] >= STOP[1]:
        _b["open"], _b["lead"] = STOP[1] - 0.25, _b["at"] - (STOP[1] - 0.25)
for _i, _b in enumerate(BEATS):
    _b["close"] = BEATS[_i + 1]["open"] if _i + 1 < len(BEATS) else _b["at"] + 2 * BAR
    if _i + 1 < len(BEATS) and BEATS[_i + 1]["open"] >= STOP[1] - 0.3 and _b["at"] < STOP[0]:
        _b["close"] = min(_b["close"], STOP[0])

# flashes: verified breakthroughs that are not beats pop into a list beside the head (date, name, what it did, tag)
FLASH = [
    ("1946-02-14", "ENIAC", "第一台通用电子计算机", None),
    ("2011-02-16", "IBM 沃森", "问答节目赢了人类冠军", "超越人类"),
    ("2012-06-26", "谷歌大脑", "1.6 万个 CPU 自学认出猫", None),
    ("2013-12-19", "DQN", "看着屏幕，自学打游戏", None),
    ("2014-01-26", "谷歌收购 DeepMind", "", "新闻"),
    ("2015-02-06", "ImageNet 认图", "准确率首次超过人类", "超越人类"),
    ("2015-03-12", "扩散模型", "从噪声里还原图像", None),
    ("2015-11-09", "谷歌开源 TensorFlow", "", None),
    ("2015-12-08", "百度 Deep Speech 2", "一作阿莫迪，后来创办 Anthropic", None),
    ("2015-12-11", "OpenAI 成立", "", "新闻"),
    ("2016-05-18", "谷歌 TPU", "AI 专用芯片", None),
    ("2016-09-27", "谷歌翻译", "换成神经网络翻译", None),
    ("2016-10-18", "语音识别", "错误率追平专业速记员", "追平人类"),
    ("2017-05-27", "AlphaGo", "3:0 战胜柯洁", "超越人类"),
    ("2017-12-01", "百度：规模定律", "数据越多，误差可预测地下降", None),
    ("2018-06-11", "GPT-1", "预训练加微调，1.17 亿参数", None),
    ("2019-02-14", "GPT-2", "15 亿参数，能续写长文", None),
    ("2019-03-27", "图灵奖", "授予深度学习三位先驱", "新闻"),
    ("2019-10-30", "AlphaStar", "星际争霸 2 宗师级", "超越人类"),
    ("2021-06-29", "GitHub Copilot", "AI 帮程序员写代码", None),
    ("2022-04-06", "DALL·E 2", "照片级的 AI 画图", None),
    ("2022-07-12", "Midjourney", "人人都能用 AI 画画", None),
    ("2023-02-24", "LLaMA", "开放权重的大模型", None),
    ("2023-03-16", "百度文心一言", "国产对话大模型发布", None),
    ("2023-03-22", "千人联名信", "呼吁暂停训练更强 AI 6 个月", "新闻"),
    ("2023-03-28", "威尔·史密斯吃面", "AI 视频还扭曲得离谱", None),
    ("2023-05-01", "辛顿离开谷歌", "公开警告 AI 风险", "新闻"),
    ("2023-06-07", "AlphaDev", "排序算法提速，进了 C++ 库", "论文"),
    ("2023-11-17", "OpenAI 风波", "奥特曼被罢免，5 天后回归", "新闻"),
    ("2023-11-14", "GraphCast", "10 天预报，1 分钟内算完", "论文"),
    ("2023-11-29", "GNoME", "220 万新晶体，约 800 年积累", "论文"),
    ("2023-07-28", "RT-2", "看图听令的机器人模型", "论文"),
    ("2023-12-06", "Gemini", "谷歌的多模态大模型", None),
    ("2023-12-14", "FunSearch", "在数学难题上找到新构造", "论文"),
    ("2024-01-17", "AlphaGeometry", "奥数几何 30 题做出 25 题", "论文"),
    ("2024-03-04", "Claude 3", "官方称多数常见评测领先", None),
    ("2024-05-08", "AlphaFold 3", "预测分子间的相互作用", None),
    ("2024-06-18", "英伟达", "市值登顶全球第一", "新闻"),
    ("2024-06-20", "Claude 3.5 Sonnet", "编程能力大幅提升", None),
    ("2024-06-25", "ESM3", "新荧光蛋白，约 5 亿年进化量", "论文"),
    ("2024-08-01", "欧盟 AI 法案", "正式生效", "新闻"),
    ("2024-12-20", "o3", "ARC-AGI 测试 87.5%", None),
    ("2024-12-26", "DeepSeek-V3", "训练算力约 558 万美元", None),
    ("2025-01-21", "星际之门", "5000 亿美元建 AI 算力", "新闻"),
    ("2025-02-19", "AI co-scientist", "多个 AI 一起提出科研假说", "论文"),
    ("2025-02-24", "Claude Code", "在终端里自己写代码", None),
    ("2025-03-25", "GPT-4o 画图", "吉卜力画风刷屏", None),
    ("2025-03-12", "AI Scientist-v2", "AI 写的论文过了研讨会评审", "论文"),
    ("2025-04-30", "DeepSeek-Prover-V2", "形式化证明通过率 88.9%", "论文"),
    ("2025-05-14", "AlphaEvolve", "改进 1969 年的矩阵乘法", None),
    ("2025-05-16", "Codex", "OpenAI 的云端编程智能体", None),
    ("2025-05-22", "Claude 4", "乐天实测：独立连续工作 7 小时", None),
    ("2025-06-25", "AlphaGenome", "一次读约百万个 DNA 碱基", "论文"),
    ("2025-07-09", "英伟达", "市值破 4 万亿美元", "新闻"),
    ("2025-07-17", "ChatGPT agent", "AI 助手开始替人办事", None),
    ("2025-08-07", "GPT-5", "快答和深度推理，自动切换", None),
    ("2025-08-26", "Nano Banana", "一句话改图刷屏", None),
    ("2025-09-17", "R1 登上《自然》", "推理阶段训练 29.4 万美元", "论文"),
    ("2025-10-06", "ChatGPT", "每周 8 亿用户", None),
    ("2025-10-29", "英伟达", "市值破 5 万亿美元", "新闻"),
    ("2025-11-18", "Gemini 3", "谷歌第三代，多模态再升级", None),
    ("2025-11-24", "Claude Opus 4.5", "任务长度超过 5 小时（METR）", None),
    ("2025-12-01", "DeepSeek-V3.2", "国产推理再进阶", None),
    ("2025-11-27", "DeepSeekMath-V2", "普特南 2024 测试 118/120", "论文"),
    ("2025-12-19", "Seed-Prover 1.5", "9 小时做出普特南 11 题", "论文"),
    ("2026-02-05", "Claude Opus 4.6", "首次支持百万 token 上下文", None),
    ("2026-02-11", "Aletheia", "自主解决 4 个 Erdős 问题", "论文"),
    ("2026-01-28", "AlphaGenome", "预测基因变异，登上《自然》", None),
    ("2026-04-16", "Claude Opus 4.7", "复杂编程和视觉加强", None),
    ("2026-04-23", "GPT-5.5", "编码、操作电脑、知识工作", None),
    ("2026-04-24", "DeepSeek-V4", "国产第四代", None),
    ("2026-05-28", "Claude Opus 4.8", "协作表现升级", None),
    ("2026-06-09", "Claude Fable 5", "同一底座，分两种开放权限", None),
    ("2026-06-09", "Embodied-R1.5", "开源具身模型，16 项领先", "论文"),
    ("2026-06-11", "MiniMax M3", "USAMO 2026 测试 36/42", "论文"),
    ("2026-07-09", "GPT-5.6", "主打编码、科学和网络安全", None),
    ("2026-07-16", "Kimi K3", "国产万亿参数旗舰", None),
    ("2026-07-21", "红皇后哥德尔机", "AI 和它的评委一起进化", "论文"),
    ("2026-07-24", "Claude Opus 5", "更低价格，接近 Fable 5", None),
    ("2026-07-28", "Claude Mythos", "改进对后量子签名 HAWK 的攻击", None),
    ("2026-07-28", "Claude Mythos", "改进对精简版 AES 的攻击", "论文"),
    ("2026-08-12", "Qwen3.8", "2.4 万亿参数，开放权重", None),
    ("2026-09-03", "GPT-6", "网络安全能力触及自家最高风险级", None),
    ("2026-09-22", "Claude Opus 5.5", "接近 Fable 5.1，成本更低", None),
    ("2026-09-29", "GPT-6.1", "接近 Astra，价格更低", None),
    ("2026-10-01", "BootLoops", "帮物理学家搭精确计算工具", "论文"),
    ("2026-10-13", "Gemini 4", "谷歌第四代旗舰", None),
    ("2023-04-05", "Segment Anything", "一键分出图里的任何物体", "论文"),
    ("2024-02-15", "Gemini 1.5 Pro", "一次能读 100 万 token", None),
    ("2024-05-13", "宇树 G1", "人形机器人 9.9 万元起", None),
    ("2024-09-19", "Qwen2.5", "国产开源模型家族", None),
    ("2025-04-29", "Qwen3", "开源模型也会「思考」", None),
    ("2025-05-28", "DeepSeek-R1-0528", "国产推理再升级", None),
    ("2025-06-25", "豆包 Seed1.6", "字节模型进阶", None),
    ("2025-07-11", "Kimi K2", "国产开源智能体模型", None),
    ("2025-08-21", "DeepSeek-V3.1", "快答、深思一键切换", None),
    ("2025-11-06", "Kimi K2 Thinking", "国产思考模型", None),
    ("2026-01-15", "Kimi K2.5", "多个助手并行办事", None),
    ("2026-01-22", "文心 5.0", "百度全模态大模型", None),
    ("2026-02-13", "宇树 UnifoLM", "人形机器人进工厂试装配", None),
    ("2026-02-14", "豆包 Seed2.0", "字节二代模型", None),
    ("2026-02-16", "Qwen3.5", "千问视觉推理", None),
    ("2026-02-17", "Claude Sonnet 4.6", "编程和操作电脑升级", None),
    ("2026-02-19", "Gemini 3.1 Pro", "谷歌推理升级", None),
    ("2026-02-26", "Nano Banana 2", "AI 绘图再升级", None),
    ("2026-03-25", "Lyria 3", "文字生成歌曲", None),
    ("2026-04-02", "Gemma 4", "谷歌开放模型", None),
    ("2026-04-09", "豆包全双工语音", "边听边说的语音模型", None),
    ("2026-04-14", "Gemini Robotics", "机器人学会读仪表", None),
    ("2026-04-20", "Kimi K2.6", "长程编程模型开放", None),
    ("2026-05-19", "Gemini 3.5 Flash", "谷歌主力模型进阶", None),
    ("2026-05-20", "宇树 G1", "听语音，实时生成动作", None),
    ("2026-05-25", "宇树 WVLA2.0", "干扰下整理杂乱会议室", None),
    ("2026-06-23", "豆包 Seed2.1", "字节智能体升级", None),
    ("2026-06-30", "Claude Sonnet 5", "规划、浏览器、终端工具", None),
    ("2026-06-30", "Gemini Omni Flash", "对话生成视频", None),
    ("2026-07-08", "GPT-Live-1", "边听边说的实时语音", None),
    ("2026-07-30", "Gemini Robotics ER 2", "机器人推理换代", None),
    ("2026-07-31", "Seedance 2.5", "字节视频创作升级", None),
    ("2026-08-05", "SeedRealtime", "实时看、听、说", None),
    ("2026-08-13", "DeepSeek-V4-Pro", "国产第四代正式版", None),
    ("2026-09-01", "Claude Fable 5.1", "长程编程与知识工作", None),
    ("2026-09-10", "DeepSeek-V4.1", "国产原生多模态", None),
    ("2026-09-15", "Gemini 3.8 Live", "语音对话时后台思考", None),
    ("2026-09-22", "GPT-6 Sol", "六代模型扩展", None),
    ("2026-09-28", "Claude Sonnet 5.5", "更快、任务成本更低", None),
    ("2026-10-06", "Nano Banana 2.1", "图里的文字更准", None),
    ("2026-10-07", "Claude Haiku 5.5", "轻量助手换代", None),
    ("2026-10-17", "DeepSeek-V4.1-Flash", "国产快版再精修", None),
    ("2024-07-18", "GPT-4o mini", "比 GPT-3.5 Turbo 便宜六成多", None),
    ("2024-07-23", "Llama 3.1", "4050 亿参数，开放权重", None),
    ("2024-08-01", "FLUX.1", "开源文生图模型", None),
    ("2024-08-27", "GameNGen", "实时生成《毁灭战士》", "论文"),
    ("2024-09-05", "AlphaProteo", "设计能结合靶点的新蛋白", "论文"),
    ("2024-11-25", "MCP", "AI 接各种工具的统一接口", None),
    ("2024-12-04", "Genie 2", "一张图生成可玩的 3D 世界", "论文"),
    ("2024-12-09", "Sora", "视频生成向公众开放", None),
    ("2024-12-11", "Gemini 2.0", "为智能体时代打造", None),
    ("2024-12-16", "Veo 2", "最高 4K 的视频生成", None),
    ("2018-03-27", "World Models", "让 AI 在自己的「梦」里学习", "论文"),
    ("2019-11-19", "MuZero", "不给规则，自学环境模型", "论文"),
    ("2020-10-22", "ViT", "图片切块，交给 Transformer", "论文"),
    ("2021-01-05", "CLIP", "图文一起学，看懂图片", "论文"),
    ("2022-04-28", "Flamingo", "看几个例子，就会看图说话", "论文"),
    ("2022-12-19", "DiT", "扩散模型换上 Transformer", "论文"),
    ("2023-04-17", "LLaVA", "开源的看图聊天模型", "论文"),
    ("2024-02-23", "Genie", "从视频里学会可玩的世界", "论文"),
    ("2024-10-31", "π0", "一个模型控制多种机器人", "论文"),
    ("2025-01-06", "NVIDIA Cosmos", "给机器人用的世界基础模型", None),
    ("2025-04-02", "DreamerV3", "从零在《我的世界》挖到钻石", None),
    ("2025-06-11", "V-JEPA 2", "看视频学出世界模型", "论文"),
    ("2025-09-29", "Claude Sonnet 4.5", "复杂任务连续专注 30 多小时", None),
    ("2023-08-30", "Swift 无人机", "竞速赢了人类世界冠军", "超越人类"),
    ("2024-06-06", "快手可灵", "国产视频生成，50 万人申请内测", None),
    ("2025-06-11", "字节 Seedance 1.0", "视频生成榜单双第一", None),
    ("2025-09-30", "Sora 2", "App 不到 5 天下载破百万", None),
    ("2026-02-05", "GPT-5.3-Codex", "官方：早期版本帮忙调试了自己", None),
    ("2022-09-21", "Whisper", "开源语音识别模型", None),
    ("2022-09-29", "Make-A-Video", "文字直接生成视频", "论文"),
    ("2022-09-29", "DreamFusion", "文字直接生成 3D 模型", "论文"),
    ("2022-10-05", "AlphaTensor", "发现更快的矩阵乘法算法", "论文"),
    ("2022-11-05", "Midjourney V4", "AI 画图再升级", None),
    ("1974-08-15", "Werbos", "论文里提出反向传播雏形", "论文"),
    ("1979-07-15", "BKG 9.8", "7 比 1 赢了世界冠军", "超越人类"),
    ("1980-04-15", "新认知机", "卷积网络的前身", "论文"),
    ("1982-04-15", "Hopfield 网络", "42 年后拿了诺贝尔奖", "论文"),
    ("1982-06-01", "第五代计算机", "日本启动智能计算机计划", "新闻"),
    ("1985-03-01", "玻尔兹曼机", "后来也写进了诺奖理由", "论文"),
    ("1989-05-01", "Q 学习", "强化学习的经典算法", "论文"),
    ("1989-12-01", "LeNet 读邮编", "卷积网络读懂手写数字", "论文"),
]
FL_LIFE = 3.0
CD_FROM = 2017.45          # the Transformer
GPT35 = date_to_year("2022-11-30")


def countdown_days(T):
    import datetime
    return max(0, (datetime.date(2022, 11, 30) - date_label(T)).days)
TFs = (2017.45, "Transformer")
FLASH_SRC = {                                          # flash name -> the seeds that light up and send a spark into it
    "Hopfield 网络": [(1943.95, "神经元模型")], "LeNet 读邮编": [(1980.29, "新认知机"), (1986.78, "反向传播")],
    "DQN": [(1989.33, "Q 学习"), (2012.75, "AlexNet")], "玻尔兹曼机": [(1982.29, "Hopfield 网络")],
    "GPT-1": [TFs], "GPT-2": [TFs, (2018.43, "GPT-1")], "LLaMA": [TFs], "Gemini": [TFs], "Claude 3": [TFs],
    "DALL·E 2": [(2015.2, "扩散模型"), (2021.01, "DALL·E")], "Midjourney": [(2015.2, "扩散模型")], "Midjourney V4": [(2015.2, "扩散模型")], "Make-A-Video": [(2015.2, "扩散模型")],
    "Sora": [(2024.124, "Sora")], "快手可灵": [(2022.965, "DiT")], "字节 Seedance 1.0": [(2022.965, "DiT")],
    "Sora 2": [(2024.124, "Sora")], "Seedance 2.5": [(2026.118, "Seedance 2.0")], "Veo 2": [(2015.2, "扩散模型")], "ViT": [TFs], "CLIP": [TFs], "Flamingo": [TFs, (2021.01, "CLIP")],
    "DiT": [TFs, (2015.2, "扩散模型")], "LLaVA": [(2021.01, "CLIP"), TFs], "Genie": [(2018.24, "World Models")],
    "Genie 2": [(2024.15, "Genie")], "π0": [(2023.57, "RT-2")], "NVIDIA Cosmos": [(2018.24, "World Models")],
    "DreamerV3": [(2018.24, "World Models")], "V-JEPA 2": [(2018.24, "World Models")], "MuZero": [(2017.79, "AlphaGo Zero")], "GPT-4o 画图": [(2024.36, "GPT-4o")], "AlphaFold 3": [(2020.91, "AlphaFold 2")],
    "ESM3": [TFs], "DeepSeek-V3": [TFs], "o3": [(2024.70, "o1")], "Codex": [(2021.49, "GitHub Copilot")],
    "Claude Code": [(2021.49, "GitHub Copilot")], "AlphaEvolve": [(2023.95, "FunSearch")],
    "R1 登上《自然》": [(2025.05, "DeepSeek-R1")], "DeepSeekMath-V2": [(2025.05, "DeepSeek-R1")],
    "GPT-5": [(2024.36, "GPT-4o"), (2024.70, "o1")], "AlphaGeometry": [TFs], "RT-2": [TFs],
    "威尔·史密斯吃面": [(2015.2, "扩散模型")], "Claude 3.5 Sonnet": [(2024.17, "Claude 3")],
}
FL_LIFE_FAST = 2.2         # after ChatGPT: shorter, more rows


def fl_life(year):
    return FL_LIFE_FAST if year > 2022.95 else FL_LIFE


def count_after(t):
    """breakthroughs (beats and flashes, not news) after ChatGPT that have appeared by time t"""
    n = sum(1 for bt in BEATS if bt["year"] > 2022.95 and bt["at"] <= t)
    return n + sum(1 for f in flashes() if f["year"] > 2022.95 and f["tag"] != "新闻" and f["t"] <= t)


@lru_cache(None)
def flashes():
    ts = np.linspace(0.0, 330.0, 66001)
    ys = np.maximum.accumulate(np.minimum(_story(ts), STORY_END))
    out = []
    for d, name, txt, tag in FLASH:
        y = date_to_year(d)
        i = int(np.searchsorted(ys, y))
        if i >= len(ts):
            continue
        q = [int(x) for x in d.split("-")]
        out.append(dict(t=snap(float(ts[i])), year=y, name=name, text=txt, tag=tag,
                        dl=f"{q[1]:02d}.{q[2]:02d}" if q[0] >= 2026 else f"{q[0]}.{q[1]:02d}"))
    return out


def flash_layout(t, hxp, hyp, y0=None):
    """Visible flashes, newest on top: (flash, baseline y, alpha, row height); older rows slide down as new ones arrive."""
    act = sorted([f for f in flashes() if 0 <= t - f["t"] < fl_life(f["year"])], key=lambda f: -f["t"])
    act = act[:6] if act and act[0]["year"] > 2022.95 else act[:5]
    out, y = [], (y0 if y0 is not None and act and act[0]["year"] > 2022.95 else hyp + (180 if act and act[0]["year"] > 2022.95 else 128))
    for f in act:
        age = t - f["t"]
        life = fl_life(f["year"])
        a = float(smooth(age / 0.15)) * (1 - float(smooth((age - life + 0.5) / 0.5)))
        h = 76 if (f["text"] or f["tag"]) else 46
        k = float(smooth(age / 0.25))
        out.append((f, y - (1 - k) * 0.7 * h, a * k * k, h))
        y += h * k
    return out


# a dozen key nodes keep a small label on the line after their moment (not more: it gets messy)
KEEP = [(1950.8, "图灵测试"), (1956.6, "达特茅斯会议"), (1958.55, "感知机"), (1975.0, "第一次寒冬"), (1986.78, "反向传播"),
        (1987.2, "第二次寒冬"), (1997.37, "深蓝"), (2012.75, "AlexNet"), (2016.2, "AlphaGo"), (2017.45, "Transformer"),
        (2020.41, "GPT-3"), (2022.915, "ChatGPT"), (2023.2, "GPT-4"), (2024.36, "GPT-4o"), (2024.70, "o1"),
        (2025.05, "DeepSeek-R1"), (2025.38, "AI 视频"), (2025.55, "奥数金牌"), (2025.88, "编程智能体"), (2026.173, "GPT-5.4"),
        (2026.385, "Erdős"), (2026.69, "千禧年难题")]
# the final overview: every important date
LANDMARKS = [
    (1943.95, "1943", "神经元模型"), (1950.8, "1950", "图灵测试"), (1956.6, "1956", "达特茅斯会议"),
    (1958.55, "1958", "感知机"), (1975.5, "1974", "第一次寒冬"), (1986.78, "1986", "反向传播"),
    (1989.0, "1987", "第二次寒冬"), (1997.37, "1997", "深蓝"), (2012.75, "2012", "AlexNet"),
    (2016.2, "2016", "AlphaGo"), (2017.45, "2017", "Transformer"), (2020.41, "2020", "GPT-3"),
    (2022.915, "2022.11", "ChatGPT"), (2023.2, "2023.03", "GPT-4"), (2024.70, "2024.09", "o1"), (2025.05, "2025.01", "DeepSeek-R1"),
    (2025.55, "2025.07", "奥数金牌"), (2026.385, "05.20", "Erdős"), (2026.69, "09.08", "千禧年难题"),
    (STORY_END, "10.18", "今天"),
]


# ======================================================================================
# cairo <-> numpy helpers and cached sprites
# ======================================================================================
_keep = []


def surf_from(rgb, alpha, dither=True):
    """rgb: (3,) or (h,w,3) floats 0-1, alpha (h,w) 0-1 -> premultiplied ARGB32 surface."""
    h, w = alpha.shape
    if np.ndim(rgb) == 1:
        rgb = np.broadcast_to(rgb, (h, w, 3))
    a = alpha[..., None]
    pm = np.concatenate([rgb[..., ::-1] * a, a], axis=2) * 255
    if dither:
        pm = pm + np.random.default_rng(7).uniform(-0.5, 0.5, pm.shape)
    arr = np.ascontiguousarray(np.clip(np.round(pm), 0, 255).astype(np.uint8))
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    if stride != w * 4:
        arr = np.ascontiguousarray(np.pad(arr, ((0, 0), (0, stride // 4 - w), (0, 0))))
    _keep.append(arr)
    return cairo.ImageSurface.create_for_data(memoryview(arr), cairo.FORMAT_ARGB32, w, h, stride)


def surf_alpha(s):
    w, h, st = s.get_width(), s.get_height(), s.get_stride()
    return np.ndarray((h, st // 4, 4), np.uint8, s.get_data())[:, :w, 3].astype(float) / 255


@lru_cache(None)
def background(pal):
    p = PAL[pal]
    rng = np.random.default_rng(2026)
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    v = ys / H
    img = hx(p["top"])[None, None] * (1 - v[..., None]) + hx(p["bot"])[None, None] * v[..., None]
    mott = gaussian_filter(rng.standard_normal((H // 8, W // 8)), 10)
    mott = np.kron(mott / (np.abs(mott).max() + 1e-9), np.ones((8, 8)))[:H, :W]
    img += gaussian_filter(mott, 8)[..., None] * np.array([0.014, 0.011, 0.008])
    dust = np.zeros((H, W))
    n = 450
    np.add.at(dust, (rng.integers(0, H, n), rng.integers(0, W, n)), rng.pareto(2.5, n) * 0.05 + 0.02)
    img += (gaussian_filter(dust, 0.8) * 2.2)[..., None] * hx(p["core"])
    r2 = ((xs - W / 2) / (0.72 * W)) ** 2 + ((ys - 0.48 * H) / (0.66 * H)) ** 2
    img *= (1 - 0.5 * smooth(r2 / 1.4))[..., None]
    return surf_from(np.clip(img, 0, 1), np.ones((H, W)))


@lru_cache(None)
def sprite_glow(size=512, falloff=(0.9, 14, 0.45, 60, 0.18, 170)):
    c = size / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rr = np.hypot(xx - c, yy - c) * (512 / size)
    a1, r1, a2, r2, a3, r3 = falloff
    a = a1 * np.exp(-(rr / r1) ** 2) + a2 * np.exp(-(rr / r2) ** 2) + a3 * np.exp(-(rr / r3) ** 2)
    return np.clip(a, 0, 1)


@lru_cache(None)
def glow_surface(hexcol):
    return surf_from(hx(hexcol), sprite_glow())


@lru_cache(None)
def streak_surface(hexcol):
    w, h = 900, 24
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    a = 0.24 * np.exp(-((xx - w / 2) / 250) ** 2 - ((yy - h / 2) / 3.2) ** 2)
    return surf_from(hx(hexcol), a)


ERAS = [  # (from, to, image, opacity, blur): what AI ran on
    (1943, 1979.5, "eniac.jpg", 0.085, 3.5),
    (1979.5, 1993.5, "symbolics_lisp_chm.jpg", 0.12, 4.0),
    (1993.5, 2011.8, "sun_e3000_cluster.jpg", 0.11, 4.0),
    (2011.8, 2019.5, "gpu_gtx1060_rear.jpg", 0.10, 9.0),
    (2019.5, 2030, "wmf_servers_2012.jpg", 0.10, 5.0),
]


@lru_cache(None)
def era_alpha(name, opacity, blur):
    im = Image.open(os.path.join(IMG, name)).convert("L")
    w0, h0 = im.size
    k = max(W / w0, H / h0)
    im = im.resize((int(w0 * k) + 1, int(h0 * k) + 1), Image.LANCZOS)
    x0, y0 = (im.size[0] - W) // 2, (im.size[1] - H) // 2
    L = gaussian_filter(np.asarray(im.crop((x0, y0, x0 + W, y0 + H)), float) / 255, blur)
    L = np.clip((L - np.percentile(L, 5)) / (np.percentile(L, 99) - np.percentile(L, 5) + 1e-9), 0, 1)
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    r2 = ((xs - W / 2) / (0.62 * W)) ** 2 + ((ys - H / 2) / (0.55 * H)) ** 2
    return opacity * L ** 1.3 * (1 - smooth(r2 / 1.2))


@lru_cache(None)
def era_surface(name, opacity, blur, palname):
    return surf_from(hx(PAL[palname]["plate"]), era_alpha(name, opacity, blur))


@lru_cache(None)
def photo_surface(name, max_w=440, max_h=520, mono=False, cold=False):
    """A picture card: sepia for the archive years, a muted warm grade later (so press colours don't shout), crisp
    edge, soft shadow. 'a|b' puts two pictures side by side."""
    if "|" in name:
        ims = [Image.open(os.path.join(IMG, n)).convert("RGB") for n in name.split("|")]
        hh = min(i.height for i in ims)
        ims = [i.resize((int(i.width * hh / i.height), hh), Image.LANCZOS) for i in ims]
        im = Image.new("RGB", (sum(i.width for i in ims) + 12 * (len(ims) - 1), hh), (20, 18, 15))
        x = 0
        for i in ims:
            im.paste(i, (x, 0)); x += i.width + 12
    else:
        im = Image.open(os.path.join(IMG, name)).convert("RGB")
    im.thumbnail((max_w, max_h), Image.LANCZOS)
    A = np.asarray(im, float) / 255
    g = A.mean(axis=2, keepdims=True)
    A = g * np.array([1.0, 0.95, 0.86]) if mono else mix(A, g, 0.45)
    A = 0.08 + 0.86 * A
    A = A * np.array([0.8, 0.88, 1.0]) * 0.8 if cold else A * np.array([1.0, 0.95, 0.86])
    h, w = A.shape[:2]
    pad = 40
    rgb = np.zeros((h + 2 * pad, w + 2 * pad, 3))
    sh = np.zeros((h + 2 * pad, w + 2 * pad))
    sh[pad + 14:pad + 14 + h, pad + 8:pad + 8 + w] = 1
    alpha = 0.55 * gaussian_filter(sh, 14)
    page = np.zeros_like(sh)
    page[pad:pad + h, pad:pad + w] = 1
    page = gaussian_filter(page, 0.7)
    rgb[pad:pad + h, pad:pad + w] = A
    alpha = page * 0.97 + (1 - page) * alpha
    return surf_from(rgb * (page > 0.01)[..., None], np.clip(alpha, 0, 1))


# ======================================================================================
# text
# ======================================================================================
def text(ctx, s, x, y, size, rgb, alpha=1.0, font=SERIF, bold=False, align="left"):
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ext = ctx.text_extents(s)
    if align == "right":
        x -= ext.x_advance
    elif align == "center":
        x -= ext.x_advance / 2
    ctx.move_to(x, y)
    ctx.set_source_rgba(*rgb, alpha)
    ctx.show_text(s)
    return ext


@lru_cache(maxsize=256)
def glow_text(s, size, hexcol, bold=False):
    """Text with a soft halo, rendered once per string: returns (surface, width, height, baseline offset)."""
    pad = int(size * 0.5)
    tmp = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    c = cairo.Context(tmp)
    c.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    c.set_font_size(size)
    e = c.text_extents(s)
    w, h = int(e.x_advance + 2 * pad), int(size * 1.4 + 2 * pad)
    m = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(m)
    c.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    c.set_font_size(size)
    c.move_to(pad, pad + size)
    c.set_source_rgba(1, 1, 1, 1)
    c.show_text(s)
    A = surf_alpha(m)
    a = np.clip(0.95 * A + 0.5 * gaussian_filter(A, size * 0.07), 0, 1)
    return surf_from(hx(hexcol), a), w, h, pad + size


# bar charts in the window: label, value, shown text (verified numbers; sources in the scenes doc)
BARS = {
    "gpt4": dict(unit="模拟律师考试：超过多少考生", vmax=100, items=[("GPT-3.5", 10, "约 10%"), ("GPT-4", 90, "约 90%")]),
    "o1": dict(unit="AIME 2024 数学竞赛正确率", vmax=100, items=[("GPT-4o", 12, "12%"), ("o1", 74, "74%")]),
    "r1": dict(unit="每百万输出 token 的价格", vmax=60, items=[("o1", 60, "$60"), ("R1", 2.19, "$2.19")]),
    "turing": dict(unit="三方对话测试：被当成真人的比例", vmax=100, ref=(50, "随机猜 50%"),
                   items=[("ELIZA", 23, "23%"), ("GPT-4.5", 73, "73%")]),
    "imo": dict(unit="国际数学奥赛得分（满分 42）", vmax=42, ref=(35, "2025 金牌线"),
                items=[("2024 银牌", 28, "28"), ("2025 金牌", 35, "35")]),
    "icpc": dict(unit="ICPC 2025 世界总决赛（共 12 题）", vmax=12,
                 items=[("Gemini", 10, "10"), ("人类最佳", 11, "11"), ("OpenAI", 12, "12")]),
    "gpt54": dict(unit="OSWorld 电脑操作测试", vmax=100, ref=(72.4, "人类 72.4%"),
                  items=[("GPT-5.2", 47.3, "47.3%"), ("GPT-5.4", 75.0, "75.0%")]),
}


# faint classic formulas behind everything, each in its era (start year, formula); STIX Two Math has the symbols
FORMULAS = [                  # _x / _{..} subscript, ^x / ^{..} superscript (drawn by draw_math)
    (1943.95, "y = H(Σ w_i x_i − θ)"),
    (1957.5, "V(s) = max_a [ r + γ V(s′) ]"),
    (1958.55, "w ← w + η (y − ŷ) x"),
    (1986.78, "∂L/∂w = ∂L/∂y · ∂y/∂w"),
    (1992.3, "δ = r + γ V(s′) − V(s)"),
    (1997.9, "c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t"),
    (1998.9, "(f ∗ g)[n] = Σ_k f[k] g[n − k]"),
    (2012.75, "θ ← θ − η ∇L(θ)"),
    (2014.45, "E[log D(x)] + E[log(1 − D(G(z)))]"),
    (2015.9, "y = F(x) + x"),
    (2017.45, "softmax(QK^T / √d_k) V"),
    (2020.41, "L(N) = (N_c / N)^{0.076}"),
    (2020.5, "x_t = √α̅_t x_0 + √(1 − α̅_t) ε"),
    (2022.93, "E[r(x, y)] − β KL(π ‖ π_0)"),
    (2025.06, "Â_i = (r_i − mean r) / std r"),
    (2026.39, "∂u/∂t + (u·∇)u = −∇p + νΔu"),
]
FORMULA_SLOTS = LAYOUT["formula_slots"]          # portrait: the empty band under the axis; landscape: above the line
FORMULA_LIFE = 11.0


@lru_cache(None)
def formula_times():
    ts = np.linspace(0.0, 330.0, 66001)
    ys = np.maximum.accumulate(np.minimum(_story(ts), STORY_END))
    return [float(ts[min(int(np.searchsorted(ys, y)), len(ts) - 1)]) for y, _ in FORMULAS]


def draw_math(ctx, src, x, y, size, draw=True):
    """Minimal maths typesetting: _x or _{..} lowered and smaller, ^x or ^{..} raised and smaller. Returns the end x."""
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in "_^" and i + 1 < n:
            if src[i + 1] == "{":
                j = src.index("}", i + 2); tok = src[i + 2:j]; i = j + 1
            else:
                tok = src[i + 1]; i += 2
            ctx.set_font_size(size * 0.62)
            if draw:
                ctx.move_to(x, y + (0.22 * size if ch == "_" else -0.42 * size)); ctx.show_text(tok)
            x += ctx.text_extents(tok).x_advance
            ctx.set_font_size(size)
            continue
        if i + 1 < n and unicodedata.category(src[i + 1]) == "Mn":   # a combining mark rides on its letter (ᾱ)
            ch = src[i:i + 2]; i += 1
        ctx.set_font_size(size)
        if draw:
            ctx.move_to(x, y); ctx.show_text(ch)
        x += ctx.text_extents(ch).x_advance if ch != " " else size * 0.28
        i += 1
    return x


def draw_formulas(ctx, t, ov):
    """Very faint, slowly drifting: they add weight without competing with the line."""
    for i, ((y_, fml), t0) in enumerate(zip(FORMULAS, formula_times())):
        age = t - t0
        if not (0 <= age < FORMULA_LIFE) or t >= OVERVIEW[0]:
            continue
        a = float(smooth(age / 1.2)) * (1 - float(smooth((age - FORMULA_LIFE + 1.5) / 1.5))) * (1 - ov)
        x, y, size = FORMULA_SLOTS[i % len(FORMULA_SLOTS)]
        ctx.select_font_face("STIX Two Math", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); ctx.set_font_size(size)
        ctx.set_source_rgba(0.96, 0.88, 0.7, 0.11 * a)
        wd = draw_math(ctx, fml, 0, y, size, draw=False)
        x = min(x, W - 40 - wd - 5 * FORMULA_LIFE)     # long ones move left so they never leave the frame
        draw_math(ctx, fml, x + 5 * age, y, size)


# ======================================================================================
# algorithm drawings for the info window
# ======================================================================================
VX, VY, VS = 48, 120, 470          # drawing canvas size (VS); drawn scaled into the window


def viz_draw(kind, ctx, p, tt, col):
    """Draw algorithm `kind` into a VSxVS context. p = 0..1 life progress, tt = seconds since it opened."""
    r, g, b = col
    def dot(x, y, rad, a=1.0, c=col):
        ctx.set_source_rgba(*c, a); ctx.arc(x, y, rad, 0, 2 * math.pi); ctx.fill()
    def line(x0, y0, x1, y1, wdt, a):
        ctx.set_source_rgba(r, g, b, a); ctx.set_line_width(wdt); ctx.move_to(x0, y0); ctx.line_to(x1, y1); ctx.stroke()
    S = VS
    if kind in ("neuron", "perceptron"):
        ins = [(S * 0.16, S * (0.22 + 0.19 * i)) for i in range(4)]
        n = (S * 0.58, S * 0.5)
        ws = [0.9, 0.35, 0.7, 0.5] if kind == "perceptron" else [0.6] * 4
        if kind == "perceptron":                       # weights learn: thickness drifts to final values
            k = smooth((tt - 1.0) / 3.5)
            ws = [mix(0.4, w, k) for w in ws]
        for (x, y), w in zip(ins, ws):
            line(x, y, *n, 1.2 + 5 * w, 0.35)
            for j in range(2):                         # pulses travel along the inputs
                f = ((tt * 0.8 + j * 0.5 + y / S) % 1.0)
                dot(mix(x, n[0], f), mix(y, n[1], f), 3.5, 0.9 * w + 0.1)
            dot(x, y, 9, 0.9)
        fire = 0.5 + 0.5 * math.sin(tt * 3.2)
        dot(*n, 26, 0.25 + 0.5 * fire)
        dot(*n, 14, 1.0)
        line(n[0] + 26, n[1], S * 0.9, n[1], 2.5, 0.5 + 0.4 * fire)   # output
        # step function glyph
        ctx.set_source_rgba(r, g, b, 0.6); ctx.set_line_width(2)
        bx, by = S * 0.72, S * 0.78
        ctx.move_to(bx, by); ctx.line_to(bx + 40, by); ctx.line_to(bx + 40, by - 40); ctx.line_to(bx + 80, by - 40); ctx.stroke()
    elif kind == "turing":                             # the imitation game: two hidden answerers behind a wall
        line(S * 0.5, S * 0.12, S * 0.5, S * 0.88, 2.0, 0.6)
        dot(S * 0.25, S * 0.32, 22, 0.85)              # a person
        ctx.set_source_rgba(r, g, b, 0.85); ctx.set_line_width(3)
        ctx.arc(S * 0.25, S * 0.62, 42, math.pi * 1.05, math.pi * 1.95); ctx.stroke()
        ctx.rectangle(S * 0.66, S * 0.26, 70, 70); ctx.stroke()      # a machine
        ctx.rectangle(S * 0.69, S * 0.29, 52, 34); ctx.stroke()
        for j in range(3):                             # questions and answers cross the wall
            f = (tt * 0.45 + j / 3) % 1.0
            x = mix(S * 0.3, S * 0.72, f if j % 2 == 0 else 1 - f)
            dot(x, S * (0.46 + 0.06 * j), 5, 0.9)
        dot(S * 0.5, S * 0.18 + 6 * math.sin(tt * 2), 7, 1.0)
    elif kind == "tree":                               # search over symbols: a tree that branches
        root = (S * 0.5, S * 0.14)
        k = smooth(tt / 4.0)
        lev = [[root]]
        for d in range(1, 4):
            prev = lev[-1]
            cur = []
            for i, (x, y) in enumerate(prev):
                for j in (-1, 1):
                    cur.append((x + j * S * 0.42 / (2 ** d), y + S * 0.24))
            lev.append(cur)
        for d in range(1, 4):
            vis = smooth(k * 3 - (d - 1))
            for i, (x, y) in enumerate(lev[d]):
                px, py = lev[d - 1][i // 2]
                line(px, py, mix(px, x, vis), mix(py, y, vis), 2, 0.55)
                if vis > 0.95:
                    dot(x, y, 7 if d < 3 else 5, 0.9)
        dot(*root, 10, 1.0)
        best = int(tt * 1.5) % 8                       # the chosen path lights up
        x, y, idx = root[0], root[1], 0
        for d in range(1, 4):
            idx = idx * 2 + ((best >> (3 - d)) & 1)
            nx, ny = lev[d][idx]
            line(x, y, nx, ny, 4, 0.9 * k)
            x, y = nx, ny
    elif kind == "chat":                               # pattern in, canned reply out: alternating bubbles
        for i in range(4):
            vis = smooth(tt * 1.1 - i * 0.9)
            left = i % 2 == 0
            w = S * (0.55 if left else 0.45) * vis
            x = S * 0.1 if left else S * 0.9 - w
            y = S * (0.15 + 0.2 * i)
            ctx.set_source_rgba(r, g, b, 0.18 if left else 0.4)
            ctx.rectangle(x, y, w, S * 0.11); ctx.fill()
            for k in range(3):
                ctx.set_source_rgba(r, g, b, 0.75)
                ctx.rectangle(x + 12, y + 12 + k * 14, max(0.0, w - 24) * (0.9 - 0.25 * k), 5); ctx.fill()
    elif kind == "rules":                              # expert system: IF boxes feed THEN boxes, rules fire in turn
        cols = [S * 0.18, S * 0.5, S * 0.82]
        boxes = [[(cols[0], S * (0.2 + 0.2 * i)) for i in range(4)], [(cols[1], S * (0.3 + 0.2 * i)) for i in range(3)],
                 [(cols[2], S * 0.5)]]
        fire = int(tt * 2) % 4
        for a_, b_ in ((0, 1), (1, 2)):
            for i, (x0, y0) in enumerate(boxes[a_]):
                for j, (x1, y1) in enumerate(boxes[b_]):
                    if (i + j) % 2 == 0 or b_ == 2:
                        line(x0 + 28, y0, x1 - 28, y1, 1.5, 0.35 if i != fire else 0.95)
        for c_, bs in enumerate(boxes):
            for i, (x, y) in enumerate(bs):
                ctx.set_source_rgba(r, g, b, 0.95 if (c_ == 0 and i == fire) or c_ == 2 else 0.5)
                ctx.set_line_width(2); ctx.rectangle(x - 28, y - 18, 56, 36); ctx.stroke()
    elif kind == "selfplay":                           # two copies play each other; a value estimate settles
        for i, (x, y) in enumerate([(S * 0.25, S * 0.4), (S * 0.75, S * 0.4)]):
            dot(x, y, 24, 0.85 if (int(tt * 2) % 2) == i else 0.35)
        for j in range(4):
            f = (tt * 0.9 + j / 4) % 1.0
            dot(mix(S * 0.3, S * 0.7, f if j % 2 == 0 else 1 - f), S * 0.4 + 20 * math.sin(f * math.pi), 4, 0.9)
        v = 0.5 + 0.35 * (1 - math.exp(-tt / 1.8)) * math.sin(tt * 1.3) ** 2
        ctx.set_source_rgba(r, g, b, 0.3); ctx.rectangle(S * 0.15, S * 0.72, S * 0.7, 14); ctx.fill()
        ctx.set_source_rgba(r, g, b, 0.95); ctx.rectangle(S * 0.15, S * 0.72, S * 0.7 * v, 14); ctx.fill()
    elif kind == "lstm":                               # a memory cell: a value circulates and is kept
        cx_, cy_ = S * 0.5, S * 0.48
        ctx.set_source_rgba(r, g, b, 0.5); ctx.set_line_width(3); ctx.arc(cx_, cy_, S * 0.22, 0, 6.283); ctx.stroke()
        for j in range(5):
            a_ = tt * 1.6 + j * 1.2566
            dot(cx_ + S * 0.22 * math.cos(a_), cy_ + S * 0.22 * math.sin(a_), 5 if j else 9, 0.9)
        for k_, x in enumerate((S * 0.12, S * 0.88)):  # input and output gates
            gate = 0.5 + 0.5 * math.sin(tt * 2.2 + k_ * 2)
            line(x, cy_, cx_ + (-1 if k_ == 0 else 1) * S * 0.22, cy_, 2 + 4 * gate, 0.3 + 0.6 * gate)
        dot(cx_, cy_, 16, 0.9)
    elif kind == "conv":                               # a small kernel slides over an image and builds a feature map
        n = 8
        cell = S * 0.06
        ox, oy = S * 0.08, S * 0.22
        pos = int(tt * 4) % ((n - 2) * (n - 2))
        kx, ky = pos % (n - 2), pos // (n - 2)
        for i in range(n):
            for j in range(n):
                v = 0.25 + 0.6 * ((i * 3 + j * 5) % 7 == 0 or (i - 4) ** 2 + (j - 3) ** 2 < 5)
                ctx.set_source_rgba(r, g, b, 0.15 + 0.55 * v); ctx.rectangle(ox + i * cell, oy + j * cell, cell - 3, cell - 3); ctx.fill()
        ctx.set_source_rgba(r, g, b, 1.0); ctx.set_line_width(3)
        ctx.rectangle(ox + kx * cell - 2, oy + ky * cell - 2, 3 * cell + 1, 3 * cell + 1); ctx.stroke()
        fx, fy = S * 0.62, S * 0.3
        fc = S * 0.05
        for q_ in range(pos + 1):
            i, j = q_ % (n - 2), q_ // (n - 2)
            ctx.set_source_rgba(r, g, b, 0.85); ctx.rectangle(fx + i * fc, fy + j * fc, fc - 3, fc - 3); ctx.fill()
        line(ox + (kx + 3) * cell, oy + (ky + 1.5) * cell, fx + kx * fc, fy + ky * fc, 1.5, 0.6)
    elif kind == "layers":                             # deep nets: layers stack up one after another
        nl = 1 + min(5, int(tt * 1.2))
        for l in range(nl):
            x = S * (0.14 + 0.15 * l)
            for i in range(5 - (l % 2)):
                y = S * (0.24 + 0.13 * i + 0.065 * (l % 2))
                dot(x, y, 8, 0.9)
                if l < nl - 1:
                    for j in range(5 - ((l + 1) % 2)):
                        line(x, y, x + S * 0.15, S * (0.24 + 0.13 * j + 0.065 * ((l + 1) % 2)), 1, 0.2)
    elif kind == "mosaic":                             # a big labelled image collection: tiles fill, boxes find objects
        n = 7
        cell = S * 0.115
        filled = int(tt * 14)
        rng_ = np.random.default_rng(9)
        tones = rng_.uniform(0.25, 0.9, n * n)
        for q_ in range(min(filled, n * n)):
            i, j = q_ % n, q_ // n
            ctx.set_source_rgba(r, g, b, 0.18 + 0.5 * tones[q_]); ctx.rectangle(S * 0.06 + i * cell, S * 0.08 + j * cell, cell - 5, cell - 5); ctx.fill()
        if filled > n * n:
            for q_ in (3, 17, 30, 44):
                i, j = q_ % n, q_ // n
                ctx.set_source_rgba(r, g, b, 1.0); ctx.set_line_width(3)
                ctx.rectangle(S * 0.06 + i * cell - 3, S * 0.08 + j * cell - 3, cell + 1, cell + 1); ctx.stroke()
    elif kind == "cnn":                                # stacked feature maps shrinking towards a decision
        for l in range(5):
            sz = S * (0.42 - 0.07 * l)
            x = S * (0.08 + 0.17 * l)
            y = S * 0.5 - sz / 2
            for d in range(3):
                ctx.set_source_rgba(r, g, b, 0.18 + 0.12 * d); ctx.rectangle(x + d * 8, y + d * 8, sz * 0.5, sz); ctx.fill()
            f = (tt * 1.2 - l * 0.35) % 2.0
            if f < 1:
                dot(x + sz * 0.25 + 8, y + sz * f, 5, 0.95)
        dot(S * 0.92, S * 0.5, 14, 0.5 + 0.5 * math.sin(tt * 3))
    elif kind == "mcts":                               # tree search guided by networks: visits pile up on one branch
        root = (S * 0.5, S * 0.12)
        kids = [(S * (0.14 + 0.144 * i), S * 0.42) for i in range(6)]
        visits = [3, 2, 4, 9, 3, 2]
        grow = 1 + tt * 2.2
        for i, (x, y) in enumerate(kids):
            v = visits[i] * (1 + (grow if i == 3 else grow * 0.15))
            line(*root, x, y, 1 + min(9, v * 0.25), 0.35 + (0.55 if i == 3 else 0))
            dot(x, y, 7, 0.9)
        best = kids[3]
        for j in range(4):
            gx, gy = best[0] + (j - 1.5) * S * 0.09, S * 0.7
            line(*best, gx, gy, 2 + (3 if j == 1 else 0), 0.5)
            dot(gx, gy, 5, 0.85)
        dot(*root, 11, 1.0)
        f = (tt * 0.9) % 1.0                          # a playout runs down the favoured branch
        dot(mix(root[0], best[0], f), mix(root[1], best[1], f), 5, 1.0)
    elif kind == "attention":                          # every word looks at every other word, some links strong
        n = 7
        xs_ = [S * (0.1 + 0.8 * i / (n - 1)) for i in range(n)]
        q = int(tt * 1.5) % n
        for j in range(n):
            if j == q:
                continue
            w = 0.15 + 0.85 * (1 if j == (q + 3) % n else 0.25 * ((q * 7 + j * 3) % 4) / 3)
            ctx.set_source_rgba(r, g, b, 0.15 + 0.75 * w); ctx.set_line_width(1 + 5 * w)
            x0, x1 = xs_[q], xs_[j]
            ctx.move_to(x0, S * 0.62)
            ctx.curve_to(x0, S * 0.62 - abs(x1 - x0) * 0.6, x1, S * 0.62 - abs(x1 - x0) * 0.6, x1, S * 0.62)
            ctx.stroke()
        for i, x in enumerate(xs_):
            dot(x, S * 0.62, 11 if i == q else 8, 1.0 if i == q else 0.7)
            ctx.set_source_rgba(r, g, b, 0.35); ctx.rectangle(x - 16, S * 0.7, 32, 8); ctx.fill()
    elif kind == "nexttoken":                          # guess the next word: a row grows, candidates compete
        n = 3 + int(tt * 1.6)
        for i in range(min(n, 9)):
            ctx.set_source_rgba(r, g, b, 0.85); ctx.rectangle(S * 0.06 + i * S * 0.1, S * 0.3, S * 0.08, S * 0.08); ctx.fill()
        probs = [0.62, 0.21, 0.1, 0.07]
        x0 = S * 0.06 + min(n, 9) * S * 0.1
        for j, pr in enumerate(probs):
            ctx.set_source_rgba(r, g, b, 0.25 + 0.7 * pr); ctx.rectangle(S * 0.1, S * (0.5 + 0.09 * j), S * 0.8 * pr * smooth(tt * 2 % 1.0 + 0.3), S * 0.05); ctx.fill()
        ctx.set_source_rgba(r, g, b, 0.95); ctx.set_line_width(2); ctx.rectangle(min(x0, S * 0.86), S * 0.3, S * 0.08, S * 0.08); ctx.stroke()
    elif kind == "protein":                            # a chain of amino acids folds into a compact shape
        n = 26
        k = smooth(tt / 4.0)
        pts = []
        for i in range(n):
            a0 = (S * (0.08 + 0.84 * i / (n - 1)), S * 0.5 + 18 * math.sin(i * 1.3))
            ang = i * 0.55
            rad = S * (0.08 + 0.25 * (i / n))
            a1 = (S * 0.5 + rad * math.cos(ang), S * 0.5 + rad * math.sin(ang) * 0.8)
            pts.append((mix(a0[0], a1[0], k), mix(a0[1], a1[1], k)))
        ctx.set_source_rgba(r, g, b, 0.6); ctx.set_line_width(3)
        ctx.move_to(*pts[0])
        for pt in pts[1:]:
            ctx.line_to(*pt)
        ctx.stroke()
        for i, pt in enumerate(pts):
            dot(*pt, 6 if i % 3 else 9, 0.9)
    elif kind == "diffusion":                          # a real denoising run: pure noise -> the 1950 Turing portrait
        img, noise = diffusion_arrays()
        steps = 50
        k = min(steps, int(steps * smooth(tt / 3.6)))
        ab = (k / steps) ** 1.7
        x = np.clip(math.sqrt(ab) * img + math.sqrt(1 - ab) * noise * 0.9 + 0.5 * (1 - math.sqrt(ab)) * 0.3, 0, 1)
        sz = int(S * 0.78)
        A8 = np.clip(x, 0, 1)
        surf = surf_from(np.dstack([A8 * r, A8 * g, A8 * b]), np.ones_like(A8), dither=False)
        ctx.save(); ctx.translate(S * 0.11, S * 0.04); ctx.scale(sz / A8.shape[1], sz / A8.shape[0])
        ctx.set_source_surface(surf, 0, 0); ctx.get_source().set_filter(cairo.FILTER_NEAREST if k < steps * 0.4 else cairo.FILTER_BILINEAR)
        ctx.paint(); ctx.restore()
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(26)
        ctx.set_source_rgba(r, g, b, 0.9); ctx.move_to(S * 0.11, S * 0.9); ctx.show_text(f"去噪 第 {k} / {steps} 步")
    elif kind == "diffusion_old":                      # (v2 placeholder, unused)
        n = 14
        cell = S * 0.84 / n
        k = smooth(tt / 4.0)
        rng_ = np.random.default_rng(max(0, int(tt * 6)))
        for i in range(n):
            for j in range(n):
                target = 1.0 if (i - 7) ** 2 + (j - 6) ** 2 < 16 or (abs(i - 7) < 2 and j > 9) else 0.1
                v = mix(rng_.uniform(0, 1), target, k)
                ctx.set_source_rgba(r, g, b, 0.08 + 0.8 * v); ctx.rectangle(S * 0.08 + i * cell, S * 0.08 + j * cell, cell - 2, cell - 2); ctx.fill()
    elif kind == "reasoning":                          # think before answering: steps, a dead end, back, then on
        pts = [(0.12, 0.8), (0.28, 0.66), (0.44, 0.56), (0.58, 0.38), (0.5, 0.2), (0.62, 0.48), (0.78, 0.32), (0.9, 0.16)]
        dead = {4}
        n = min(len(pts), 1 + int(tt * 1.6))
        for i in range(1, n):
            a0, a1 = pts[i - 1] if i != 5 else pts[3], pts[i]
            line(a0[0] * S, a0[1] * S, a1[0] * S, a1[1] * S, 2.5, 0.25 if i in dead else 0.85)
        for i in range(n):
            x, y = pts[i]
            dot(x * S, y * S, 8, 0.3 if i in dead else 1.0)
            if i in dead:
                line(x * S - 8, y * S - 8, x * S + 8, y * S + 8, 2, 0.6)
    elif kind == "evolve":                             # many candidate programs; the best ones survive and mutate
        gens = 1 + int(tt * 1.4)
        for gidx in range(min(gens, 6)):
            rng_ = np.random.default_rng(gidx)
            base = 0.2 + 0.13 * gidx
            for j in range(9):
                v = min(0.95, base + rng_.uniform(-0.12, 0.12))
                x = S * (0.08 + 0.1 * j)
                ctx.set_source_rgba(r, g, b, 0.15 + 0.12 * gidx); ctx.rectangle(x, S * 0.85 - S * 0.7 * v, S * 0.07, S * 0.7 * v); ctx.fill()
        line(S * 0.06, S * 0.85, S * 0.96, S * 0.85, 1.5, 0.6)
    elif kind == "unitdist":                           # points at unit distance: the old grid, then a new, denser pattern
        k = smooth((tt - 1.5) / 2.5)
        n = 5
        pts = []
        for i in range(n):
            for j in range(n):
                gx, gy = S * (0.2 + 0.15 * i), S * (0.2 + 0.15 * j)
                ang = (i * 2.4 + j * 1.1)
                nx, ny = S * 0.5 + S * 0.32 * math.cos(ang) * (0.35 + 0.65 * ((i + j) % 3) / 2), S * 0.5 + S * 0.32 * math.sin(ang) * (0.35 + 0.65 * ((i * j) % 3) / 2)
                pts.append((mix(gx, nx, k), mix(gy, ny, k)))
        for i, (x0, y0) in enumerate(pts):
            for j in range(i + 1, len(pts)):
                x1, y1 = pts[j]
                d = math.hypot(x1 - x0, y1 - y0)
                if abs(d - S * 0.15) < S * (0.012 + 0.03 * k):
                    line(x0, y0, x1, y1, 1.6, 0.7)
        for (x, y) in pts:
            dot(x, y, 5, 0.95)
    elif kind == "ns":                                 # the Navier-Stokes equations, with the flow they describe
        for j in range(9):                             # streamlines
            ctx.set_source_rgba(r, g, b, 0.35); ctx.set_line_width(1.6)
            y0 = S * (0.52 + 0.05 * j)
            ctx.move_to(S * 0.04, y0)
            for i in range(1, 40):
                x = S * (0.04 + 0.024 * i)
                ctx.line_to(x, y0 + 14 * math.sin(i * 0.35 + tt * 2.0 + j * 0.7) * (1 + 0.15 * j))
            ctx.stroke()
        ctx.select_font_face("STIX Two Math", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)   # has ∂ ∇ Δ ν
        ctx.set_source_rgba(r, g, b, min(1.0, tt / 1.2))
        ctx.set_font_size(30); ctx.move_to(S * 0.04, S * 0.2); ctx.show_text("∂u/∂t + (u·∇)u = −∇p + νΔu + f")
        ctx.set_font_size(30); ctx.move_to(S * 0.04, S * 0.32); ctx.show_text("∇·u = 0")
    elif kind == "vonneumann":                         # stored-program computer: programs and data share one memory
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        def box(x, y, w, h, label, hi, sub=None):
            ctx.set_source_rgba(r, g, b, 0.95 if hi else 0.5); ctx.set_line_width(3 if hi else 1.8)
            ctx.rectangle(x, y, w, h); ctx.stroke()
            ctx.set_font_size(30); e = ctx.text_extents(label)
            ctx.move_to(x + (w - e.x_advance) / 2, y + h / 2 + (0 if sub else 11)); ctx.show_text(label)
            if sub:
                ctx.set_font_size(22); e2 = ctx.text_extents(sub)
                ctx.set_source_rgba(r, g, b, 0.75); ctx.move_to(x + (w - e2.x_advance) / 2, y + h / 2 + 32); ctx.show_text(sub)
        phase = int(tt * 1.2) % 3                      # 0 fetch, 1 execute, 2 write back
        ctx.set_source_rgba(r, g, b, 0.3); ctx.set_line_width(1.2)
        ctx.rectangle(S * 0.06, S * 0.05, S * 0.88, S * 0.42); ctx.stroke()
        ctx.set_font_size(22); ctx.set_source_rgba(r, g, b, 0.7); ctx.move_to(S * 0.09, S * 0.1); ctx.show_text("中央处理器")
        ctrl, alu = (S * 0.1, S * 0.15, S * 0.36, S * 0.24), (S * 0.54, S * 0.15, S * 0.36, S * 0.24)
        mem = (S * 0.24, S * 0.56, S * 0.52, S * 0.2)
        box(*ctrl, "控制器", phase == 0)
        box(*alu, "运算器", phase == 1)
        box(*mem, "存储器", phase == 2, sub="程序 + 数据")
        box(S * 0.03, S * 0.84, S * 0.26, S * 0.12, "输入", False)
        box(S * 0.71, S * 0.84, S * 0.26, S * 0.12, "输出", False)
        paths = [((ctrl[0] + ctrl[2] / 2, mem[1]), (ctrl[0] + ctrl[2] / 2, ctrl[1] + ctrl[3])),   # fetch an instruction
                 ((ctrl[0] + ctrl[2], ctrl[1] + ctrl[3] / 2), (alu[0], alu[1] + alu[3] / 2)),     # tell the ALU what to do
                 ((alu[0] + alu[2] / 2, alu[1] + alu[3]), (alu[0] + alu[2] / 2, mem[1]))]         # write the result back
        for k_, ((x0, y0), (x1, y1)) in enumerate(paths):
            line(x0, y0, x1, y1, 2.5 if k_ == phase else 1.2, 0.9 if k_ == phase else 0.35)
            if k_ == phase:
                f = (tt * 1.2) % 1.0
                dot(mix(x0, x1, f), mix(y0, y1, f), 7, 1.0)
        line(S * 0.16, S * 0.84, S * 0.32, S * 0.76, 1.4, 0.45)
        line(S * 0.68, S * 0.76, S * 0.84, S * 0.84, 1.4, 0.45)
    elif kind == "backprop":                           # forward pulses (gold), then the error flows back (red)
        layers = [[(S * 0.15, S * (0.25 + 0.25 * i)) for i in range(3)], [(S * 0.5, S * (0.18 + 0.21 * i)) for i in range(4)],
                  [(S * 0.85, S * (0.38 + 0.24 * i)) for i in range(2)]]
        phase = (tt % 2.6) / 2.6
        back = phase > 0.5
        f = (phase - 0.5) * 2 if back else phase * 2
        for l in range(2):
            for i, (x0, y0) in enumerate(layers[l]):
                for j, (x1, y1) in enumerate(layers[l + 1]):
                    w = 1 + 2.5 * ((i * 7 + j * 3 + int(tt / 2.6)) % 4) / 3
                    line(x0, y0, x1, y1, w, 0.3)
                    if not back and l == int(f * 2 - 1e-9):
                        ff = f * 2 - l
                        dot(mix(x0, x1, ff), mix(y0, y1, ff), 3.5, 0.95)
                    if back and l == 1 - int(f * 2 - 1e-9):
                        ff = 1 - (f * 2 - (1 - l))
                        dot(mix(x0, x1, ff), mix(y0, y1, ff), 3.5, 0.95, hx("#e8432e"))
        for l in layers:
            for (x, y) in l:
                dot(x, y, 10, 0.95)

    elif kind == "xor":                                # one straight line never splits XOR: some point is always wrong
        pts = [(0.3, 0.3, 1), (0.7, 0.7, 1), (0.3, 0.7, 0), (0.7, 0.3, 0)]
        ang = 0.5 + tt * 0.9
        c0 = (0.5 + 0.1 * math.sin(tt * 1.3), 0.5 + 0.1 * math.cos(tt * 1.1))
        dx, dy = math.cos(ang), math.sin(ang)
        line((c0[0] - 1.5 * dx) * S, (c0[1] - 1.5 * dy) * S, (c0[0] + 1.5 * dx) * S, (c0[1] + 1.5 * dy) * S, 2.5, 0.8)
        side = [(dx * (y - c0[1]) - dy * (x - c0[0])) > 0 for x, y, _ in pts]
        wrong = [sd != bool(c) for sd, (_, _, c) in zip(side, pts)]
        if sum(wrong) > 2:
            wrong = [not w_ for w_ in wrong]
        for (x, y, c), wr in zip(pts, wrong):
            if c:
                dot(x * S, y * S, 18, 0.95)
            else:
                ctx.set_source_rgba(r, g, b, 0.95); ctx.set_line_width(4); ctx.arc(x * S, y * S, 16, 0, 2 * math.pi); ctx.stroke()
            if wr:
                ctx.set_source_rgba(0.91, 0.26, 0.18, 0.95); ctx.set_line_width(5)
                ctx.move_to(x * S - 26, y * S - 26); ctx.line_to(x * S + 26, y * S + 26)
                ctx.move_to(x * S + 26, y * S - 26); ctx.line_to(x * S - 26, y * S + 26); ctx.stroke()
    elif kind == "gan":                                # a forger and a judge train each other: fakes go from noise to passing
        face = ["00111100", "01000010", "10100101", "10000001", "10100101", "10011001", "01000010", "00111100"]
        tgt = np.array([[int(ch) for ch in row] for row in face], float)
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        def label(s_, x, y, size, a=0.9, c=None):
            ctx.set_font_size(size); e = ctx.text_extents(s_)
            ctx.set_source_rgba(*(c if c is not None else col), a); ctx.move_to(x - e.x_advance / 2, y); ctx.show_text(s_)
        def sample(x, y, cell, q, seed):
            v = mix(np.random.default_rng(seed).uniform(0, 1, (8, 8)), tgt, q)
            for i in range(8):
                for j in range(8):
                    ctx.set_source_rgba(r, g, b, 0.08 + 0.85 * v[j, i]); ctx.rectangle(x + i * cell, y + j * cell, cell - 1, cell - 1); ctx.fill()
        ctx.set_source_rgba(r, g, b, 0.7); ctx.set_line_width(2.5)
        ctx.rectangle(S * 0.03, S * 0.3, S * 0.25, S * 0.2); ctx.stroke()
        ctx.rectangle(S * 0.72, S * 0.3, S * 0.25, S * 0.2); ctx.stroke()
        label("生成器", S * 0.155, S * 0.42, 30); label("造假", S * 0.155, S * 0.25, 26, 0.6)
        label("判别器", S * 0.845, S * 0.42, 30); label("识假", S * 0.845, S * 0.25, 26, 0.6)
        per = 1.1
        k, ph = int(tt / per), (tt % per) / per
        q = min(1.0, 0.08 + 0.24 * k)
        cell = S * 0.022
        sample(mix(S * 0.3, S * 0.52, float(smooth(ph / 0.55))), S * 0.31, cell, q, k)
        if ph > 0.55:
            ok = q > 0.8
            label("真" if ok else "假", S * 0.845, S * 0.66, 56, 1.0, col if ok else (0.91, 0.26, 0.18))
        for j in range(min(k, 6)):                     # earlier fakes, getting better
            sample(S * (0.04 + 0.155 * j), S * 0.78, S * 0.015, min(1.0, 0.08 + 0.24 * j), j)
    elif kind == "resnet":                             # very deep: blocks pile up row by row; shortcuts skip every two blocks
        n = int(8 + 144 * float(smooth(tt / 2.2)))
        cols, cell, rowh = 16, S * 0.054, S * 0.074
        x0, y0 = S * 0.07, S * 0.1
        def pos(i):
            r_, c_ = divmod(i, cols)
            return x0 + (cols - 1 - c_ if r_ % 2 else c_) * cell, y0 + r_ * rowh
        for i in range(n):
            x, y = pos(i)
            ctx.set_source_rgba(r, g, b, 0.75); ctx.rectangle(x, y, cell * 0.62, cell * 0.62); ctx.fill()
            if i % 2 == 0 and i + 2 < n and i // cols == (i + 2) // cols:
                x2, _ = pos(i + 2)
                ctx.set_source_rgba(r, g, b, 0.45); ctx.set_line_width(1.6)
                ctx.move_to(x + cell * 0.31, y)
                ctx.curve_to(x + cell * 0.31, y - rowh * 0.42, x2 + cell * 0.31, y - rowh * 0.42, x2 + cell * 0.31, y); ctx.stroke()
        px, py = pos(int(tt * 30) % max(n, 1))
        dot(px + cell * 0.31, py + cell * 0.31, 7, 1.0)
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(56)
        lab = f"{n} 层"; e = ctx.text_extents(lab)
        ctx.set_source_rgba(r, g, b, 1.0); ctx.move_to(S * 0.95 - e.x_advance, S * 0.97); ctx.show_text(lab)
    elif kind == "cloze":                              # fill in the blank, reading from both sides
        sents = [("床前明月光", 3, "月日灯"), ("疑是地上霜", 4, "霜雪光"), ("举头望明月", 4, "月星日")]
        per = 1.7
        k, ph = int(tt / per) % 3, (tt % per) / per
        chars, m, cands = sents[k]
        bw, gap = S * 0.15, S * 0.03
        x0, y0 = (S - 5 * bw - 4 * gap) / 2, S * 0.3
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        xm = x0 + m * (bw + gap) + bw / 2
        for i in range(5):                             # both sides look at the blank
            if i == m:
                continue
            xi = x0 + i * (bw + gap) + bw / 2
            w_ = 0.35 + 0.5 * math.sin(tt * 4 + i) ** 2
            ctx.set_source_rgba(r, g, b, (0.25 + 0.5 * w_) * min(1.0, ph / 0.3)); ctx.set_line_width(2)
            ctx.move_to(xi, y0 - 6); ctx.curve_to(xi, y0 - S * 0.14, xm, y0 - S * 0.14, xm, y0 - 6); ctx.stroke()
        for i, ch in enumerate(chars):
            x = x0 + i * (bw + gap)
            ctx.set_source_rgba(r, g, b, 0.9 if i == m else 0.4); ctx.set_line_width(3 if i == m else 1.6)
            if i == m:
                ctx.set_dash([8, 6])
            ctx.rectangle(x, y0, bw, bw); ctx.stroke(); ctx.set_dash([])
            s_, a_ = (ch, 1.0) if (i != m or ph > 0.6) else ("？", 0.5)
            ctx.set_font_size(58); e = ctx.text_extents(s_)
            ctx.set_source_rgba(r, g, b, a_); ctx.move_to(x + (bw - e.x_advance) / 2 - e.x_bearing * 0.0, y0 + bw * 0.74); ctx.show_text(s_)
        for j, (cc, pr) in enumerate(zip(cands, (0.82, 0.11, 0.05))):
            y = S * 0.6 + j * S * 0.11
            ctx.set_font_size(40); ctx.set_source_rgba(r, g, b, 0.9); ctx.move_to(S * 0.12, y + S * 0.06); ctx.show_text(cc)
            ctx.set_source_rgba(r, g, b, 0.3 + 0.6 * pr)
            ctx.rectangle(S * 0.24, y + S * 0.015, S * 0.66 * pr * float(smooth(min(1.0, ph / 0.5))), S * 0.05); ctx.fill()
    elif kind == "params":                             # GPT-1 -> GPT-2 -> GPT-3: square area = parameter count
        items = [("GPT-1", 0.117, "1.17 亿", 0.2, S * 0.02), ("GPT-2", 1.5, "15 亿", 0.8, S * 0.2), ("GPT-3", 175.0, "1750 亿", 1.4, S * 0.4)]
        base = S * 0.74
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        for name, pc, lab, t0, x in items:
            k = float(smooth((tt - t0) / (1.0 if pc > 100 else 0.4)))
            if k <= 0:
                continue
            side = max(4.0, S * 0.56 * math.sqrt(pc / 175.0) * k)
            ctx.set_source_rgba(r, g, b, 0.9 if pc > 100 else 0.75); ctx.rectangle(x, base - side, side, side); ctx.fill()
            ctx.set_font_size(26); ctx.set_source_rgba(r, g, b, min(1.0, k * 2)); ctx.move_to(x, base + 38); ctx.show_text(name)
            ctx.set_font_size(26); ctx.set_source_rgba(r, g, b, 0.75 * min(1.0, k * 2)); ctx.move_to(x, base + 76); ctx.show_text(lab)
        line(S * 0.02, base, S * 0.98, base, 1.5, 0.5)
    elif kind == "computer":                           # an AI uses a computer like a person: looks, moves the mouse, clicks, types
        sx0, sy0, sw, sh = S * 0.06, S * 0.08, S * 0.88, S * 0.6
        ctx.set_source_rgba(r, g, b, 0.8); ctx.set_line_width(3); ctx.rectangle(sx0, sy0, sw, sh); ctx.stroke()
        line(S * 0.5, sy0 + sh, S * 0.5, S * 0.76, 3, 0.6); line(S * 0.36, S * 0.76, S * 0.64, S * 0.76, 3, 0.6)
        field = (sx0 + sw * 0.08, sy0 + sh * 0.12, sw * 0.62, sh * 0.15)
        btns = [(sx0 + sw * (0.08 + 0.3 * i), sy0 + sh * 0.42, sw * 0.24, sh * 0.16) for i in range(3)]
        rows = [(sx0 + sw * 0.08, sy0 + sh * (0.7 + 0.1 * i), sw * (0.8 - 0.15 * i), sh * 0.04) for i in range(3)]
        seg = 0.9
        targets = [(field[0] + field[2] * 0.5, field[1] + field[3] * 0.5)] + [(bx + bw_ / 2, by + bh_ / 2) for bx, by, bw_, bh_ in btns]
        order = [0, 1, 3, 2]
        i_ = int(tt / seg)
        ph = (tt % seg) / seg
        a0, a1 = targets[order[(i_ - 1) % 4]], targets[order[i_ % 4]]
        cx_, cy_ = mix(a0[0], a1[0], float(smooth(ph / 0.6))), mix(a0[1], a1[1], float(smooth(ph / 0.6)))
        for j, (bx, by, bw_, bh_) in enumerate([field] + btns):
            hot = j == order[i_ % 4] and ph > 0.6
            ctx.set_source_rgba(r, g, b, 0.95 if hot else 0.45); ctx.set_line_width(3 if hot else 1.6)
            ctx.rectangle(bx, by, bw_, bh_); ctx.stroke()
            if hot:
                ctx.set_source_rgba(r, g, b, 0.18); ctx.rectangle(bx, by, bw_, bh_); ctx.fill()
        for bx, by, bw_, bh_ in rows:
            ctx.set_source_rgba(r, g, b, 0.3); ctx.rectangle(bx, by, bw_, bh_); ctx.fill()
        nch = int(tt * 5) % 13                         # typing into the field
        for c_ in range(nch):
            ctx.set_source_rgba(r, g, b, 0.85); ctx.rectangle(field[0] + 10 + c_ * 16, field[1] + field[3] * 0.3, 11, field[3] * 0.4); ctx.fill()
        if ph > 0.6:                                   # click ripple
            k = (ph - 0.6) / 0.4
            ctx.set_source_rgba(r, g, b, 1 - k); ctx.set_line_width(2.5); ctx.arc(a1[0], a1[1], 6 + 30 * k, 0, 2 * math.pi); ctx.stroke()
        ctx.set_source_rgba(1, 0.98, 0.94, 1.0)     # the pointer
        ctx.move_to(cx_, cy_); ctx.line_to(cx_, cy_ + 34); ctx.line_to(cx_ + 9, cy_ + 26); ctx.line_to(cx_ + 16, cy_ + 40)
        ctx.line_to(cx_ + 21, cy_ + 37); ctx.line_to(cx_ + 14, cy_ + 24); ctx.line_to(cx_ + 25, cy_ + 23); ctx.close_path(); ctx.fill()
    elif kind == "sandbox":                            # a locked test box: the agent probes the walls, breaks out, emails the researcher
        bx, by, bw_, bh_ = S * 0.06, S * 0.3, S * 0.46, S * 0.46
        crack_t = 2.2
        ctx.set_source_rgba(r, g, b, 0.85); ctx.set_line_width(3.5); ctx.rectangle(bx, by, bw_, bh_); ctx.stroke()
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(28)
        ctx.move_to(bx, by - 16); ctx.show_text("测试沙盒")
        hx_, hy_ = S * 0.84, S * 0.14                  # the researcher
        dot(hx_, hy_, 16, 0.9)
        ctx.set_source_rgba(r, g, b, 0.9); ctx.arc(hx_, hy_ + 52, 30, math.pi, 2 * math.pi); ctx.fill()
        ctx.set_font_size(26); e = ctx.text_extents("研究员"); ctx.move_to(hx_ - e.x_advance / 2, hy_ + 92); ctx.show_text("研究员")
        if tt < crack_t:
            u, v = (tt * 0.9) % 2, (tt * 1.3) % 2
            px = bx + 20 + (bw_ - 40) * (u if u < 1 else 2 - u)
            py = by + 20 + (bh_ - 40) * (v if v < 1 else 2 - v)
            dot(px, py, 9, 1.0)
        else:
            k = float(smooth((tt - crack_t) / 0.6))
            cr = [(bx + bw_, by + bh_ * 0.28), (bx + bw_ - 12, by + bh_ * 0.38), (bx + bw_ + 9, by + bh_ * 0.48),
                  (bx + bw_ - 8, by + bh_ * 0.58), (bx + bw_ + 6, by + bh_ * 0.7)]
            ctx.set_source_rgba(0.91, 0.26, 0.18, 0.95); ctx.set_line_width(4)
            ctx.move_to(*cr[0])
            for q_ in cr[1:1 + int(round(k * 4))]:
                ctx.line_to(*q_)
            ctx.stroke()
            px = mix(bx + bw_ - 24, S * 0.66, float(smooth((tt - crack_t - 0.3) / 0.6)))
            dot(px, by + bh_ * 0.5, 9, 1.0)
            ke = float(smooth((tt - crack_t - 0.9) / 0.9))
            if ke > 0:                                 # the email
                ex, ey = mix(S * 0.66, hx_ - 22, ke), mix(by + bh_ * 0.5, hy_ + 130, ke)
                ctx.set_source_rgba(r, g, b, 1.0); ctx.set_line_width(2.5)
                ctx.rectangle(ex - 26, ey - 17, 52, 34); ctx.stroke()
                ctx.move_to(ex - 26, ey - 17); ctx.line_to(ex, ey + 4); ctx.line_to(ex + 26, ey - 17); ctx.stroke()
    elif kind == "robot":                              # told in words, a robot arm picks up the right thing and puts it away
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(30)
        ctx.set_source_rgba(r, g, b, min(1.0, tt / 0.4)); ctx.move_to(S * 0.06, S * 0.1); ctx.show_text("「把香蕉放进碗里」")
        tbl = S * 0.82
        line(S * 0.02, tbl, S * 0.98, tbl, 2.5, 0.6)
        base = (S * 0.14, tbl)
        ban0, bowl = (S * 0.5, tbl - 14), (S * 0.8, tbl - 30)
        ph = (tt % 3.6) / 3.6                          # reach, grip, carry, drop
        if ph < 0.3:
            tgt, held = (mix(S * 0.3, ban0[0], ph / 0.3), mix(S * 0.4, ban0[1] - 20, ph / 0.3)), False
        elif ph < 0.4:
            tgt, held = (ban0[0], ban0[1] - 20), True
        elif ph < 0.75:
            k = float(smooth((ph - 0.4) / 0.35))
            tgt, held = (mix(ban0[0], bowl[0], k), mix(ban0[1] - 20, bowl[1] - 70, k) - 60 * math.sin(math.pi * k)), True
        else:
            tgt, held = (bowl[0], bowl[1] - 70), False
        L1, L2 = S * 0.36, S * 0.34                    # two-link arm, inverse kinematics
        dxa, dya = tgt[0] - base[0], tgt[1] - base[1]
        dd = min(L1 + L2 - 1, math.hypot(dxa, dya))
        a2 = math.acos(max(-1, min(1, (dd * dd - L1 * L1 - L2 * L2) / (2 * L1 * L2))))
        a1 = math.atan2(dya, dxa) - math.atan2(L2 * math.sin(a2), L1 + L2 * math.cos(a2))
        ex_, ey_ = base[0] + L1 * math.cos(a1), base[1] + L1 * math.sin(a1)
        line(*base, ex_, ey_, 9, 0.85); line(ex_, ey_, *tgt, 7, 0.85)
        dot(*base, 14, 0.9); dot(ex_, ey_, 9, 0.9)
        line(tgt[0] - 12, tgt[1], tgt[0] - 12, tgt[1] + 18, 4, 0.9); line(tgt[0] + 12, tgt[1], tgt[0] + 12, tgt[1] + 18, 4, 0.9)
        bpos = (tgt[0], tgt[1] + 22) if held else (ban0 if ph < 0.4 else (bowl[0], bowl[1] - 4))
        ctx.set_source_rgba(r, g, b, 1.0); ctx.set_line_width(9)
        ctx.arc(bpos[0], bpos[1] - 24, 26, 0.35, math.pi - 0.35); ctx.stroke()
        ctx.set_line_width(3); ctx.arc(bowl[0], bowl[1] - 10, 46, 0, math.pi); ctx.stroke()
        line(bowl[0] - 50, bowl[1] - 10, bowl[0] + 50, bowl[1] - 10, 3, 0.9)
        for j, (ox, oy) in enumerate(((S * 0.3, tbl - 18), (S * 0.62, tbl - 22))):   # other things on the table, left alone
            ctx.set_source_rgba(r, g, b, 0.45); ctx.rectangle(ox - 18, oy - 18 - 8 * j, 36, 36 + 8 * j); ctx.fill()
    elif kind == "metr":                               # how long a task AI can finish half the time, log scale: a straight climb
        pts = [("GPT-4", 2023.2, 3.5, "3.5 分钟"), ("Claude 3.7", 2025.15, 60, ""), ("o3", 2025.29, 121, ""),
               ("GPT-5", 2025.6, 214, ""), ("Opus 4.5", 2025.9, 320, "5 小时 20 分")]
        xo = lambda yr: S * (0.24 + 0.7 * (yr - 2023.0) / 3.1)
        yo = lambda m_: S * (0.84 - 0.68 * (math.log10(m_) - math.log10(2)) / (math.log10(500) - math.log10(2)))
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); ctx.set_font_size(22)
        for m_, lab in ((1 * 3, "3 分钟"), (10, "10 分钟"), (60, "1 小时"), (300, "5 小时")):
            line(S * 0.24, yo(m_), S * 0.97, yo(m_), 1, 0.18)
            ctx.set_source_rgba(r, g, b, 0.6); e = ctx.text_extents(lab); ctx.move_to(S * 0.22 - e.x_advance, yo(m_) + 7); ctx.show_text(lab)
        for yr in (2023, 2024, 2025, 2026):
            ctx.set_source_rgba(r, g, b, 0.6); e = ctx.text_extents(str(yr)); ctx.move_to(xo(yr) - e.x_advance / 2, S * 0.92); ctx.show_text(str(yr))
        n = min(len(pts), 1 + int(tt * 1.5))
        ctx.set_source_rgba(r, g, b, 0.85); ctx.set_line_width(3)
        for i in range(n):
            x, y = xo(pts[i][1]), yo(pts[i][2])
            ctx.line_to(x, y) if i else ctx.move_to(x, y)
        ctx.stroke()
        ctx.set_font_size(24)
        for i in range(n):
            name, yr, m_, val = pts[i]
            x, y = xo(yr), yo(m_)
            dot(x, y, 9 if val else 6, 1.0)
            if val:                                    # only the two ends are named: GPT-4 and Opus 4.5
                ctx.set_source_rgba(r, g, b, 0.9); ctx.set_font_size(26)
                if i == 0:
                    ctx.move_to(x + 16, y + 8); ctx.show_text(f"{name}  {val}")
                else:
                    e = ctx.text_extents(name); ctx.move_to(x - e.x_advance - 16, y - 34); ctx.show_text(name)
                    e = ctx.text_extents(val); ctx.move_to(x - e.x_advance - 16, y + 2); ctx.show_text(val)
    elif kind == "math10":                             # ten research-level results in one release, each with a machine-checked proof
        areas = ["球堆积", "纠错码", "群论构造", "猜想反例", "计算下限", "量子博弈", "格点难题", "体积猜想", "拉姆齐数", "极值图论"]
        n = min(10, int(tt * 4))
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(28)
        for i, lab in enumerate(areas):
            cx_, cy_ = S * (0.05 + 0.47 * (i % 2)), S * (0.06 + 0.18 * (i // 2))
            on = i < n
            ctx.set_source_rgba(r, g, b, 0.9 if on else 0.2); ctx.set_line_width(2 if on else 1.2)
            ctx.rectangle(cx_, cy_, S * 0.43, S * 0.14); ctx.stroke()
            ctx.move_to(cx_ + 16, cy_ + S * 0.09); ctx.show_text(lab)
            if on:                                     # a check mark: the Lean proof went through
                ctx.set_line_width(4); ctx.move_to(cx_ + S * 0.33, cy_ + S * 0.07)
                ctx.line_to(cx_ + S * 0.36, cy_ + S * 0.1); ctx.line_to(cx_ + S * 0.41, cy_ + S * 0.04); ctx.stroke()
    elif kind == "selfloop":                           # the model in the middle; its early versions debug, deploy, evaluate it
        cx_, cy_, R = S * 0.5, S * 0.48, S * 0.33
        ctx.set_source_rgba(r, g, b, 0.35); ctx.set_line_width(2.5); ctx.arc(cx_, cy_, R, 0, 2 * math.pi); ctx.stroke()
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(34)
        for k_, (s_, dy_) in enumerate((("GPT-5.3", -6), ("Codex", 32))):
            e = ctx.text_extents(s_); ctx.set_source_rgba(r, g, b, 1.0); ctx.move_to(cx_ - e.x_advance / 2, cy_ + dy_); ctx.show_text(s_)
        steps = ["调试训练", "管理部署", "分析评测"]
        n = min(3, int(tt * 1.5) + 1)
        ctx.set_font_size(28)
        for i, st in enumerate(steps):
            ang = -math.pi / 2 + i * 2 * math.pi / 3
            x, y = cx_ + R * math.cos(ang), cy_ + R * math.sin(ang)
            on = i < n
            e = ctx.text_extents(st)
            ctx.set_source_rgba(0.06, 0.05, 0.04, 1.0); ctx.rectangle(x - e.x_advance / 2 - 12, y - 24, e.x_advance + 24, 44); ctx.fill()
            ctx.set_source_rgba(r, g, b, 0.95 if on else 0.25); ctx.set_line_width(2 if on else 1.2)
            ctx.rectangle(x - e.x_advance / 2 - 12, y - 24, e.x_advance + 24, 44); ctx.stroke()
            ctx.move_to(x - e.x_advance / 2, y + 9); ctx.show_text(st)
        ang = tt * 2.0 - math.pi / 2                   # a pulse runs round the loop
        dot(cx_ + R * math.cos(ang), cy_ + R * math.sin(ang), 8, 1.0)
    elif kind.startswith("bars:"):                     # before -> after, as bars; the last bar is the new one
        d_ = BARS[kind[5:]]
        items, vmax = d_["items"], d_["vmax"]
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); ctx.set_font_size(27)
        ctx.set_source_rgba(r, g, b, 0.75); ctx.move_to(S * 0.03, S * 0.1); ctx.show_text(d_["unit"])
        n = len(items)
        top, rowh = S * 0.2, min(S * 0.24, S * 0.66 / n)
        x0, wmax = S * 0.34, S * 0.46
        for i, (lab, v, disp) in enumerate(items):
            k = float(smooth((tt - 0.2 - 0.5 * i) / 0.7))
            hi = i == n - 1
            y = top + i * rowh
            ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if hi else cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(28); ctx.set_source_rgba(r, g, b, 1.0 if hi else 0.7)
            e = ctx.text_extents(lab); ctx.move_to(x0 - 14 - e.x_advance, y + rowh * 0.5); ctx.show_text(lab)
            bw_ = max(3.0, wmax * v / vmax * k)
            ctx.set_source_rgba(r, g, b, 0.95 if hi else 0.4); ctx.rectangle(x0, y + rowh * 0.18, bw_, rowh * 0.46); ctx.fill()
            ctx.set_font_size(34 if hi else 28); ctx.set_source_rgba(r, g, b, k)
            ctx.move_to(x0 + bw_ + 12, y + rowh * 0.53); ctx.show_text(disp)
        if d_.get("ref"):
            rv, rl = d_["ref"]
            xr = x0 + wmax * rv / vmax
            ctx.set_source_rgba(r, g, b, 0.7); ctx.set_line_width(2); ctx.set_dash([7, 6])
            ctx.move_to(xr, top - 6); ctx.line_to(xr, top + n * rowh + 6); ctx.stroke(); ctx.set_dash([])
            ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); ctx.set_font_size(23)
            e = ctx.text_extents(rl); ctx.move_to(min(S - e.x_advance - 4, xr - e.x_advance / 2), top + n * rowh + 34); ctx.show_text(rl)
    elif kind == "elo":                                # AlphaGo Zero from nothing: passes the Lee Sedol version at 36 h, 100:0 at 72 h
        xo = lambda h_: S * (0.08 + 0.84 * h_ / 72)
        yo = lambda e_: S * (0.84 - 0.66 * e_)
        curve = lambda h_: 0.92 / (1 + math.exp(-(h_ - 28) / 8))
        lee = curve(36)
        line(S * 0.08, S * 0.84, S * 0.94, S * 0.84, 1.6, 0.6)
        ctx.set_source_rgba(r, g, b, 0.6); ctx.set_line_width(2); ctx.set_dash([7, 6])
        ctx.move_to(S * 0.08, yo(lee)); ctx.line_to(S * 0.94, yo(lee)); ctx.stroke(); ctx.set_dash([])
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); ctx.set_font_size(24)
        ctx.move_to(S * 0.1, yo(lee) + 32); ctx.show_text("击败李世石的那一版")
        for h_, lab in ((0, "0"), (36, "36 小时"), (72, "72 小时")):
            e = ctx.text_extents(lab); ctx.move_to(min(S * 0.98 - e.x_advance, max(0, xo(h_) - e.x_advance / 2)), S * 0.92); ctx.show_text(lab)
        hn = 72 * float(smooth(tt / 2.4))
        ctx.set_source_rgba(r, g, b, 1.0); ctx.set_line_width(4)
        for j in range(61):
            h_ = hn * j / 60
            ctx.line_to(xo(h_), yo(curve(h_))) if j else ctx.move_to(xo(0), yo(curve(0)))
        ctx.stroke()
        dot(xo(hn), yo(curve(hn)), 8, 1.0)
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(28)
        if hn >= 36:
            dot(xo(36), yo(lee), 7, 0.9); ctx.move_to(xo(36) - 150, yo(lee) - 20); ctx.show_text("超过它")
        if hn >= 71.5:
            ctx.set_font_size(40); e = ctx.text_extents("100 : 0"); ctx.move_to(S * 0.94 - e.x_advance, yo(curve(72)) - 26); ctx.show_text("100 : 0")
    elif kind == "voice":                              # you stop talking; about 0.32 s later the AI answers
        ph = tt % 3.4
        mid = S * 0.42
        def wave(xa, xb, seed, prog):
            rng_ = np.random.default_rng(seed)
            amps = rng_.uniform(0.25, 1.0, 40)
            for j in range(int(40 * prog)):
                x = xa + (xb - xa) * j / 40
                h_ = S * 0.16 * amps[j] * math.sin(math.pi * j / 40) ** 0.5
                line(x, mid - h_, x, mid + h_, 4, 0.9)
        wave(S * 0.03, S * 0.4, 1, min(1.0, ph / 1.0))
        if ph > 1.0:
            g0, g1 = S * 0.42, S * 0.58
            ctx.set_source_rgba(r, g, b, 0.8); ctx.set_line_width(2)
            ctx.move_to(g0, mid - S * 0.22); ctx.line_to(g0, mid - S * 0.26); ctx.line_to(g1, mid - S * 0.26); ctx.line_to(g1, mid - S * 0.22); ctx.stroke()
            ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(40)
            e = ctx.text_extents("0.32 秒"); ctx.move_to((g0 + g1) / 2 - e.x_advance / 2, mid - S * 0.3); ctx.show_text("0.32 秒")
        if ph > 1.25:
            wave(S * 0.6, S * 0.97, 2, min(1.0, (ph - 1.25) / 1.0))
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); ctx.set_font_size(28)
        ctx.set_source_rgba(r, g, b, 0.8); ctx.move_to(S * 0.12, S * 0.72); ctx.show_text("你说话")
        if ph > 1.25:
            ctx.move_to(S * 0.68, S * 0.72); ctx.show_text("AI 回答")
    elif kind == "world":                              # a world generated as you move through it: the ground streams in, things appear
        hz = S * 0.44
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(28)
        ctx.set_source_rgba(r, g, b, min(1.0, tt / 0.4)); ctx.move_to(S * 0.04, S * 0.1); ctx.show_text("「雪山脚下的小镇」")
        line(0, hz, S, hz, 1.5, 0.6)
        sp = tt * 0.45
        for i in range(14):                            # ground lines rushing towards you
            z = ((i / 14.0) - sp) % 1.0
            line(0, hz + (S - hz) * z ** 2.2, S, hz + (S - hz) * z ** 2.2, 1.2, 0.15 + 0.6 * z)
        for j in range(-7, 8):
            line(S / 2 + j * S * 0.015, hz, S / 2 + j * S * 0.2, S, 1.0, 0.3)
        for m in range(5):                             # mountains on the horizon
            mx = S * (0.1 + 0.2 * m)
            ctx.set_source_rgba(r, g, b, 0.35); ctx.move_to(mx - S * 0.12, hz); ctx.line_to(mx, hz - S * (0.08 + 0.05 * (m % 2))); ctx.line_to(mx + S * 0.12, hz); ctx.fill()
        rng_ = np.random.default_rng(3)
        for k in range(12):                            # houses and trees appear far away and come closer
            z0, xo = rng_.uniform(0, 1), rng_.choice([-1, 1]) * rng_.uniform(0.25, 1.0)
            z = (z0 + sp * 0.6) % 1.0
            sc = z ** 2.2
            x = S / 2 + xo * S * 0.55 * (0.05 + sc)
            y = hz + (S - hz) * sc
            h_ = S * (0.01 + 0.28 * sc)
            ctx.set_source_rgba(r, g, b, 0.25 + 0.7 * z); ctx.rectangle(x - h_ * 0.3, y - h_, h_ * 0.6, h_); ctx.fill()
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); ctx.set_font_size(24)
        ctx.set_source_rgba(r, g, b, 0.85); e = ctx.text_extents("实时生成 · 24 帧/秒"); ctx.move_to(S * 0.96 - e.x_advance, S * 0.97); ctx.show_text("实时生成 · 24 帧/秒")
    elif kind in ("video", "video_av"):                # frames of a clip appear one after another; video_av adds its sound
        n = 4
        fw, fh = S * 0.21, S * 0.3
        y0 = S * 0.18
        ctx.set_source_rgba(r, g, b, 0.45); ctx.set_line_width(2)
        ctx.rectangle(S * 0.02, y0 - S * 0.06, S * 0.96, fh + S * 0.12); ctx.stroke()
        for k in range(14):                            # sprocket holes
            for yy in (y0 - S * 0.045, y0 + fh + S * 0.02):
                ctx.rectangle(S * (0.04 + 0.068 * k), yy, S * 0.03, S * 0.02); ctx.fill()
        shown = min(n, int(tt * 1.6) + 1)
        for i in range(n):
            x = S * (0.04 + 0.235 * i)
            ctx.set_source_rgba(r, g, b, 0.85 if i < shown else 0.15); ctx.set_line_width(2); ctx.rectangle(x, y0, fw, fh); ctx.stroke()
            if i < shown:                              # a little scene: horizon, sun, a figure walking across the shots
                k = float(smooth((tt * 1.6 - i) / 0.6))
                hy = y0 + fh * 0.72
                line(x + 2, hy, x + fw - 2, hy, 1.5, 0.6 * k)
                dot(x + fw * 0.8, y0 + fh * 0.22, 9, 0.6 * k)
                fx = x + fw * (0.2 + 0.18 * i)
                dot(fx, hy - fh * 0.24, 6, k)
                line(fx, hy - fh * 0.2, fx, hy - fh * 0.07, 3, k)
                sw = 7 if i % 2 else -7
                line(fx, hy - fh * 0.07, fx - sw, hy, 3, k); line(fx, hy - fh * 0.07, fx + sw, hy, 3, k)
        ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); ctx.set_font_size(24)
        if kind == "video_av":
            for i in range(min(3, shown)):
                ctx.set_source_rgba(r, g, b, 0.8); ctx.move_to(S * (0.06 + 0.235 * i), y0 + fh + S * 0.12); ctx.show_text(f"镜头 {i + 1}")
            mid = S * 0.76
            for j in range(int(60 * min(1.0, tt / 2.0))):   # the sound, generated together with the picture
                x = S * 0.04 + S * 0.92 * j / 60
                h_ = S * 0.06 * (0.3 + 0.7 * abs(math.sin(j * 0.7 + tt * 3)) * (0.5 + 0.5 * math.sin(j * 0.21)))
                line(x, mid - h_, x, mid + h_, 3, 0.85)
            ctx.set_source_rgba(r, g, b, 0.8); ctx.move_to(S * 0.04, S * 0.92); ctx.show_text("声音和画面一起生成")
        else:
            ctx.set_source_rgba(r, g, b, 0.8); ctx.move_to(S * 0.04, S * 0.72); ctx.show_text("一句话 → 一段视频")

def palette_weights(T):
    cold = winter_amount(T)
    mono = float(1 - smooth((T - 2012.74) / 0.22))    # archival grey until colour returns on the AlexNet burst
    warm = 1.0
    return dict(mono=mono * (1 - cold), cold=cold, warm=warm)


def pal_color(key, T):
    w = palette_weights(T)
    c = hx(PAL["warm"][key])
    c = mix(c, hx(PAL["mono"][key]), w["mono"])
    return mix(c, hx(PAL["cold"][key]), w["cold"])


def clock_str(T):
    import datetime
    y = int(math.floor(T + 1e-6))
    if T < 2023:
        return str(y), 150
    d = datetime.date(y, 1, 1) + datetime.timedelta(days=int((T - y) * 365))
    if T < 2026:
        return f"{y}.{d.month:02d}", 140
    return f"{y}.{d.month:02d}.{d.day:02d}", 112


# ======================================================================================
# camera: the past squeezes left (x) and down (y); the overview at the end widens the frame
# ======================================================================================
X_LEFT, X_HEAD, Y_TOP, Y_BASE = LAYOUT["x_left"], LAYOUT["x_head"] * W, LAYOUT["y_top"] * H, LAYOUT["y_base"] * H


def camera(T, ov=0.0):
    c = float(np.interp(T, [1943, 1968, 1985, 2000, 2012, 2018, 2022, 2024, 2025.5, 2026.8],
                        [90, 70, 45, 30, 16, 9, 5, 3, 2, 1.4]))
    c = mix(c, 6.0, ov)
    x_head = min(X_HEAD, X_LEFT + (X_HEAD - X_LEFT) * (T - 1943.0) / 25.0)
    x_head = mix(x_head, 0.90 * W, ov)
    i = min(int(np.searchsorted(GRID, T)), len(GRID) - 1)
    floor = float(np.interp(T, [1943, 1965, 1990], [125, 100, 60]))
    m = max(floor, float(data()["run_max"][i]))
    p = mix(float(np.interp(T, [1943, 2000, 2012, 2020, 2026.8], [1.0, 1.0, 1.15, 1.35, 1.6])), 1.6, ov)
    return dict(T=T, c=c, x_head=x_head, m=m, p=p, D=math.log1p(max(T - 1943.0, 1e-3) / c),
                x_left=mix(X_LEFT, LAYOUT["x_left_ov"], ov),
                y_top=mix(Y_TOP, LAYOUT["y_top_ov"] * H, ov), y_base=mix(Y_BASE, LAYOUT["y_base_ov"] * H, ov))


def sx(cam, t):
    u = np.log1p(np.maximum(cam["T"] - np.asarray(t, float), 0.0) / cam["c"]) / cam["D"]
    return cam["x_head"] - (cam["x_head"] - cam["x_left"]) * u


def sy(cam, v):
    return cam["y_base"] - (cam["y_base"] - cam["y_top"]) * (np.maximum(np.asarray(v, float), 0.0) / cam["m"]) ** cam["p"]


def line_y(cam, t):
    return sy(cam, I(t))


def sample_ts(cam, t0, n=360):
    T, c, D = cam["T"], cam["c"], cam["D"]
    u0 = math.log1p(max(T - t0, 0.0) / c) / D
    return T - c * np.expm1(np.linspace(u0, 0.0, n) * D)


def wrap(s, n):
    """Wrap to n full-width characters per line, rebalanced so the last line is never one or two characters."""
    lines = _wrap(s, n)
    k = n
    while len(lines) > 1 and len(lines[-1].strip()) <= 2 and k > 8:
        k -= 1
        lines = _wrap(s, k)
    return lines


def _wrap(s, n):
    """Latin words and numbers are never split (they count ~0.55 each); no line starts with closing punctuation; when a
    line overflows it breaks after the last comma/colon/space in it if that is past 25% of the line."""
    import re
    tokens = re.findall(r"[A-Za-z0-9½.%·'’\-+/]+|\s|.", s)
    width = lambda tok: sum(0.55 if ord(c) < 0x2E80 else 1.0 for c in tok)
    out, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if sum(width(x) for x in cur) > n and len(cur) > 1 and tok not in "，。、；：」）！？":
            brk = max((i for i, x in enumerate(cur[:-1]) if x in "，、；：。"), default=-1)   # punctuation first
            ok = brk >= 0 and sum(width(x) for x in cur[:brk + 1]) >= 0.25 * n
            if not ok:                                 # a space only if it leaves at least half a line (no lone "OpenAI")
                brk = max((i for i, x in enumerate(cur[:-1]) if x == " "), default=-1)
                ok = brk >= 0 and sum(width(x) for x in cur[:brk + 1]) >= 0.5 * n
            if ok:
                out.append("".join(cur[:brk + 1]).rstrip()); cur = [x for x in cur[brk + 1:] if not (x.isspace() and not out[-1])]
            else:
                out.append("".join(cur[:-1]).rstrip()); cur = [] if tok.isspace() else [tok]
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def date_label(t):
    import datetime
    y = int(math.floor(t + 1e-9))
    d = datetime.date(y, 1, 1) + datetime.timedelta(days=int((t - y) * 365))
    return d


def ruler_ticks(cam):
    """Ticks for the bottom time axis: decades, years, then months (2023+), then days (2026), keeping 84 px apart."""
    T = cam["T"]
    cands = [(float(y), str(y), 0) for y in range(1950, 2030, 10) if y <= T]
    cands += [(float(y), str(y), 1) for y in range(1943, 2027) if y <= T and y % 10]
    cands += [(y + (m - 1) / 12, f"{y}.{m:02d}", 2) for y in (2023, 2024, 2025, 2026) for m in range(1, 13)
              if (m > 1) and y + (m - 1) / 12 <= T]
    if T > 2026.3:
        import datetime
        d0 = datetime.date(2026, 5, 1)
        for k in range(0, 200, 4):
            d = d0 + datetime.timedelta(days=k)
            ty = 2026 + (d - datetime.date(2026, 1, 1)).days / 365
            if ty <= T and d.day != 1:
                cands.append((ty, f"{d.month:02d}.{d.day:02d}", 3))
    placed = []
    for lvl in range(4):
        for ty, lab, l in sorted([c for c in cands if c[2] == lvl], key=lambda c: -c[0]):
            x = float(sx(cam, ty))
            if x < cam["x_left"] - 2 or any(abs(x - px) < 84 for px, _, _ in placed):
                continue
            placed.append((x, lab, lvl))
    return placed


# ======================================================================================
# the frame
# ======================================================================================
WIN = dict(x=36, y=96, w=500, title_y=150, viz_y=176, viz_s=360, desc_y=596)


def draw_window_text(ctx, bt, a, T):
    cream = pal_color("cream", T)
    text(ctx, bt["title"], WIN["x"] + 14, WIN["title_y"], 34, cream, a, bold=True)
    y0 = WIN["desc_y"]
    if bt.get("first"):                                # the "first time" line, in gold, above the explanation
        for ln in wrap(bt["first"], 14):
            text(ctx, ln, WIN["x"] + 14, y0, 32, hx("#f2c14e"), a, bold=True)
            y0 += 44
        y0 += 4
    lines = wrap(bt["desc"], 15)[:3]
    for k, ln in enumerate(lines):
        text(ctx, ln, WIN["x"] + 14, y0 + k * 40, 30, cream, 0.92 * a)
    y = y0 + len(lines) * 40
    if bt.get("story"):
        sl = bt["story"].split("\n")[:2] if "\n" in bt["story"] else wrap(bt["story"], 17)[:2]
        for k, ln in enumerate(sl):
            text(ctx, ln, WIN["x"] + 14, y + 16 + k * 34, 25, cream, 0.6 * a)
        y += 16 + 34 * len(sl)
    if bt.get("fore"):                                 # foreshadowing: how long until what this node leads to
        text(ctx, bt["fore"], WIN["x"] + 14, y + 12, 26, hx("#f2c14e"), 0.85 * a)
        y += 46
    return y


@lru_cache(None)
def diffusion_arrays(n=128):
    """The Turing portrait (public domain, already in the film) as the target, plus a fixed noise field."""
    im = Image.open(os.path.join(IMG, "turing_aged16.jpg")).convert("L")
    w0, h0 = im.size
    side = min(w0, h0)
    im = im.crop(((w0 - side) // 2, int((h0 - side) * 0.25), (w0 - side) // 2 + side, int((h0 - side) * 0.25) + side)).resize((n, n), Image.LANCZOS)
    img = np.asarray(im, float) / 255
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    noise = np.clip(np.random.default_rng(2022).normal(0.5, 0.3, (n, n)), 0, 1)
    return img, noise


@lru_cache(None)
def math_titles():
    """Result-family titles of github.com/openai/math (Apache-2.0), parsed into ai_history/openai_math_families.json.
    Famous names (Riemann, Hilbert, BSD, Navier-Stokes, Kakeya, Hodge, Langlands, Erdős, Ramsey...) are drawn larger."""
    import re
    d = json.load(open(os.path.join(PROJ, "openai_math_families.json"), encoding="utf-8"))
    fams = [tuple(f) for f in d["families"]]
    famous = {i for i, (_, tt) in enumerate(fams) if re.search(
        r"Riemann|Hilbert|BSD|Birch|Navier|Kakeya|Hodge|Langlands|Goldbach|Erdős|Ramsey|De Giorgi|Catalan", tt)}
    return fams, famous


def render_frame(t):
    s_ = _render_frame(t)
    shake = 0.0
    for bt in BEATS:                                   # a small camera shake on the big breakthroughs
        if bt.get("major") and 0 <= t - bt["at"] < 0.45:
            shake = max(shake, (3 + 6 * min(1.0, bt["major"] / 150)) * (1 - (t - bt["at"]) / 0.45))
    if shake <= 0:
        return s_
    out = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c = cairo.Context(out)
    c.set_source_rgb(0, 0, 0); c.paint()
    c.set_source_surface(s_, shake * math.sin(t * 91), shake * math.cos(t * 77)); c.paint()
    return out


def _render_frame(t):
    T = story(t)
    ov = float(smooth((t - OVERVIEW[0]) / (OVERVIEW[1] - OVERVIEW[0])))
    cam = camera(T, ov)
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(s)
    pw = palette_weights(T)
    intro = float(smooth(t / 0.25))
    cd_on = CD_FROM <= T < GPT35 - 1e-4 and ov < 0.5 and not (STOP[0] <= t < STOP[1] + 0.2)

    ctx.set_source_surface(background("warm"), 0, 0); ctx.paint()
    if pw["mono"] > 0.01:
        ctx.set_source_surface(background("mono"), 0, 0); ctx.paint_with_alpha(pw["mono"])
    if pw["cold"] > 0.01:
        ctx.set_source_surface(background("cold"), 0, 0); ctx.paint_with_alpha(pw["cold"])
    palname = "cold" if pw["cold"] > 0.5 else ("mono" if pw["mono"] > 0.5 else "warm")
    for a, b, name, op, bl in ERAS:
        k = float(smooth((T - a) / 1.5) * (1 - smooth((T - b) / 1.5))) * (1 - ov)
        if k > 0.01:
            ctx.set_source_surface(era_surface(name, op, bl, palname), 0, 0); ctx.paint_with_alpha(k)
    light = 1 - 0.78 * winter_amount(T)
    draw_formulas(ctx, t, ov)

    # snow in the winters
    if pw["cold"] > 0.02:
        rng = np.random.default_rng(11)
        n = 220
        x0, y0 = rng.uniform(0, W, n), rng.uniform(0, H, n)
        sp, sz, dr = rng.uniform(25, 70, n), rng.choice([1.2, 1.8, 2.8], n, p=[0.6, 0.3, 0.1]), rng.uniform(0, 6.28, n)
        for x, y, r_ in zip(x0 + 14 * np.sin(t * 0.6 + dr), (y0 + sp * t) % H, sz):
            ctx.set_source_rgba(0.91, 0.93, 0.97, 0.55 * pw["cold"]); ctx.arc(x, y, r_, 0, 6.283); ctx.fill()

    # speed lines: from ChatGPT on, time visibly runs faster
    spd = float(smooth((T - 2022.93) / 2.5)) * (1 - ov)
    if spd > 0.01:
        rng = np.random.default_rng(3)
        n = int(10 + 50 * spd)
        ys_, ln_, v_, ph_ = rng.uniform(0.1 * H, 0.9 * H, 60), rng.uniform(60, 320, 60), rng.uniform(900, 3200, 60), rng.uniform(0, W + 400, 60)
        for i in range(n):
            x = (ph_[i] - v_[i] * (0.4 + spd) * t) % (W + 400) - 200
            ctx.set_source_rgba(1, 0.93, 0.78, 0.10 + 0.18 * spd); ctx.set_line_width(1.2)
            ctx.move_to(x, ys_[i]); ctx.line_to(x + ln_[i] * (0.5 + spd), ys_[i]); ctx.stroke()

    # the line
    has_line = T > 1943.2
    if has_line:
        ts = sample_ts(cam, 1943.0, 380)
        xs, ys = sx(cam, ts), line_y(cam, ts)
        q = 3
        gs = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // q, H // q)
        gc = cairo.Context(gs); gc.scale(1 / q, 1 / q)
        gc.set_source_rgba(1, 1, 1, light); gc.set_line_width(10)
        gc.move_to(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            gc.line_to(x, y)
        gc.stroke()
        A = surf_alpha(gs)
        G = np.clip(0.55 * gaussian_filter(A, 2.0) + 0.45 * gaussian_filter(A, 7.0) * 1.6, 0, 1) * intro
        gsurf = surf_from(pal_color("glow", T), G)
        ctx.save(); ctx.scale(q, q); ctx.set_source_surface(gsurf, 0, 0); ctx.get_source().set_filter(cairo.FILTER_BILINEAR); ctx.paint(); ctx.restore()
        base = mix(hx("#f2c98a"), np.array([0.86, 0.85, 0.83]), 0.6 * pw["mono"])
        for j in range(0, len(ts) - 1, 12):
            e = min(len(ts) - 1, j + 13)
            tm = ts[(j + e) // 2]
            age = float(np.clip(1 - (T - tm) / 40, 0.3, 1))
            c = mix(base, hx(COLD_HEX), 0.8 * winter_amount(tm))
            ctx.set_source_rgba(*c, (0.5 + 0.5 * age) * intro); ctx.set_line_width(3.4)
            ctx.move_to(xs[j], ys[j])
            for x, y in zip(xs[j + 1:e + 1], ys[j + 1:e + 1]):
                ctx.line_to(x, y)
            ctx.stroke()
        for r in data()["rows"]:                       # every verified milestone is a dot on the line
            if r["w"] < 4 or r["t"] > T or r["t"] < 1943:
                continue
            ctx.set_source_rgba(*pal_color("core", T), 0.8 * intro)
            ctx.arc(float(sx(cam, r["t"])), float(line_y(cam, r["t"])), 2.0 + 0.3 * max(r["w"], 0), 0, 6.283); ctx.fill()
        hxp, hyp = float(xs[-1]), float(ys[-1])
    else:
        hxp, hyp = cam["x_left"], cam["y_base"]

    # the head
    boost = 0.0
    for bt in BEATS:
        if not bt.get("setback") and 0 <= t - bt["at"] < 1.2:
            boost = max(boost, 1 - (t - bt["at"]) / 1.2)
    gl = glow_surface("#ffd978" if pw["cold"] < 0.5 else "#cfe0f2")
    sc = 0.9 + 0.8 * boost
    pulse = 0.75 + 0.25 * math.sin(t * 3.0) if not has_line else 1.0
    ctx.save(); ctx.translate(hxp, hyp); ctx.scale(sc, sc); ctx.translate(-256, -256)
    ctx.set_source_surface(gl, 0, 0); ctx.paint_with_alpha(min(1.0, (0.35 + 0.65 * light + 0.5 * boost) * intro * pulse)); ctx.restore()
    ctx.set_source_surface(streak_surface("#ffd978"), hxp - 450, hyp - 12); ctx.paint_with_alpha(light * intro)
    ctx.set_source_rgba(1, 0.98, 0.94, intro); ctx.arc(hxp, hyp, 7 + 3 * boost, 0, 6.283); ctx.fill()
    if boost > 0:
        ctx.set_source_rgba(1, 0.92, 0.75, 0.8 * boost); ctx.set_line_width(2.5)
        ctx.arc(hxp, hyp, 20 + 160 * (1 - boost), 0, 6.283); ctx.stroke()
    for bt in BEATS:                                   # big breakthroughs: particles of light burst out of the head
        dtb = t - bt["at"]
        if not bt.get("major") or not (0 <= dtb < 1.8):
            continue
        rng = np.random.default_rng(int(bt["at"] * 10))
        mj = bt["major"]
        n = int(min(170, 40 + 0.65 * mj))
        ang, spd, size = rng.uniform(0, 6.283, n), rng.uniform(260, 900, n) * (0.6 + 0.5 * min(1.0, mj / 150)), rng.uniform(0.10, 0.24, n)
        f = 1 - math.exp(-dtb * 2.6)                   # fast out, then slowing
        fade = 1 - float(smooth(dtb / 1.8))
        gl_p = glow_surface("#ffe2a6")
        for a_, v_, sz in zip(ang, spd, size):
            px, py = hxp + math.cos(a_) * v_ * f * 0.6, hyp + math.sin(a_) * v_ * f * 0.6 + 120 * dtb * dtb
            ctx.save(); ctx.translate(px, py); ctx.scale(sz, sz); ctx.translate(-256, -256)
            ctx.set_source_surface(gl_p, 0, 0); ctx.paint_with_alpha(0.9 * fade); ctx.restore()
            ctx.set_source_rgba(1, 0.97, 0.88, fade); ctx.arc(px, py, 2.0 + 12 * sz, 0, 6.283); ctx.fill()
        ctx.set_source_rgba(1, 0.9, 0.7, 0.7 * fade); ctx.set_line_width(4)
        ctx.arc(hxp, hyp, 30 + min(820, 160 + 2.6 * mj) * f, 0, 6.283); ctx.stroke()
        if dtb < 0.12:                                 # a short white flash
            ctx.set_source_rgba(1, 0.96, 0.86, (0.1 + 0.2 * min(1.0, mj / 150)) * (1 - dtb / 0.12)); ctx.paint()

    # echoes: when a node fires, the earlier seeds it fulfils light up and a spark travels from each to the new node,
    # landing at the moment of the burst
    for bt in BEATS:
        if not bt.get("echo") or not has_line or ov > 0.5:
            continue
        for k_, (ty0, lab0) in enumerate(bt["echo"]):
            de = t - (bt["at"] - 1.3) - 0.12 * k_
            if not (0 <= de < 5.2):
                continue
            x0, y0 = float(sx(cam, ty0)), float(line_y(cam, ty0))
            fade = 1 - float(smooth((de - 4.2) / 1.0))
            lit = float(smooth(de / 0.3)) * fade
            ctx.save(); ctx.translate(x0, y0); sc_ = 0.42 + 0.08 * math.sin(de * 6); ctx.scale(sc_, sc_); ctx.translate(-256, -256)
            ctx.set_source_surface(glow_surface("#ffd978"), 0, 0); ctx.paint_with_alpha(0.95 * lit); ctx.restore()
            rr_ = 10 + 34 * ((de * 1.1) % 1.0)
            ctx.set_source_rgba(*hx("#f2c14e"), (1 - ((de * 1.1) % 1.0)) * lit); ctx.set_line_width(2)
            ctx.arc(x0, y0, rr_, 0, 6.283); ctx.stroke()
            ctx.set_source_rgba(1, 0.97, 0.88, lit); ctx.arc(x0, y0, 6, 0, 6.283); ctx.fill()
            text(ctx, lab0, x0, y0 - 28 - 32 * (k_ % 2), 28, hx("#f2c14e"), lit, align="center")
            cxp, cyp = (x0 + hxp) / 2, min(y0, hyp) - 0.25 * abs(hxp - x0) - 60
            bez = lambda u: ((1 - u) ** 2 * x0 + 2 * (1 - u) * u * cxp + u * u * hxp, (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cyp + u * u * hyp)
            u_p = float(smooth((de - 0.3) / (1.0 - 0.12 * k_)))
            if u_p > 0:
                ctx.set_source_rgba(*hx("#f2c14e"), 0.55 * fade); ctx.set_line_width(2.0)
                for j in range(int(60 * u_p) + 1):
                    ctx.line_to(*bez(j / 60 * u_p)) if j else ctx.move_to(*bez(0))
                ctx.stroke()
            if 0 < u_p < 1:                            # the spark and its tail
                for j in range(8):
                    px, py = bez(max(0.0, u_p - 0.012 * j))
                    sc_ = 0.3 * (1 - j / 8)
                    ctx.save(); ctx.translate(px, py); ctx.scale(sc_, sc_); ctx.translate(-256, -256)
                    ctx.set_source_surface(glow_surface("#ffe7b0"), 0, 0); ctx.paint_with_alpha(0.9 * (1 - j / 8)); ctx.restore()
                px, py = bez(u_p)
                ctx.set_source_rgba(1, 0.98, 0.92, 1.0); ctx.arc(px, py, 5, 0, 6.283); ctx.fill()

    # keepers in the winter dark
    if pw["cold"] > 0.05:
        rng = np.random.default_rng(5)
        for i in range(7):
            x, y = rng.uniform(120, W - 120), rng.uniform(H * 0.25, H * 0.75)
            fl = 0.5 + 0.5 * math.sin(t * rng.uniform(0.8, 1.6) + i)
            ctx.save(); ctx.translate(x, y); ctx.scale(0.35, 0.35); ctx.translate(-256, -256)
            ctx.set_source_surface(glow_surface("#f3cf8e"), 0, 0); ctx.paint_with_alpha(pw["cold"] * (0.35 + 0.45 * fl)); ctx.restore()

    # bottom time axis: the same squeeze as the line; its unit shrinks from years to months to days
    ax_y = cam["y_base"] + 78
    ruler_a = intro * (0.0 if not has_line else 1.0)
    if ruler_a > 0:
        ctx.set_source_rgba(*pal_color("cream", T), 0.28 * ruler_a); ctx.set_line_width(1)
        ctx.move_to(cam["x_left"], ax_y); ctx.line_to(min(W - 20, hxp + 20), ax_y); ctx.stroke()
        for x, lab, lvl in ruler_ticks(cam):
            ctx.set_source_rgba(*pal_color("cream", T), 0.5 * ruler_a); ctx.set_line_width(1)
            ctx.move_to(x, ax_y); ctx.line_to(x, ax_y + (9 if lvl < 2 else 6)); ctx.stroke()
            text(ctx, lab, x, ax_y + 36, 22 if lvl < 3 else 20, pal_color("cream", T), (0.62 if lvl else 0.8) * ruler_a, align="center")

    # the info window (top left): title, drawing / number / card, plain explanation; at the node it flies into the head
    for bt in BEATS:
        if not (bt["open"] - 0.2 <= t <= bt["close"] + 0.4) or ov > 0.5:
            continue
        a_txt = float(smooth((t - bt["open"] - 0.3) / 0.4)) * (1 - float(smooth((t - bt["close"]) / 0.3)))   # old out, then new in
        y_end = WIN["desc_y"] + 190
        if a_txt > 0.01:
            y_end = max(y_end, draw_window_text(ctx, bt, a_txt * (1 - ov), T))
        col = tuple(mix(hx("#f2c98a"), pal_color("cream", T), 0.6))
        win = bt.get("win")
        setb = bool(bt.get("setback"))
        if win and win[0] == "viz":
            kind, src, frozen = win[1], win[2], win[3]
            life = max(0.0, t - bt["open"])           # the window starts fading in 0.2 s early; its clock starts at 0
            tt_, p_ = (src["lead"] + 0.8, 1.0) if frozen else (life, life / bt["lead"])
            vcol = tuple(mix(np.array(col), hx(COLD_HEX), 0.85)) if setb else col
            vs = cairo.ImageSurface(cairo.FORMAT_ARGB32, VS, VS)
            vc = cairo.Context(vs)
            viz_draw(kind, vc, p_, tt_, vcol)
            halo = surf_from(np.array(vcol), np.clip(gaussian_filter(surf_alpha(vs), 5) * 1.2, 0, 0.6))
            k0 = WIN["viz_s"] / VS
            cx0, cy0 = WIN["x"] + WIN["w"] / 2, WIN["viz_y"] + WIN["viz_s"] / 2
            a_win = float(smooth(life / 0.6)) * (1 - float(smooth((t - bt["close"]) / 0.3))) * (1 - ov)
            flies = not frozen and not setb
            fly = float(smooth((t - (bt["at"] - INJECT)) / INJECT)) if flies else 0.0
            dim = (float(smooth(fly / 0.2)) if t <= bt["at"] else 1 - float(smooth((t - bt["at"] - 0.2) / 0.6))) if flies else 0.0
            wa = a_win * (1 - 0.92 * dim) * (0.7 if frozen else 1.0)   # the drawing stays in the window until the next node
            ctx.save(); ctx.translate(cx0 - VS * k0 / 2, cy0 - VS * k0 / 2); ctx.scale(k0, k0)
            ctx.set_source_surface(halo, 0, 0); ctx.paint_with_alpha(0.8 * wa)
            ctx.set_source_surface(vs, 0, 0); ctx.paint_with_alpha(wa); ctx.restore()
            if flies and fly > 0 and t <= bt["at"]:   # a copy shrinks and flies into the head
                u = fly
                cxp, cyp = mix(cx0, hxp, 0.5), min(cy0, hyp) - 60
                cx = (1 - u) ** 2 * cx0 + 2 * (1 - u) * u * cxp + u * u * hxp
                cy = (1 - u) ** 2 * cy0 + 2 * (1 - u) * u * cyp + u * u * hyp
                k = k0 * (1 - 0.94 * u)
                va = a_win * (1 - 0.6 * u)
                ctx.save(); ctx.translate(cx - VS * k / 2, cy - VS * k / 2); ctx.scale(k, k)
                ctx.set_source_surface(halo, 0, 0); ctx.paint_with_alpha(0.8 * va)
                ctx.set_source_surface(vs, 0, 0); ctx.paint_with_alpha(va); ctx.restore()
        elif win and win[0] == "img":
            ps = photo_surface(win[1], max_w=420, max_h=330, mono=bt["year"] < 2012.7 or setb, cold=setb)
            pw_, ph_ = ps.get_width(), ps.get_height()
            ctx.set_source_surface(ps, WIN["x"] + WIN["w"] / 2 - pw_ / 2, WIN["viz_y"] + WIN["viz_s"] / 2 - ph_ / 2)
            ctx.paint_with_alpha(a_txt * (1 - ov) * (0.8 if win[3] else 1.0))
        elif bt.get("readout"):
            big, small = bt["readout"]
            a = a_txt * (1 - ov)
            gs_, gw_, gh_, gb_ = glow_text(big, 140, "#f2c14e", True)
            if gw_ > 560:                              # "12% → 74%": shrink to the window
                gs_, gw_, gh_, gb_ = glow_text(big, int(140 * 540 / gw_), "#f2c14e", True)
            ctx.set_source_surface(gs_, WIN["x"] - 20, WIN["viz_y"] + 30); ctx.paint_with_alpha(a)
            text(ctx, small, WIN["x"] + 14, WIN["viz_y"] + 30 + gh_ - 30, 34, pal_color("cream", T), a)
        if bt.get("cat") and a_txt > 0.01:            # the field (or 超越人类), as a tab on the window's top edge
            gold = hx("#f2c14e")
            ctx.select_font_face(SERIF, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); ctx.set_font_size(24)
            e = ctx.text_extents(bt["cat"])
            bx, by, bw_, bh_ = WIN["x"] + 14, WIN["y"] - 17, e.x_advance + 22, 34
            ctx.set_source_rgba(0.06, 0.05, 0.04, 0.95 * a_txt * (1 - ov)); ctx.rectangle(bx, by, bw_, bh_); ctx.fill()
            ctx.set_source_rgba(*gold, 0.9 * a_txt * (1 - ov)); ctx.set_line_width(1.5); ctx.rectangle(bx, by, bw_, bh_); ctx.stroke()
            text(ctx, bt["cat"], bx + 11, by + 25, 24, gold, a_txt * (1 - ov), bold=True)
        if a_txt > 0.01:                               # a hairline frame keeps the window's area readable
            ctx.set_source_rgba(*pal_color("cream", T), 0.16 * a_txt * (1 - ov)); ctx.set_line_width(1.2)
            ctx.rectangle(WIN["x"], WIN["y"], WIN["w"], y_end - WIN["y"]); ctx.stroke()

    # node labels and pictures
    lab_boxes = []                                     # (x left, x right, baseline) of node names, so the count can step around them
    for bt in BEATS:
        dt = t - bt["at"]
        if has_line and -0.1 < dt < bt["close"] - bt["at"] + 2.0 and ov < 0.5:
            a = float(smooth(dt / 0.5)) * (1 - float(smooth((t - bt["close"] - 1.0) / 0.8))) * (1 - ov)
            x, y = float(sx(cam, min(bt["year"], T))), float(line_y(cam, min(bt["year"], T)))
            if bt.get("setback"):
                text(ctx, bt["label"], x, y + 72, 34, hx(COLD_HEX), a * 0.95, align="center")
            else:
                lx = min(x + 30, W - 40)
                e_ = text(ctx, bt["label"], lx, y + 54, 34, pal_color("cream", T), a, align="left" if x + 30 + 300 < W else "right")
                if a > 0.05:
                    lab_boxes.append((lx if x + 30 + 300 < W else lx - e_.x_advance, (lx + e_.x_advance) if x + 30 + 300 < W else lx, y + 54))
        if has_img(bt.get("image")) and not bt.get("card") and -0.1 < dt < 2.2 * BAR and ov < 0.5:
            a = float(smooth(dt / 0.6) * (1 - smooth((dt - 1.7 * BAR) / (0.5 * BAR))))
            if t > bt["close"]:
                a *= 1 - float(smooth((t - bt["close"]) / 0.4))
            pr_, py0_, pmw_, pmh_ = LAYOUT["photo"]
            ps = photo_surface(bt["image"], max_w=pmw_, max_h=pmh_, mono=bt["year"] < 2012.7)
            pw_, ph_ = ps.get_width(), ps.get_height()
            ctx.set_source_surface(ps, W - pr_ - pw_, py0_ + (pmh_ - ph_) / 2 - 4 * dt + (92 if cd_on else 0)); ctx.paint_with_alpha(a * (1 - ov))

    # kept labels: key nodes stay named on the line once their moment has passed (small, skipped if crowded)
    cnt_y = hyp + 76                                   # the list steps below any node name in its way
    for xl_, xr_, yl_ in sorted(lab_boxes, key=lambda b: b[2]):
        if xr_ > min(hxp + 40, W - 380) - 10 and cnt_y - 50 < yl_ < cnt_y + 50:
            cnt_y = yl_ + 56
    fl = flash_layout(t, hxp, hyp, cnt_y + 52) if has_line and ov < 0.5 else []
    fl_box = (min(hxp + 40, W - 380), fl[0][1] - 30, W, fl[-1][1] + fl[-1][3]) if fl else None
    if has_line and ov < 0.5:
        taken = []
        current = [bt["year"] for bt in BEATS if bt["at"] - 0.1 < t < bt["close"] + 1.5]
        for ty, name in reversed(KEEP):
            if ty > T - 0.02 or any(abs(ty - c) < 0.05 for c in current):
                continue
            x, y = float(sx(cam, ty)), float(line_y(cam, ty))
            if x < cam["x_left"] + 10 or any(abs(x - px) < 110 and abs(y - py) < 46 for px, py in taken):
                continue
            if fl_box and fl_box[0] - 120 < x < fl_box[2] and fl_box[1] - 40 < y < fl_box[3]:
                continue
            taken.append((x, y))
            cold = name.endswith("寒冬")
            text(ctx, name, x + 10, y + 34, 24, hx(COLD_HEX) if cold else pal_color("cream", T), 0.55 * (1 - ov))

    # 请选择你的派别: Turing and von Neumann side by side
    for bt in BEATS:
        if bt.get("faction") and 0.3 <= t - bt["at"] < 4.2 and ov < 0.5:
            a = float(smooth((t - bt["at"] - 0.3) / 0.4)) * (1 - float(smooth((t - bt["at"] - 3.6) / 0.6)))
            gs_, gw_, gh_, gb_ = glow_text("请选择你的派别", 52, "#f2c14e", True)
            fy1, fy2 = LAYOUT["faction_y"]
            ctx.set_source_surface(gs_, W - 26 - 470 / 2 - gw_ / 2, fy2 - gb_); ctx.paint_with_alpha(a)
            text(ctx, "图灵", W - 26 - 470 * 0.75, fy1, 30, pal_color("cream", T), a, align="center")
            text(ctx, "冯·诺依曼", W - 26 - 470 * 0.25, fy1, 30, pal_color("cream", T), a, align="center")

    # flashes: the smaller breakthroughs pop into a list beside the head, more and more often
    if has_line and ov < 0.5:
        for f, y, a, h_ in flash_layout(t, hxp, hyp, cnt_y + 52):
            x0 = min(hxp + 40, W - 380)
            age = t - f["t"]
            gold = hx("#f2c14e")
            text(ctx, f["dl"], x0, y, 22, gold, 0.9 * a, font=LATIN)
            xn = x0 + (62 if len(f["dl"]) <= 5 else 80)
            nc = mix(gold, pal_color("cream", T), float(smooth((age - 0.2) / 0.6)))
            text(ctx, f["name"], xn, y, 28, nc, a, bold=True)
            xt = x0
            if f["tag"]:                               # 超越人类 / 追平人类 in gold, 新闻 / 论文 in cream
                tc = pal_color("cream", T) if f["tag"] in ("新闻", "论文") else gold
                e = text(ctx, f["tag"], x0 + 6, y + 34, 20, tc, a, bold=True)
                ctx.set_source_rgba(*tc, 0.8 * a); ctx.set_line_width(1.4)
                ctx.rectangle(x0, y + 14, e.x_advance + 12, 27); ctx.stroke()
                xt = x0 + e.x_advance + 22
            if f["text"]:
                text(ctx, f["text"], xt, y + 34, 24, pal_color("cream", T), 0.75 * a)
            for k_, (ys_, lab_) in enumerate(FLASH_SRC.get(f["name"], ())):   # its foundations spark in
                ag = age - 0.06 * k_
                if not (0 <= ag < 1.1):
                    continue
                fade = 1 - float(smooth((ag - 0.8) / 0.3))
                x0s, y0s = float(sx(cam, ys_)), float(line_y(cam, ys_))
                x1s, y1s = float(sx(cam, f["year"])), float(line_y(cam, f["year"]))
                cxs, cys = (x0s + x1s) / 2, min(y0s, y1s) - 0.2 * abs(x1s - x0s) - 40
                bz = lambda u: ((1 - u) ** 2 * x0s + 2 * (1 - u) * u * cxs + u * u * x1s, (1 - u) ** 2 * y0s + 2 * (1 - u) * u * cys + u * u * y1s)
                ctx.save(); ctx.translate(x0s, y0s); ctx.scale(0.26, 0.26); ctx.translate(-256, -256)
                ctx.set_source_surface(glow_surface("#ffd978"), 0, 0); ctx.paint_with_alpha(0.85 * fade); ctx.restore()
                text(ctx, lab_, x0s, y0s - 22 - 26 * (k_ % 2), 22, gold, 0.85 * fade, align="center")
                u_ = float(smooth(ag / 0.7))
                ctx.set_source_rgba(*gold, 0.28 * fade); ctx.set_line_width(1.4)
                for j in range(31):
                    ctx.line_to(*bz(j / 30 * u_)) if j else ctx.move_to(*bz(0))
                ctx.stroke()
                if 0 < u_ < 1:
                    px_, py_ = bz(u_)
                    ctx.save(); ctx.translate(px_, py_); ctx.scale(0.18, 0.18); ctx.translate(-256, -256)
                    ctx.set_source_surface(glow_surface("#ffe7b0"), 0, 0); ctx.paint_with_alpha(0.9); ctx.restore()
                    ctx.set_source_rgba(1, 0.98, 0.92, 1.0); ctx.arc(px_, py_, 3.5, 0, 6.283); ctx.fill()
            if age < 0.7:                              # a small ring on the line where it happened
                px, py = float(sx(cam, f["year"])), float(line_y(cam, f["year"]))
                ctx.set_source_rgba(*(gold if f["tag"] else pal_color("core", T)), 0.9 * (1 - age / 0.7)); ctx.set_line_width(2)
                ctx.arc(px, py, 6 + 44 * float(smooth(age / 0.7)), 0, 6.283); ctx.stroke()

    # the clock (top right): year -> year.month -> date
    yr, size = clock_str(T)
    gsur, gw, gh, gb = glow_text(yr, size, "#efe3c6" if pw["cold"] < 0.5 else "#c9d3df")
    cy_ = LAYOUT["clock_y"]
    ctx.set_source_surface(gsur, W - 40 - gw + int(size * 0.5), cy_ - gb); ctx.paint_with_alpha(intro)
    unit = "单位：年" if T < 2023 else ("单位：月" if T < 2026 else "单位：日")
    switch = [(2023.0, "进入「月」为单位"), (2026.0, "进入「日」为单位")]
    hint = None
    for ty, msg in switch:
        tt0 = snap(float(np.interp(ty, [k[1] for k in KEYS if k[0] < 306 - 15.442], [k[0] for k in KEYS if k[0] < 306 - 15.442])), half=False)
        if 0 <= t - tt0 < 3.6:
            hint = (msg, float(smooth((t - tt0) / 0.4)) * (1 - float(smooth((t - tt0 - 3.0) / 0.6))))
    if hint:
        gs_, gw_, gh_, gb_ = glow_text(hint[0], 40, "#f2c14e", True)
        ctx.set_source_surface(gs_, W - 30 - gw_ + 20, cy_ + 50 - gb_ + 10); ctx.paint_with_alpha(hint[1] * (1 - ov))
    elif cd_on:                                        # counting down to GPT-3.5
        ca = float(smooth((T - CD_FROM) / 0.15))
        text(ctx, "距离 GPT-3.5 上线", W - 40, cy_ + 48, 26, pal_color("cream", T), 0.8 * ca, align="right")
        gs_, gw_, gh_, gb_ = glow_text(f"还有 {countdown_days(T)} 天", 46, "#f2c14e", True)
        ctx.set_source_surface(gs_, W - 40 - gw_ + 23, cy_ + 104 - gb_); ctx.paint_with_alpha(ca)
    elif has_line and T > 2022.9:
        text(ctx, unit, W - 40, cy_ + 50, 26, pal_color("cream", T), 0.55 * intro * (1 - ov), align="right")

    # opening: what the line means
    if t < 3.4:                                        # one short line, then straight into the story
        a = float(smooth(t / 0.3)) * (1 - float(smooth((t - 2.8) / 0.5)))
        gs_, gw_, gh_, gb_ = glow_text("这条线，就是 AI 的 70 年", 56, "#efe3c6", True)
        ctx.set_source_surface(gs_, (W - gw_) / 2, 0.62 * H - gb_); ctx.paint_with_alpha(a)
    text(ctx, "AI 发展", cam["x_left"] - 40, cam["y_base"] - 300, 24, pal_color("cream", T), 0.4 * intro * (1 if has_line else 0))

    # the stop before ChatGPT
    if STOP[0] <= t < STOP[1] + 0.2:
        k = float(smooth((t - STOP[0]) / 0.6)) * (1 - float(smooth((t - STOP[1] + 0.1) / 0.3)))
        ctx.set_source_rgba(0, 0, 0, 0.6 * k); ctx.paint()
        gs_, gw_, gh_, gb_ = glow_text("时间，开始加速了。", 66, "#efe3c6", True)
        a = float(smooth((t - STOP[0] - 0.5) / 0.6)) * (1 - float(smooth((t - STOP[1] + 0.25) / 0.25)))
        ctx.set_source_surface(gs_, (W - gw_) / 2, 0.36 * H - gb_); ctx.paint_with_alpha(a)
        n_ = countdown_days(T)                         # the countdown, big, while time holds its breath
        gs_, gw_, gh_, gb_ = glow_text(f"距离 GPT-3.5 上线，还有 {n_} 天" if n_ > 0 else "GPT-3.5 上线", 52, "#f2c14e", True)
        ctx.set_source_surface(gs_, (W - gw_) / 2, 0.36 * H + 120 - gb_); ctx.paint_with_alpha(a)

    # the wall: the real result titles of OpenAI's maths collection pour in, then the head breaks through them
    if WALL[0] - 1.5 <= t < EXCUSE[1]:
        fams, famous = math_titles()
        rng = np.random.default_rng(719)
        n_all = len(fams)
        px_, py_ = rng.uniform(-200, W - 80, n_all), rng.uniform(80, H - 260, n_all)
        appear = np.sort(rng.uniform(0, 1, n_all)) ** 0.8
        order = rng.permutation(n_all)
        fill = (t - (WALL[0] - 1.5)) / (WALL[1] - WALL[0] + 1.5)
        burst = max(0.0, t - WALL[1])
        for k in range(n_all):
            if appear[k] > fill:
                break
            num, title = fams[order[k]]
            big = order[k] in famous
            x, y = px_[k], py_[k]
            if burst > 0:
                dx, dy = x - hxp, y - hyp
                dd = math.hypot(dx, dy) + 1e-3
                push = 900 * burst * burst + 380 * burst
                x, y = x + dx / dd * push, y + dy / dd * push
            age = fill - appear[k]
            a = min(1.0, age * 8) * (1 - float(smooth(burst / 1.4)))
            if a <= 0.01:
                continue
            size = (38 if big else 22) * (1.6 - 0.6 * min(1.0, age * 6))      # flies in from close, settles
            col = hx("#f2c14e") if big else pal_color("cream", T)
            text(ctx, f"{num}  {title}" if not big else title, x, y, size, col, a * (0.95 if big else 0.6), font=LATIN)
        for j, eq in enumerate(("∂u/∂t + (u·∇)u = −∇p + νΔu + f", "x ∈ ℚ", "L(E, s)", "C^{1,α}", "p − 1", "ℓ¹")):
            a = float(smooth((fill - 0.2 - 0.12 * j) * 5)) * (1 - float(smooth(burst / 1.4)))
            if a > 0.01:
                ctx.select_font_face("STIX Two Math", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                ctx.set_font_size(54 if j == 0 else 64)
                ex, ey = (60 + (j * 331) % int(0.7 * W), int(0.2 * H) + (j * 523) % int(0.52 * H))
                if burst > 0:
                    dx, dy = ex - hxp, ey - hyp; dd = math.hypot(dx, dy) + 1e-3; push = 900 * burst * burst + 380 * burst
                    ex, ey = ex + dx / dd * push, ey + dy / dd * push
                ctx.set_source_rgba(*hx("#f6d99c"), 0.85 * a); ctx.move_to(ex, ey); ctx.show_text(eq)
        if burst > 0:
            ctx.save(); ctx.translate(hxp, hyp); sc = 1.2 + 4 * min(1.0, burst); ctx.scale(sc, sc); ctx.translate(-256, -256)
            ctx.set_source_surface(glow_surface("#ffe7b0"), 0, 0); ctx.paint_with_alpha(max(0.0, 1 - burst / 2.5)); ctx.restore()
    if EXCUSE[0] <= t < EXCUSE[1] + 0.5:
        a = float(smooth((t - EXCUSE[0] - 0.3) / 0.5)) * (1 - float(smooth((t - EXCUSE[1]) / 0.5)))
        gs_, gw_, gh_, gb_ = glow_text("借过一下。", 120, "#fff4d8", True)
        ctx.set_source_surface(gs_, (W - gw_) / 2, 0.5 * H - gb_ + 40); ctx.paint_with_alpha(a)

    # the overview: every important date, labelled along the whole line, then hold and fade
    if ov > 0.01 and has_line:
        boxes = [(0, cam["y_base"] + 40, W, H)]        # every label is a box; the time axis is off limits

        def free(bx):
            return all(bx[2] + 6 < o[0] or o[2] + 6 < bx[0] or bx[3] + 4 < o[1] or o[3] + 4 < bx[1] for o in boxes)
        for k, (ty, dl, name) in enumerate(LANDMARKS):
            a = float(smooth((t - OVERVIEW[0] - 1.2 - 0.17 * k) / 0.5))
            if a <= 0:
                continue
            x, y = float(sx(cam, ty)), float(line_y(cam, ty))
            ctx.set_source_rgba(1, 0.97, 0.9, a); ctx.arc(x, y, 5, 0, 6.283); ctx.fill()
            dyx = abs((float(line_y(cam, ty)) - float(line_y(cam, ty - 0.05))) / (float(sx(cam, ty)) - float(sx(cam, ty - 0.05)) + 1e-6))
            wn = 25 * len(name) * (0.62 if name.isascii() else 1.0)
            wd = 21 * len(dl) * 0.55
            cands = []
            if dyx <= 1.2:                             # flat part: stacked above the point
                for tier in range(16):
                    ly = y - 64 - 62 * tier
                    w_ = max(wn, wd) + 10
                    cands.append(("up", ly, (x - w_ / 2, ly - 48, x + w_ / 2, ly + 8)))
            for row in range(10):                      # steep part (or no room above): to the left, stacked down
                ly = y + 8 + 36 * row
                w_ = wn + wd + 24
                cands.append(("left", ly, (x - 30 - w_, ly - 26, x - 30, ly + 6)))
            for kind, ly, bx in cands:
                if free(bx):
                    break
            else:                                      # nothing free: the spot with the least overlap
                def over(bx):
                    return sum(max(0, min(bx[2], o[2]) - max(bx[0], o[0])) * max(0, min(bx[3], o[3]) - max(bx[1], o[1])) for o in boxes)
                kind, ly, bx = min(cands, key=lambda c: over(c[2]))
            boxes.append(bx)
            ctx.set_source_rgba(*pal_color("cream", T), 0.4 * a); ctx.set_line_width(1)
            if kind == "up":
                ctx.move_to(x, y - 6); ctx.line_to(x, ly + 8); ctx.stroke()
                text(ctx, dl, x, ly - 26, 21, hx("#f2c14e"), a, align="center")
                text(ctx, name, x, ly + 2, 25, pal_color("cream", T), a, align="center")
            else:
                ctx.move_to(x - 6, y); ctx.line_to(x - 26, ly - 8); ctx.stroke()
                text(ctx, name, x - 30, ly, 25, pal_color("cream", T), a, align="right")
                text(ctx, dl, x - 30 - wn - 10, ly, 21, hx("#f2c14e"), a, align="right")
    if ov > 0.5 and has_line and t >= OVERVIEW[1]:   # 1943 -> GPT-3.5: 79 years; GPT-3.5 -> today: 1418 days
        a1 = float(smooth((t - OVERVIEW[1]) / 0.4))                       # on the beat after the pull-back
        a2 = float(smooth((t - snap(OVERVIEW[1] + BEAT, half=False)) / 0.4))
        yb = ax_y + 84
        x0b, xkb, xtb = float(sx(cam, 1943.95)), float(sx(cam, 2022.915)), float(sx(cam, STORY_END))
        cream, gold = pal_color("cream", T), hx("#f2c14e")
        def brace(xa, xb, col, a):
            ctx.set_source_rgba(*col, a); ctx.set_line_width(2)
            ctx.move_to(xa, yb - 12); ctx.line_to(xa, yb); ctx.line_to(xb, yb); ctx.line_to(xb, yb - 12); ctx.stroke()
        brace(x0b, xkb - 6, cream, 0.7 * a1)
        text(ctx, "1943 → GPT-3.5：79 年", (x0b + xkb) / 2, yb + 46, 32, cream, 0.9 * a1, align="center")
        if a2 > 0:
            ts_ = np.linspace(2022.915, STORY_END, 160)  # the last stretch of the line, picked out in gold
            xs_, ys_ = sx(cam, ts_), line_y(cam, ts_)
            ctx.set_source_rgba(*gold, 0.9 * a2); ctx.set_line_width(7)
            ctx.move_to(float(xs_[0]), float(ys_[0]))
            for x_, y_ in zip(xs_[1:], ys_[1:]):
                ctx.line_to(float(x_), float(y_))
            ctx.stroke()
            brace(xkb, xtb, gold, a2)
            gs_, gw_, gh_, gb_ = glow_text("GPT-3.5 → 今天：1418 天", 50, "#f2c14e", True)
            ctx.set_source_surface(gs_, min(xtb + 27, W - 20) - gw_, yb + 130 - gb_); ctx.paint_with_alpha(a2)
    if t >= CLOSING_T0:
        cx0, cy0, cdy = LAYOUT["closing"]
        if LAYOUT["closing_panel"]:
            ap = float(smooth((t - CLOSING_T0) / 0.6))
            ctx.set_source_rgba(0.035, 0.03, 0.025, 0.72 * ap)
            ctx.rectangle(cx0 - 40, cy0 - 60, 900, cdy * len(CLOSING) + 118); ctx.fill()
        for k, (lhs, rhs) in enumerate(CLOSING):
            a = float(smooth((t - CLOSING_TS[k]) / 0.35))
            if a <= 0:
                continue
            y = cy0 + cdy * k
            e = text(ctx, lhs + "：", cx0, y, 34, pal_color("cream", T), a)
            gs_, gw_, gh_, gb_ = glow_text(rhs, 44, "#f2c14e", True)
            ctx.set_source_surface(gs_, cx0 + e.x_advance - 22, y - gb_ + 6); ctx.paint_with_alpha(a)
        if t >= QUESTION_T:
            a = float(smooth((t - QUESTION_T) / 0.6))
            gs_, gw_, gh_, gb_ = glow_text("下一次，还要多久？", 62, "#fff4d8", True)
            ctx.set_source_surface(gs_, cx0 - 31, cy0 + cdy * len(CLOSING) + 70 - gb_); ctx.paint_with_alpha(a)
    if t >= FADE[0]:
        ctx.set_source_rgba(0, 0, 0, float(smooth((t - FADE[0]) / (FADE[1] - FADE[0])))); ctx.paint()
    return s


# ======================================================================================
# CLI
# ======================================================================================
def frame_bytes(t):
    s = render_frame(t)
    s.flush()
    return bytes(s.get_data())


def cmd_snap(times):
    d = os.path.join(OUT, "snaps" if ORIENT == "portrait" else "snaps_landscape")
    os.makedirs(d, exist_ok=True)
    thumbs = []
    for t in times:
        s = render_frame(t)
        p = os.path.join(d, f"t{t:07.2f}.png")
        s.write_to_png(p)
        im = Image.open(p).convert("RGB")
        im.thumbnail((360, 640) if ORIENT == "portrait" else (640, 360))
        thumbs.append((t, im))
        print(p)
    if len(thumbs) > 1:
        cols = min(6, len(thumbs))
        rows = (len(thumbs) + cols - 1) // cols
        cw, ch = (370, 670) if ORIENT == "portrait" else (650, 390)
        sheet = Image.new("RGB", (cols * cw, rows * ch), (25, 25, 25))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(sheet)
        for i, (t, im) in enumerate(thumbs):
            x, y = (i % cols) * cw + 5, (i // cols) * ch + 5
            sheet.paste(im, (x, y))
            dr.text((x + 4, y + ch - 26), f"{t:.2f}s  ({story(t):.2f})", fill=(230, 230, 230))
        sheet.save(os.path.join(d, "contact.png"))
        print(os.path.join(d, "contact.png"))


def _render_chunk(args):
    a, b, path = args
    n0, n1 = int(round(a * FPS)), int(round(b * FPS))
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for n in range(n0, n1):
        p.stdin.write(frame_bytes(n / FPS))
    p.stdin.close()
    p.wait()
    return path


def cmd_render(a, b, jobs, name):
    from multiprocessing import Pool
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    step = max(2.0, (b - a) / jobs)
    edges = list(np.arange(a, b, step)) + [b]
    tasks = [(edges[i], edges[i + 1], os.path.join(OUT, "chunks", f"{name}_{i:03d}.mp4")) for i in range(len(edges) - 1)]
    with Pool(jobs) as pool:
        parts = pool.map(_render_chunk, tasks)
    lst = os.path.join(OUT, "chunks", f"{name}.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    video = os.path.join(OUT, f"{name}_video.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video], check=True)
    final = os.path.join(OUT, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-ss", str(a), "-t", str(b - a),
                    "-stream_loop", "1", "-i", os.path.join(BGM, BGM_FILE), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", f"afade=t=out:st={max(0.0, b - a - 3.0):.2f}:d=3", "-c:a", "aac", "-b:a", "192k", "-t", str(b - a), final], check=True)
    print(final)


def cmd_check():
    d = data()
    db = downbeats()
    print("rows", len(d["rows"]), "downbeats", len(db))
    for bt in BEATS:
        near = min(db, key=lambda x: abs(x - bt["at"]))
        flag = "" if abs(near - bt["at"]) < 0.02 else "  <-- off the downbeat"
        need = 1.0 + sum(1 for ch in bt["title"] + bt["desc"] if "\u4e00" <= ch <= "\u9fff") / 5
        show = bt["close"] - bt["open"]
        print(f"{bt['at']:8.3f} {bt['year']:9.3f} story {story(bt['at']):9.3f}  window {show:4.1f}s (read {need:4.1f}s)"
              f"{'  <-- short' if show < need - 0.3 else ''}  {bt['title']}{flag}")
    for t in (0, 17, 45, 63, 80, 95, 110, 121, 140, 160, 214, 260, 300, 310):
        T = story(t)
        print(f"t={t:5.1f}  T={T:8.3f}  I={float(I(T)):7.1f}  run_max={float(np.interp(T, GRID, d['run_max'])):7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("snap"); s1.add_argument("times", type=float, nargs="+")
    s2 = sub.add_parser("render"); s2.add_argument("a", type=float); s2.add_argument("b", type=float)
    s2.add_argument("--jobs", type=int, default=8); s2.add_argument("--name", default="preview")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "snap":
        cmd_snap(a.times)
    elif a.cmd == "render":
        cmd_render(a.a, a.b, a.jobs, a.name)
    else:
        cmd_check()
