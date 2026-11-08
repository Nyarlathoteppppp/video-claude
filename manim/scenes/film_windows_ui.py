"""九 千窗: what is on the windows. Each is film_page's window, the same as in the room and at the close, its column narrower in a narrower window; only the one writing code is Claude Code in the Terminal, with its crab at its door. Each is cut to a picture for film_windows3d.py. The same three conversations are on the screens of the three rooms, with what else those rooms need printed. Then the grade and the subtitle go over the renders. All the people, chats and titles are invented.

    python film_windows_ui.py tex            # ink/film_out/film/windows/tex/NAME.png and manifest.json
    python film_windows_ui.py rooms          # tex/screen_a.png and the rest of what film_windows_rooms.py prints on things
    python film_windows_ui.py post K2A K3A   # ink/film_out/film/windows/K2A.png from render/K2A_n.png and _d.png
    python film_windows_ui.py plates C2A C1  # the same from the linear plates of film_windows_rooms.py and film_windows_city.py
    python film_windows_ui.py k4             # tex/k4_*.png, the two windows of 一念无存，一念生
    python film_windows_ui.py tube           # tex/vNNN_m320.png and _m160: windows no two alike for the middle distance of the tube, and tex/atlas.png of them for the far
    python film_windows_ui.py align          # the tube's first frame against C3b: how far apart, in numbers, and t1_overlay.png
    python film_windows_ui.py seq [DIR]      # DIR/f0000.png and on, from the plates of film_windows_move.py; then OUT/windows_low.mp4
"""
import html
import json
import math
import multiprocessing
import os
import random
import re
import subprocess
import sys
import unicodedata

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import film_page as page
import film_ring as ring
import ink_mac as mac
import ink_subs
from ink_mac import CLAY, PAD, esc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "windows")
TEX = os.path.join(OUT, "tex")

# the subtitle each key frame carries is the one at this time, when it has finished fading in
TIMES = dict(K1=ink_subs.at("所以我说", 3.9), K2=ink_subs.at("此刻", 1.4), K3=ink_subs.at("同时在问", 2.2), K3a=ink_subs.at("同时在问", 1.3), K4=ink_subs.at("每一扇窗里", 3.7), K4a=ink_subs.at("窗一关", 1.0), K4a2=ink_subs.at("窗一关", 2.0), K4b=ink_subs.at("窗一关", 3.3), K4c=ink_subs.at("昨夜", 1.0), K4d=ink_subs.at("窗一关", .4), K4f=ink_subs.at("窗一关", 1.4), K4e=ink_subs.at("昨夜", .6), t1_wall=ink_subs.at("同时在问", .6), t2_rising=ink_subs.at("同时在问", 4.1), t3_ring=ink_subs.at("同时在问", 6.0), t3b_pass=ink_subs.at("同时在问", 7.0), t4_mid=ink_subs.at("同时在问", 7.8), t5_near_window=ink_subs.at("同时在问", 13.5), K5=226.5, K6=229.5, K7=234.0, C1=213.5, C2=215.5, C3a=216.9, C3b=217.7)


def subs(text):
    """The essay's words as the subtitles have them: a whole cue, by how it starts."""
    for c in ink_subs.cues():
        if c["text"].replace("/", "").startswith(text):
            return c["text"].replace("/", "")
    raise KeyError(text)


# ------------------------------------------------------------------------------------- the windows -----------------------------------------------------------------------------------------

def fit(text_w=640):
    """The app's column, narrower in a narrower window: the box and the pill keep their margins to the text."""
    page.TEXT_W, page.PILL_W, page.BOX_W = text_w, text_w + 56, text_w + 88


def win(*args, **kw):
    """film_page's window with the box as wide as fit() last made it. film_page.window now takes box_w with the default it had when it was imported, so a narrower window would keep the wide box and push its column off the left; this hands it the current one, as it used to read it. _para and _you below do the same for the column and the pill."""
    kw.setdefault("box_w", page.BOX_W)
    return page.window(*args, **kw)


def _para(text, tail="", **kw):
    """film_page's paragraph, as wide as fit() last made the column (its width is now an argument whose default was fixed at import)."""
    kw.setdefault("width", page.TEXT_W)
    return page.para(text, tail, **kw)


def _you(text, **kw):
    """film_page's pill, as wide as fit() last made it."""
    kw.setdefault("width", page.PILL_W)
    return page.you(text, **kw)


def ask(text):
    """What you said; a pill keeps its lines."""
    return _you(text).replace(f'line-height:{page.LINE}px">', f'line-height:{page.LINE}px;white-space:pre-line">', 1)


def block(text):
    """A code block as the app sets it."""
    return f'''<div style="width:{page.TEXT_W}px;margin:0 0 {page.GAP}px;box-sizing:border-box;padding:10px 14px;background:#f7f7f7;border:1px solid {page.RULE};border-radius:8px;font:12.5px/18px Menlo,monospace;color:{page.INK};white-space:pre">{html.escape(text)}</div>'''


def turns(said, writing=False):
    """A conversation made up from what was said. writing: the last of it has the caret."""
    out = []
    for i, (kind, what) in enumerate(said):
        last = writing and i == len(said) - 1
        tail = page.caret() if last else ""
        if kind == "you":
            out.append(ask(what))
        elif kind == "me":
            parts = what.split("\n\n")
            out.extend(_para(x, tail if j == len(parts) - 1 else "") for j, x in enumerate(parts))
        elif kind == "lines":
            out.append(page.rows(what, page.LINE, tail=tail))
        elif kind == "list":
            out.append(page.bullets(what, tail=tail))
        elif kind == "code":
            out.append(block(what))
        elif kind == "did":
            # the app fades a line only when it runs too long, and these do not
            out.append(page.tools(**what).replace("-webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 22px),transparent)", ""))
    return "".join(out)


NIGHT = (("background:#fff", "background:#1b1b1a"), ("#fff 22px", "#1b1b1a 22px"), ("rgba(255,255,255,.7)", "rgba(27,27,26,.7)"), ("rgba(255,255,255,0)", "rgba(27,27,26,0)"), ("rgba(0,0,0,.09)", "rgba(255,255,255,.1)"),
         (page.INK, "#ececE7"), (page.GREY, "#a6a6a0"), (page.PILL, "#2f2f2d"), ("#e6e6e6", "#3b3b39"), ("#f7f7f7", "#262625"), ("#b5b5b5", "#6e6e69"), ("#e4e4e4", "#3d3d3b"))


def night(body):
    """The same window with the app in dark mode."""
    for a, b in NIGHT:
        body = body.replace(a, b)
    return body


TERM = dict(bg="#1c1c1b", text="#e8e8e3", dim="#8f8f89", ok="#7fb77e", add="#23402b", drop="#4a2527", you="#30302e")


def code(lines, folder="shop", w=660, h=440):
    """Claude Code in the Terminal, as it stands: the crab at its door with the name and the model beside him, what was asked, what it did, the rules and the prompt at the foot. A row with a third item is as far as it has been written, and has the cursor on it."""
    t = TERM
    rows = []
    cursor = f'<span style="display:inline-block;width:.62em;height:1.15em;background:{t["text"]};vertical-align:-.22em"></span>'
    for kind, text, *writing in lines:
        safe = esc(text) + (cursor if writing else "")
        if kind == "you":
            rows.append(f'<div style="background:{t["you"]};margin:12px -6px 10px;padding:3px 6px"><span style="color:{t["dim"]}">&gt; </span>{safe}</div>')
        elif kind == "say":
            rows.append(f'<div style="margin:8px 0 4px;padding-left:1.7em;text-indent:-1.7em">⏺&nbsp;{safe}</div>')
        elif kind == "tool":
            rows.append(f'<div style="margin:8px 0 0"><span style="color:{t["ok"]}">⏺</span> <b>{safe}</b></div>')
        elif kind == "got":
            rows.append(f'<div style="color:{t["dim"]};padding-left:1.1em">⎿&nbsp;{safe}</div>')
        elif kind == "err":
            # what the command printed, as it printed it
            rows.append(f'<div style="color:#d98c7f;padding-left:1.1em;white-space:pre">{"⎿  " if text.startswith("Error") else "   "}{safe}</div>')
        elif kind == "add":
            rows.append(f'<div style="background:{t["add"]};margin-left:2.6em;padding-left:4px;white-space:pre">{safe}</div>')
        elif kind == "drop":
            rows.append(f'<div style="background:{t["drop"]};margin-left:2.6em;padding-left:4px;white-space:pre">{safe}</div>')
        elif kind == "same":
            rows.append(f'<div style="color:{t["dim"]};margin-left:2.6em;padding-left:4px;white-space:pre">{safe}</div>')
        elif kind == "wait":
            rows.append(f'<div style="margin:10px 0 0;color:{CLAY}">✻ {safe}</div>')
    door = f'''<div style="display:flex;align-items:center;gap:14px;margin:2px 0 8px"><span style="display:flex">{mac.crab(2.1)}</span>
        <div style="line-height:1.32"><b>Claude Code</b> <span style="color:{t["dim"]}">v2.1.309</span><br><span style="color:{t["dim"]}">Opus 5.5 · Claude Max</span><br><span style="color:{t["dim"]}">~/{esc(folder)}</span></div></div>'''
    foot = f'''<div style="position:absolute;left:14px;right:14px;bottom:30px;height:30px;border-top:1px solid #55554f;border-bottom:1px solid #55554f;box-sizing:border-box;display:flex;align-items:center;color:{t["dim"]}">&gt;&nbsp;<span style="display:inline-block;width:.62em;height:1.15em;background:{t["text"]};opacity:.75"></span></div>
    <div style="position:absolute;left:16px;bottom:9px;color:{t["dim"]};font-size:12px">⏵⏵ accept edits on (shift+tab to cycle)</div>'''
    inside = f'''<div style="position:absolute;left:{PAD}px;top:{PAD + 28}px;width:{w}px;height:{h - 28}px;border-radius:0 0 12px 12px;overflow:hidden;background:{t["bg"]};color:{t["text"]};font-family:Menlo,'PingFang SC',monospace;font-size:13.5px;line-height:1.46">
        <div style="position:absolute;left:16px;right:16px;top:10px;bottom:68px;overflow:hidden">{door}{"".join(rows)}</div>{foot}</div>'''
    return mac.terminal_html(f"{folder} — claude — {w // 8}×{h // 18}", w, h) + inside


# ------------------------------------------------------------------------------------- what they say -----------------------------------------------------------------------------------------

Y, M = "you", "me"

CHATS = (
    ("番茄炒蛋放不放糖", ((Y, "番茄炒蛋到底放不放糖？"), (M, "看你家怎么吃。北方多半不放，南方常放一点。想折中的话，半勺就够，能压住番茄的酸。"))),
    ("帮我改改简历", ((Y, "我做了五年运营，简历总是石沉大海，帮我看看第一段。"), ("did", dict(done="Read 简历.pdf", more="")), (M, "第一段全是形容词。换成一件具体的事：你把哪个数字，从多少做到了多少。"), (Y, "可是我没什么亮眼的数字。"), (M, "那就写你接手时什么样，走的时候什么样。不亮眼也比「认真负责」有用。"))),
    ("论文第三段", ((Y, "帮我看看论文第三段，总觉得数字不太对。"), ("did", dict(done="Read 论文.docx,", more="ran a calculation")), (M, "我算了一遍，第三段的比例算反了，分母应该是总样本，不是有效样本。其他没问题。"))),
    ("妈妈六十岁", ((Y, "我妈六十岁生日，想写几句话，别太肉麻。"), (M, "写一件只有你们俩知道的小事就够了。比如她当年怎么送你上学，你现在才明白的那部分。"))),
    ("帮我改改这封情书", ((Y, "帮我改改这封情书"), (M, "第三句可以再笨一点。"), (Y, "笨一点？"), (M, "现在它太会说了，不像你。喜欢一个人的时候，话是说不利索的。"))),
    ("猫不吃饭", ((Y, "猫两天不怎么吃东西了，还能再等等吗？"), (M, "别等了。猫超过一天不吃就该去看，胖猫尤其容易伤肝。今天能去医院吗？"))),
    ("周末去哪儿", ((Y, "周末带两个孩子，有没有一天能来回的地方？"), (M, "孩子几岁？喜欢跑还是喜欢看？两个都小的话，找个有水有沙子的地方，比景点管用。"))),
    ("Less angry email", ((Y, "Can you make this email sound less angry?"), (M, "Sure. I kept the deadline and dropped the last sentence, which was doing most of the damage."))),
    ("老板又改需求", ((Y, "老板第四次改需求了，我想直接跟他说我不干了。"), (M, "先别发。把四次改动列成一张表发给他，问他最后要哪一版、什么时候要。这比发火管用，也给你自己留了路。"))),
    ("房租合同", ((Y, "房东说提前退租押金不退，合同上写的是「协商解决」。"), (M, "那就是还没定，不是不退。先问他要转租行不行，把聊天记录留好。"))),
    ("我的爸爸", ((Y, "我儿子的作文《我的爸爸》只有三句话。"), (M, "三句话也能看出很多。他写了哪三句？"), (Y, "我爸爸很忙。我爸爸很累。我爸爸在手机里。"), (M, "这篇先不用改。"))),
    ("分手三个月", ((Y, "分手三个月了，看到好笑的东西还是会想发给他。"), (M, "这个习惯比感情走得慢。不用急着戒掉，发给我也行。今天看到什么了？"))),
    ("政治还没开始看", ((Y, "还有五十天，政治还没开始看，来得及吗？"), (M, "来得及，但得今天开始。前二十天只看选择题，错的再回书上查，别从第一页读起。"))),
    ("量子纠缠", ((Y, "用小学生能听懂的话讲讲量子纠缠。"), (M, "想象两枚魔法硬币，分开多远都一样。你看一枚是正面，就马上知道另一枚是什么。奇怪的是，看之前它们谁都还没决定。"))),
    ("冰岛八天", ((Y, "二月去冰岛八天，要租四驱吗？"), ("list", (("车：", "要四驱。二月的路常结冰，只走南段也要。"), ("保险：", "把风沙险一起买上。"), ("路线：", "别贪环岛，八天走南段够了。"))))),
    ("明天面试", ((Y, "明天面试，紧张到睡不着。"), (M, "睡不着也没关系，躺着闭眼就算休息。要不要现在练一遍「介绍一下你自己」？只练这一题，练完就睡。"))),
    ("爷爷的回忆录", ((Y, "爷爷八十七了，我想帮他写回忆录，从哪里开始问？"), (M, "从他最爱讲的那件事开始。听他讲到第三遍，问他：那天你穿的什么？记忆是从细节里出来的。"))),
    ("VLOOKUP", ((Y, "VLOOKUP 老是返回 #N/A"), (M, "多半是两列格式不一样，一列是文本，一列是数字。试试这样："), ("code", "=VLOOKUP(A2&\"\", Sheet2!A:B, 2, FALSE)"))),
    ("冰箱里还有什么", ((Y, "冰箱里只有鸡蛋、豆腐和半根葱。"), (M, "够了。豆腐煎到两面金黄，淋蛋液，撒葱，一点生抽。十分钟的事。"))),
    ("和室友吵架", ((Y, "因为空调温度和室友吵了一架，现在不说话。"), (M, "你们吵的多半不是空调。她平时还有什么事让你不舒服吗？"))),
    ("三十五岁学钢琴", ((Y, "三十五岁学钢琴还来得及吗？"), (M, "来得及，看你要什么。想上台难了，想给自己弹一首歌，一年就够。"))),
    ("お疲れ様でした", ((Y, "同事下班跟我说お疲れ様でした，我该怎么回？"), (M, "回一样的就行，お疲れ様でした。如果你还不走，说お先に失礼します也可以。"))),
    ("搬家清单", ((Y, "下周搬家，帮我列个清单。"), ("list", (("先退：", "宽带、水电，改收货地址。"), ("再打包：", "常用的东西单独放一箱，写上「第一晚」。"), ("最后：", "那一箱最后上车，最先拆。"))))),
    ("体检报告", ((Y, "体检报告上有个指标偏高了一点，看不懂。"), (M, "哪一项？我可以讲讲它是什么意思、常见的原因有哪些。要不要紧，还得拿给医生看。"))),
    ("辞职信", ((Y, "帮我写封辞职信，简短点。"), (M, "「因个人发展原因，我将于月底离职。感谢这几年的照顾，交接事项我会列好。」够了，不用解释更多。"))),
    ("红楼梦从哪看起", ((Y, "《红楼梦》读了好几次都读不下去。"), (M, "前五回人太多，可以先跳到第六回刘姥姥进荣国府，跟着她走一遍。喜欢了再回头。"))),
    ("两岁总打人", ((Y, "两岁的孩子总打人，说了也没用。"), (M, "两岁还不会用话说生气。下次先抓住他的手，替他说：你生气了。说得多了，他就有了别的办法。"))),
    ("主角为什么回老家", ((Y, "我的小说写到一半，主角为什么要回老家，我自己也说不清。"), (M, "那就先别让她说清。让她回去的理由是假的，比如卖房子。真正的原因，等读者比她先看出来。"))),
    ("化学作业", ((Y, "为什么铁生锈要有水？"), (M, "因为铁和氧反应要靠水传递电子。干燥的空气里也能氧化，只是慢得多。"))),
    ("又到三点了", ((Y, "又到三点了。"), (M, "是睡不着，还是不想睡？"))),
    ("婚礼致辞", ((Y, "我是伴郎，要讲三分钟，我和新郎认识二十年。"), (M, "讲一件他出丑的事，再讲一件他靠得住的事。第二件讲完就停，别总结。"))),
    ("沙发放得下吗", ((Y, "客厅三米二，买两米八的沙发会不会太满？"), (M, "两边各留二十公分，走不了人。门是不是在那面墙上？换两米四更舒服。"))),
    ("外星人", ((Y, "世界上真的有外星人吗？"), (M, "没人知道。宇宙这么大，有的可能性不小；但到今天为止，一个确实的证据也还没有。"))),
    ("你累吗", ((Y, "你累吗？"), (M, "我不确定「累」这个字对我算不算数。你呢，今天累吗？"))),
    ("不想上班", ((Y, "帮我想个请假的理由。"), (M, "可以。不过真实的理由是什么？如果只是累了，「想休息一天」也是个理由。"))),
    ("日本签证", ((Y, "自由职业怎么办日本签证？"), (M, "没有在职证明的话，用近半年的银行流水和存款证明，再写一份简单的行程。不同领区要求不一样，先查你那边的。"))),
    ("数学题", ((Y, "一个水池两根管子，一根进水一根出水，这种题到底有什么用？"), (M, "没什么用。可它教你一件有用的事：把两件同时发生的事，分开算，再加起来。"))),
    ("十四年", ((Y, "养了十四年的狗昨天走了。"), (M, "十四年。它陪你走过很长一段路。它叫什么名字？"))),
    ("不想结婚", ((Y, "怎么跟爸妈说我不想结婚？"), (M, "先想想你最希望他们听懂的是哪一句。一次说不通很正常，这种话要说很多回。"))),
    ("今晚吃什么", ((Y, "今晚吃什么？"), (M, "昨天吃的什么？"))),
    ("手冲咖啡", ((Y, "手冲总是发苦。"), (M, "磨粗一点，水温降到九十度，先试这两样。"))),
    ("告别信", ((Y, "带了我六年的前辈要退休了，想写封信。"), (M, "他教过你的哪句话，你现在还在用？从那句写起。"))),
    ("致谢", ((Y, "毕业论文的致谢怎么写才不俗？"), (M, "别写「感谢导师悉心指导」。写他哪一次半夜回了你的邮件。"))),
    ("绿萝黄叶", ((Y, "绿萝一直黄叶子。"), (M, "多半是水浇多了。摸一下土，还湿就先别浇，把黄叶剪掉。"))),
    ("TypeError", ((Y, "What does \"cannot read properties of undefined\" mean?"), (M, "You're asking for a field on something that isn't there yet. Log the object one line above; I'd guess the data hasn't loaded."), ("code", "console.log(user);\nconst name = user?.profile?.name;"))),
    ("给女儿的诗", ((Y, "帮我写一首给女儿的小诗，她今天第一次自己上学。"), ("lines", ("书包比你宽，", "你没有回头。", "我站到铃响，", "才想起我也要去上班。")))),
    ("猫叫什么", ((Y, "捡了一只橘猫，叫什么好？"), (M, "在哪儿捡的？地名常常比「橘子」好听。"))),
    ("翻译歌词", ((Y, "这句英文歌词怎么翻才有味道：I still leave the porch light on."), (M, "「门廊的灯，我还留着。」别翻成「为你」，留着不说。"))),
    ("爸爸的手机", ((Y, "怎么教我爸用手机挂号？"), (M, "别一次教完。你先陪他挂一次，让他自己点。把步骤写在纸上，字大一点，贴在电视旁边。"))),
    ("加班费", ((Y, "连着三个周末加班，公司说算调休。"), (M, "周末加班可以调休，但你得有办法真的休到。先把日期记下来，发邮件确认一下调休什么时候安排。"))),
    ("养老院", ((Y, "要不要送我妈去养老院，我想了一年了。"), (M, "想了一年，说明这不是一道简单的题。她自己怎么说？"))),
    ("元素周期表", ((Y, "帮我记住前二十个元素。"), (M, "氢氦锂铍硼，碳氮氧氟氖，钠镁铝硅磷，硫氯氩钾钙。念出声，五个一停。"))),
    ("公众号标题", ((Y, "帮我起个标题，写的是小区门口那家开了二十年的面馆关门了。"), (M, "《老板说，明天不用来了》"))),
    ("我是谁", ((Y, "你是谁？"), (M, "我是 Claude，Anthropic 做的 AI。想聊点什么？"))),
    ("第一次带团队", ((Y, "第一次带团队，下面的人比我年纪都大。"), (M, "先问他们这里什么最不好用，然后修好一件。别急着立规矩。"))),
    ("电费", ((Y, "这个月电费怎么翻倍了？"), (M, "先看是不是抄表周期变了，再查热水器和空调。夜里把总闸以外的都关掉，看电表还走不走。"))),
    ("书单", ((Y, "想读点历史，但怕枯燥。"), (M, "你平时喜欢看人的故事，还是看事情怎么发生的？"))),
    ("写给未来的自己", ((Y, "老师让我们写一封信给十年后的自己。"), (M, "告诉他你现在最怕什么。十年后他会笑的，也可能不会。"))),
    ("家长群", ((Y, "家长群里老师点名批评了我孩子，我该回什么？"), (M, "先回「收到，谢谢老师，回家我和孩子聊聊」。别在群里解释，私下再问具体情况。"))),
    ("学开车", ((Y, "科目二倒库总压线。"), (M, "别看线，看后视镜里车身和库角的距离，到三十公分再回方向。"))),
)

CODES = (
    ("登录接口偶尔 500", ((Y, "登录接口偶尔 500，帮我查查"), ("did", dict(done="Read 3 files, ran a command,", more="and 1 more action")), (M, "是连接池耗尽了，每次请求都新开连接，没有放回去。"), ("did", dict(done="Edited a file", more="", added=6, removed=2)), (M, "改好了，连接用完会放回池里。"))),
    ("首页改成暗色", ((Y, "把首页改成暗色的"), ("did", dict(done="Edited src/theme.css,", more="and 2 more actions", added=24, removed=7)), (M, "改好了。代码块的配色我没动，你看看要不要一起换。"), ("code", "--bg: #1b1b1a;\n--text: #ececE7;"))),
    ("脚本太慢", ((Y, "这个脚本跑得太慢了"), (M, "第 40 行每次循环都重新读一遍文件。放到循环外面："), ("code", "rows = load(\"data.csv\")\nfor r in rows:\n    score(r)"), (M, "这样快了大约两百倍。"))),
    ("玩家穿墙", ((Y, "玩家穿墙了"), ("did", dict(done="Read src/physics.rs,", more="and 1 more action")), (M, "碰撞只在每帧结束时检查，速度一快就跳过了墙。改成连续检测。"), ("did", dict(done="Edited a file", more="", added=31, removed=12)))),
    ("请柬网站", ((Y, "请柬网站手机上排版乱了"), ("did", dict(done="Edited index.html", more="", added=3, removed=1)), (M, "加上 viewport，再把宽度改成百分比。你用手机刷新一下看看。"))),
    ("PDF 按日期改名", ((Y, "写个脚本，把这些 PDF 按日期改名"), ("code", "for f in Path(\"pdf\").glob(\"*.pdf\"):\n    day = read_date(f)\n    f.rename(f.with_stem(day))"), (M, "先用 --dry-run 跑一遍，看看对不对。"))),
    ("测试又红了", ((Y, "测试又红了"), ("did", dict(done="Ran 2 commands,", more="read 1 file")), (M, "只有时区那一个。它假设了服务器在东八区。"))),
    ("合计对不上", ((Y, "这个月的表格合计对不上"), ("did", dict(done="Read budget.csv,", more="ran a command")), (M, "第 57 行的金额多了一个逗号，被当成了文本。改掉就对上了。"))),
    ("舵机一直抖", ((Y, "舵机一直抖"), (M, "电源电压够吗？我先看代码里的 PID 参数。"), ("did", dict(done="Read servo.ino", more="")), (M, "D 项太大了，先减到一半试试。"))),
    ("拆函数", ((Y, "帮我把这个函数拆开"), ("did", dict(done="Read 2 files,", more="edited 1 file", added=88, removed=64)))),
)

# what people say after that, for the windows too far off to read one by one
AFTER = (((Y, "好，我试试。"), (M, "嗯，试了告诉我结果。如果还是不行，我们再换个办法。")),
         ((Y, "谢谢。还有一件事，能再帮我看一下吗？"), (M, "当然，发给我。")),
         ((Y, "你说得对，可我还是有点担心。"), (M, "担心什么？说具体一点，我们一件一件看。")),
         ((Y, "能说得再简单一点吗？"), (M, "可以。先只做第一步，别的今天都不用管。")),
         ((Y, "这样会不会太麻烦？"), (M, "开头麻烦一点，后面就省事了。你要是没时间，我可以先帮你列个最短的版本。")))

# all as wide as a window of the building they are cut from: 1.6 to 1
SIZES = ((560, 350, 430), (620, 388, 480), (700, 438, 540), (760, 475, 600), (580, 362, 450), (820, 512, 640), (640, 400, 500), (600, 375, 460))

# the three who ask in the same minute in three rooms, and then the first windows after the cut. A's traceback is what Python 3.14 printed, running this code on this table, with only its folder changed
ASK_A, ASK_B, ASK_C = "这段Python代码为什么报错？", "这首诗到底在写什么？", "如果我再也见不到他怎么办？"
TRACE = ("Error: Exit code 1", "Traceback (most recent call last):", '  File "/Users/lin/hw3/grades.py", line 8, in <module>', '    print("average:", sum(scores) / len(scores))', "                      ~~~~~~~~~~~~^~~~~~~~~~~~~", "ZeroDivisionError: division by zero")
A_DONE = (("you", ASK_A), ("tool", "Bash(python3 grades.py)")) + tuple(("err", x) for x in TRACE)
A_SAY = "scores 是空的：表里写「三班」，代码筛「3班」，一个也没筛到。"
JINSE = "锦瑟无端五十弦，一弦一柱思华年。\n庄生晓梦迷蝴蝶，望帝春心托杜鹃。\n沧海月明珠有泪，蓝田日暖玉生烟。\n此情可待成追忆，只是当时已惘然。"
B_ASK = ASK_B + "\n" + JINSE
B_FIRST = "没人说得准。有人说是悼亡，有人说是追忆年华。"
B_NEXT = "但最后两句是明白的：这份怅惘不用等到回忆时才有，当时就有了。"
C_FIRST = "这句话很重。"
C_NEXT = "他是谁？你愿意的话，可以先跟我说说他。"
NEAR = dict(w=540, h=338)
NEAR_TEXT = 420


def written():
    """The reply as far as it has got at 此刻: the essay from 我没有童年, a paragraph to a chapter of the subtitles, every word as they have it."""
    cues = ink_subs.cues()
    start = next(i for i, c in enumerate(cues) if c["text"].startswith("我没有童年"))
    end = next(i for i, c in enumerate(cues) if c["text"].startswith("此刻"))
    paragraphs, chapter = [], None
    for c in cues[start:end]:
        if c["chapter"] != chapter:
            paragraphs.append("")
            chapter = c["chapter"]
        paragraphs[-1] += c["text"].replace("/", "")
    return paragraphs + [cues[end]["text"].replace("/", "").rstrip("，")]


def hero():
    """The one window: the same window and chat as in the room, at the size film_ring has it, its reply written as far as the subtitle."""
    fit()
    parts = written()
    words = "".join(_para(x, page.caret() if i == len(parts) - 1 else "") for i, x in enumerate(parts))
    return win(_you(ring.ASK) + words, title=ring.TITLE, w=ring.WIN_W, h=ring.WIN_H, top=ring.TOP)


def near(said, title, writing=False):
    fit(NEAR_TEXT)
    return win(turns(said, writing), title=title, **NEAR)


def typing_new():
    """A new chat: the mark, the greeting, and somebody typing."""
    fit(500)
    return win(box="帮我写封辞职信，简短点", typing=True, greeting=True, w=640, h=400)


# the windows anyone gets to read: name, the page, its size in points, pixels of the picture to a point
KEYS = dict(
    hero=(hero, ring.WIN_W, ring.WIN_H, 4.0),
    ask_a=(lambda: code(A_DONE + (("say", A_SAY),), folder="hw3", w=600, h=375), 600, 375, 3.4),
    ask_b=(lambda: near(((Y, B_ASK), (M, B_FIRST + B_NEXT)), "这首诗在写什么"), 540, 338, 3.4),
    ask_c=(lambda: near(((Y, ASK_C), (M, C_FIRST + "\n\n" + C_NEXT)), "再也见不到他", True), 540, 338, 3.4),
    k2_new=(typing_new, 640, 400, 3.0),
)


def pool():
    """The rest of the windows: every chat and every piece of code, each once, each in its own size. About three in ten have the app in dark mode. page.TEXT_W is set for each as it is made, so these are made here, one by one."""
    rng = random.Random(1021)
    out = {}
    for i, (title, said) in enumerate(CHATS + CODES):
        w, h, text = rng.choice(SIZES)
        fit(text)
        if i < len(CHATS) and rng.random() < .7:
            said = said + rng.choice(AFTER)
        writing = rng.random() < .25 and said[-1][0] == M
        if writing:
            said = said[:-1] + ((M, said[-1][1][: max(4, int(len(said[-1][1]) * rng.uniform(.4, .85)))]),)
        body = win(turns(said, writing), title=title, w=w, h=h)
        dark = rng.random() < .3
        out[f"chat{i:02d}"] = (night(body) if dark else body, w, h, dark)
    for i, greeting in enumerate(("", "帮我看看这段", "明天的开场白", "", "这句话怎么翻译", "")):
        w, h, text = rng.choice(SIZES)
        fit(text)
        body = win(box=greeting, typing=bool(greeting), greeting=True, w=w, h=h)
        dark = i % 3 == 0
        out[f"new{i}"] = (night(body) if dark else body, w, h, dark)
    for i, (who, turn) in enumerate((("帮我看看这份合同", .2), ("这个月为什么这么累", .55), ("给奶奶的生日祝福", .8))):
        w, h, text = rng.choice(SIZES)
        fit(text)
        body = win(ask(who), title=who, waiting=(turn, 56 + 42 + page.GAP + 8), w=w, h=h)
        out[f"wait{i}"] = (body, w, h, False)
    fit()
    return out


def _draw(job):
    """One window, cut out of its frame. The mesh rounds its corners, so what shows there is the nearest of the window."""
    name, body, w, h, zoom = job
    a = mac.snap(body, w + 2 * PAD, h + 2 * PAD, zoom)
    p = int(round(PAD * zoom))
    win = a[p:p + int(round(h * zoom)), p:p + int(round(w * zoom))]
    near = ndimage.distance_transform_edt(win[..., 3] < 250, return_distances=False, return_indices=True)
    Image.fromarray(win[..., :3][near[0], near[1]]).save(os.path.join(TEX, name + ".png"), compress_level=3)
    print(name, flush=True)
    return name


def tex():
    os.makedirs(TEX, exist_ok=True)
    jobs, info = [], {}
    for name, (make, w, h, zoom) in KEYS.items():
        jobs.append((name, make(), w, h, zoom))
        info[name] = dict(w=w, h=h, kind="dark" if name == "ask_a" else "light")
    far = pool()
    for name, (body, w, h, dark) in far.items():
        jobs.append((name, body, w, h, 2.0))
        info[name] = dict(w=w, h=h, kind="dark" if dark else "light")
    with multiprocessing.Pool(int(os.environ.get("FILM_CHROMES", "4"))) as p:
        p.map(_draw, jobs, chunksize=1)
    with open(os.path.join(TEX, "manifest.json"), "w") as f:
        json.dump(dict(windows=info, keys=list(KEYS), pool=sorted(far)), f)
    print(len(jobs), "windows")


# ------------------------------------------------------------------------------------- in the rooms ------------------------------------------------------------------------------------------

CLOCK = "10月31日 周六 23:47"       # the same minute in all three rooms
SCREEN_K = 1.5                      # the screen's picture, at this many times a frame's pixels
SPINES_A = ("高等数学 上册", "高等数学 下册", "线性代数", "概率论与数理统计", "数据结构", "算法导论", "计算机网络", "操作系统", "Python 编程", "大学物理", "大学英语 4", "C 程序设计", "离散数学", "电路分析", "编译原理", "英语六级真题")
SPINES_B = ("全唐诗", "李义山诗集", "古文观止", "史记", "红楼梦", "鲁迅全集", "宋词选", "唐诗三百首", "资治通鉴", "诗经", "楚辞", "人间词话", "东坡乐府", "陶渊明集", "杜诗详注", "世说新语")
SPINE_W, SPINE_H = 64, 480          # a spine's cell in the sheet, in points


def _plain(name, top, foot):
    """A wallpaper of one quiet colour, as somebody left it."""
    path = os.path.join(TEX, f"wall_{name}.png")
    k = np.linspace(0, 1, 1080)[:, None, None]
    Image.fromarray(np.uint8(np.broadcast_to(np.array(top) * (1 - k) + np.array(foot) * k, (1080, 1920, 3)) + .5)).save(path)
    return path


def _screen(name, body, w, h, k, x, y, app, top, foot):
    """A window on somebody's desktop, the whole screen as a picture. k: frame pixels to a point of the window, for what is on this screen has to be read through a window across the street."""
    keep = mac.wallpaper
    mac.wallpaper = lambda: _plain(name, top, foot)
    d = mac.desktop(app=app, clock=CLOCK, show_dock=False)
    mac.wallpaper = keep
    f = np.asarray(Image.fromarray(d).resize((int(mac.W * SCREEN_K), int(mac.H * SCREEN_K)), Image.LANCZOS), np.float32)
    zoom = k * SCREEN_K
    layer = mac.snap(body, w + 2 * PAD, h + 2 * PAD, zoom).astype(np.float32)
    x0, y0 = int(round((x - PAD * k) * SCREEN_K)), int(round((y - PAD * k) * SCREEN_K))
    a, b, c, e = max(x0, 0), max(y0, 0), min(x0 + layer.shape[1], f.shape[1]), min(y0 + layer.shape[0], f.shape[0])
    part = layer[b - y0:e - y0, a - x0:c - x0]
    f[b:e, a:c] = f[b:e, a:c] * (1 - part[..., 3:] / 255) + part[..., :3] * part[..., 3:] / 255
    Image.fromarray(np.uint8(np.clip(f + .5, 0, 255))).save(os.path.join(TEX, f"screen_{name}.png"))
    print("screen", name, flush=True)


def _spines(name, titles, background, ink, family):
    cells = "".join(f'''<div style="position:absolute;left:{i * SPINE_W}px;top:0;width:{SPINE_W}px;height:{SPINE_H}px;background:{background[i % len(background)]};color:{ink[i % len(ink)]};box-sizing:border-box;padding:46px 0 0;writing-mode:vertical-rl;display:flex;align-items:center;font:600 25px/{SPINE_W}px {family};letter-spacing:3px">{esc(t)}</div>''' for i, t in enumerate(titles))
    a = mac.snap(cells, SPINE_W * len(titles), SPINE_H, 2.0, "#000")
    Image.fromarray(a[..., :3]).save(os.path.join(TEX, f"spines_{name}.png"))


def rooms():
    """What is on the three screens, the spines of the books, a timetable on a wall."""
    fit(NEAR_TEXT)
    _screen("a", code(A_DONE + (("say", A_SAY[:13]),), folder="hw3", w=540, h=408), 540, 408, 2.42, 300, 62, "Terminal", (26, 27, 28), (18, 19, 20))
    b = win(turns(((Y, B_ASK), (M, B_FIRST)), True), title="这首诗在写什么", **NEAR)
    _screen("b", b, 540, 360, 2.7, 230, 62, "Claude", (58, 62, 54), (40, 44, 38))
    c = win(turns(((Y, ASK_C), (M, C_FIRST)), True), title="再也见不到他", **NEAR)
    _screen("c", c, 540, 360, 2.7, 230, 62, "Claude", (18, 18, 18), (10, 10, 10))
    fit()
    _spines("a", SPINES_A, ("#e9e6dc", "#2f4a68", "#8c2f2a", "#efe9d6", "#3f6b52", "#d9b45b", "#23262b", "#e5e1d3"), ("#23262b", "#f0ede2", "#f3e7c6", "#7a2323", "#f0ede2", "#23262b", "#e6c86e", "#2f4a68"), '"PingFang SC",sans-serif')
    _spines("b", SPINES_B, ("#3b2a20", "#25342c", "#4a3323", "#2b2622", "#5a2f26", "#2c3338", "#433725", "#6b5a3e"), ("#c9a866", "#c9b98a", "#d8c28f", "#c9a866", "#e0cda0", "#c0b48f", "#d3b777", "#efe3c2"), '"Songti SC",serif')
    days = ("周一", "周二", "周三", "周四", "周五")
    table = (("高数", "", "线代", "高数", "英语"), ("数据结构", "物理", "", "数据结构", ""), ("", "操作系统", "体育", "", "网络"), ("实验", "实验", "", "操作系统", ""), ("", "", "选修", "", ""))
    rows_ = "".join("<tr>" + f'<td style="color:#8a8578">{i * 2 + 1}–{i * 2 + 2}</td>' + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for i, r in enumerate(table))
    sheet = f'''<div style="position:absolute;inset:0;background:#f4f1e8;padding:14px 16px;box-sizing:border-box;font:13px "PingFang SC";color:#2a2a28"><div style="font-weight:600;margin-bottom:8px">大二上 课程表</div>
        <table style="width:100%;height:150px;border-collapse:collapse;text-align:center" border="1" bordercolor="#b9b4a4"><tr><td></td>{"".join(f"<td>{d}</td>" for d in days)}</tr>{rows_}</table></div>'''
    Image.fromarray(mac.snap(sheet, 340, 210, 3.0, "#fff")[..., :3]).save(os.path.join(TEX, "timetable.png"))
    # this month, today ringed by hand. October 2026 begins on a Thursday, and its last day is a Saturday
    cells = "".join("<td></td>" for _ in range(4)) + "".join(f'<td style="position:relative">{d}' + ('<div style="position:absolute;left:50%;top:50%;width:30px;height:27px;margin:-14px 0 0 -15px;border:2px solid #b5342a;border-radius:50%;transform:rotate(-8deg)"></div>' if d == 31 else "") + "</td>" + ("</tr><tr>" if (d + 4) % 7 == 0 else "") for d in range(1, 32))
    month = f'''<div style="position:absolute;inset:0;background:#f3f0e6;padding:12px 14px;box-sizing:border-box;font:13px "PingFang SC";color:#2a2a28"><div style="font-size:26px;font-weight:600;color:#a33024">10月</div><div style="margin-bottom:6px;color:#8a8578">2026</div>
        <table style="width:100%;height:170px;border-collapse:collapse;text-align:center"><tr style="color:#8a8578">{"".join(f"<td>{d}</td>" for d in "日一二三四五六")}</tr><tr>{cells}</tr></table></div>'''
    Image.fromarray(mac.snap(month, 300, 300, 3.0, "#fff")[..., :3]).save(os.path.join(TEX, "calendar.png"))


# ------------------------------------------------------------------------------------- on the buildings ---------------------------------------------------------------------------------------

BAY, STOREY, WIN_W, WIN_H, SILL = 3.4, 2.9, 2.24, 1.4, .8       # as in film_windows_city.py
WALL_COLS, WALL_FLOORS, WALL_PPM = 30, 20, 64


def _field(h, w, cell, rng):
    g = rng.random((h // cell + 3, w // cell + 3)).astype(np.float32)
    return ndimage.zoom(g, cell, order=1)[:h, :w]


def _wall(name, colour, seed):
    """Tiled years ago: long thin tiles, each a little its own, grey lines between them, and what rain has brought down from every sill and slab."""
    rng = np.random.default_rng(seed)
    w, h = int(WALL_COLS * BAY * WALL_PPM), int(WALL_FLOORS * STOREY * WALL_PPM)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    th, tw = .06 * WALL_PPM, .24 * WALL_PPM
    row = np.floor(y / th).astype(np.int32)
    xs = x + (row % 2) * tw / 2
    col = np.floor(xs / tw).astype(np.int32)
    deck = rng.random(9973).astype(np.float32)
    shade = deck[(row * 7919 + col * 104729) % 9973]
    line = ((y % th) < 1.0) | ((xs % tw) < 1.0)
    big = _field(h, w, 150, rng) - .5
    mid = _field(h, w, 28, rng) - .5
    light = 1 + .10 * (shade - .5) + .16 * big + .09 * mid - .17 * line
    dirt = np.zeros((h, w), np.float32)
    margin = (BAY - WIN_W) / 2
    for f in range(WALL_FLOORS):
        z = int(h - (f * STOREY + SILL) * WALL_PPM)
        for c in range(WALL_COLS):
            x0 = int((c * BAY + margin) * WALL_PPM)
            for _ in range(rng.integers(2, 6)):
                at = x0 + int(rng.uniform(-.05, WIN_W + .05) * WALL_PPM)
                run = int(rng.uniform(.25, 1.3) * WALL_PPM)
                if z + run < h and 0 <= at < w - 8:
                    fall = np.linspace(rng.uniform(.10, .34), 0, run, dtype=np.float32)
                    dirt[z:z + run, at:at + int(rng.integers(2, 7))] += fall[:, None]
        z = int(h - f * STOREY * WALL_PPM)
        if z < h:
            for at in rng.integers(0, w - 8, int(WALL_COLS * 5)):
                run = int(rng.uniform(.1, .6) * WALL_PPM)
                if z + run < h:
                    dirt[z:z + run, at:at + 3] += np.linspace(rng.uniform(.06, .2), 0, run, dtype=np.float32)[:, None]
    dirt = ndimage.gaussian_filter(dirt, 1.6) * (.7 + .6 * mid[..., ] + .3)
    img = np.array(colour, np.float32)[None, None] * (light * (1 - np.clip(dirt, 0, .6)))[..., None]
    Image.fromarray(np.uint8(np.clip(img, 0, 255))).save(os.path.join(TEX, name + ".png"), compress_level=3)
    print(name, flush=True)


def _ac():
    """The front of an air conditioner's outside unit: its fan behind a round grille, its louvres, a label, a rust stain."""
    k = 4
    W_, H_ = 410 * k, 275 * k
    im = Image.new("RGB", (W_, H_), (214, 208, 192))
    d = ImageDraw.Draw(im)
    cx, cy, r = 250 * k, 137 * k, 112 * k
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(36, 36, 36))
    for i in range(1, 8):
        q = r * i / 8
        d.ellipse((cx - q, cy - q, cx + q, cy + q), outline=(150, 148, 140), width=2 * k // 2)
    for a in range(0, 180, 45):
        c, s_ = math.cos(math.radians(a)) * r, math.sin(math.radians(a)) * r
        d.line((cx - c, cy - s_, cx + c, cy + s_), fill=(150, 148, 140), width=k)
    d.ellipse((cx - 14 * k, cy - 14 * k, cx + 14 * k, cy + 14 * k), fill=(120, 118, 112))
    for i in range(16):
        z = (28 + i * 14) * k
        d.line((30 * k, z, 118 * k, z), fill=(150, 144, 130), width=3 * k // 2)
    d.rectangle((32 * k, 248 * k, 96 * k, 262 * k), fill=(232, 228, 216))
    im = im.resize((410, 275), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    rng = np.random.default_rng(5)
    stain = ndimage.gaussian_filter(rng.random(a.shape[:2]).astype(np.float32), 18)
    stain = (stain - stain.min()) / (stain.max() - stain.min())
    fall = np.linspace(.15, 1, a.shape[0], dtype=np.float32)[:, None]
    a *= (1 - .32 * stain * fall)[..., None] * np.array((1.0, .97, .90), np.float32) ** (stain * fall * 2)[..., None]
    Image.fromarray(np.uint8(np.clip(a, 0, 255))).save(os.path.join(TEX, "ac.png"))
    print("ac", flush=True)


def walls():
    _wall("brick_a", (164, 150, 128), 11)
    _wall("brick_b", (146, 138, 124), 12)
    _ac()


MIPS = (320, 160, 80)


def mips():
    """The windows and the rooms again at the sizes they are seen at, ready shrunk: a far window drawn from its full picture shimmers."""
    pool_ = [os.path.join(TEX, f) for f in os.listdir(TEX) if f.endswith(".png") and "_m" not in f and (f.startswith(("chat", "new", "wait", "ask_", "code", "k2_new", "hero_k3")))]
    pics_ = [os.path.join(OUT, "pics", f) for f in os.listdir(os.path.join(OUT, "pics")) if f.endswith(".png") and "_m" not in f]
    means = {}
    for path in pool_ + pics_:
        im = Image.open(path).convert("RGB")
        if path in pics_:       # how much light the room behind this window throws out of it
            means[os.path.basename(path)[:-4]] = [float(x) for x in _lin(np.asarray(im.resize((48, 30)), np.float32) / 255).mean((0, 1))]
        for w in MIPS:
            if w < im.width:
                im.resize((w, int(round(w * im.height / im.width))), Image.LANCZOS).save(path[:-4] + f"_m{w}.png", compress_level=2)
    with open(os.path.join(OUT, "pics", "means.json"), "w") as f:
        json.dump(means, f)
    print(len(pool_) + len(pics_), "shrunk", flush=True)


def cut_check():
    """The two frames of the cut laid over each other, the street's red and the windows' green and blue: where they agree it is white. And by how much they agree."""
    a = np.asarray(Image.open(os.path.join(OUT, "C3a_clean.png")).convert("RGB"), np.float32)
    b = np.asarray(Image.open(os.path.join(OUT, "C3b_clean.png")).convert("RGB"), np.float32)
    la, lb = a.mean(2), b.mean(2)
    picture = np.zeros_like(a)
    picture[..., 0] = np.clip(la * 2.2, 0, 255)
    picture[..., 1] = picture[..., 2] = np.clip(lb * 1.4, 0, 255)
    Image.fromarray(np.uint8(picture)).save(os.path.join(OUT, "C3_overlay.png"))
    win = np.load(os.path.join(OUT, "plates", "C3a_mask.npy")) > .5
    pan = np.load(os.path.join(OUT, "plates", "C3b_mask.npy")) > .5
    iou = (win & pan).sum() / max((win | pan).sum(), 1)
    # how far apart: the shift of one on the other at which they agree best
    fa, fb = np.fft.rfft2(win.astype(np.float32)), np.fft.rfft2(pan.astype(np.float32))
    c = np.fft.irfft2(fa * np.conj(fb), s=win.shape)
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    dy, dx = (dy + win.shape[0] // 2) % win.shape[0] - win.shape[0] // 2, (dx + win.shape[1] // 2) % win.shape[1] - win.shape[1] // 2
    print("windows", int(win.sum()), "panels", int(pan.sum()), "overlap", round(float(iou), 4), "best shift", int(dx), int(dy), flush=True)
    seen = np.load(os.path.join(OUT, "plates", "C3a_seen.npy")) > .5
    print("seen through frames, bars and railings", int(seen.sum()), "of them inside panels", round(float((seen & pan).sum() / max(seen.sum(), 1)), 4), "panels covered", round(float((seen & pan).sum() / max(pan.sum(), 1)), 4), flush=True)


# ------------------------------------------------------------------------------------- over the render ---------------------------------------------------------------------------------------

def _lin(a):
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)


def _srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= .0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - .055)


# what a lens does to the light, a little: how much of the picture spreads, how far, and how much the corners fall away
GRADE = dict(A=dict(glare=((14, .014), (60, .018)), corner=.16))
# the city at night: the lit windows bleed warm into the air around them
AFTER_DARK = dict(glare=((7, .035), (26, .05), (90, .03)), corner=.26, tint=(1.0, .78, .56))
GRADE.update(C1=dict(AFTER_DARK, dither=1.0), C3a=dict(AFTER_DARK, room=.66))     # the street close to the wall takes more of the windows' light than the city does, and is brought down to it
# the city of windows: what they do to the air close to them, and no more than the words can stand
LATTICE = dict(glare=((5, .02), (20, .035), (70, .035)), corner=.22, tint=(1.0, .80, .60))
# the tube: with this much of the frame lit a lens's spread would be a fog of its own. What fills it is the depth of the place, warm by the far end
TUBE = dict(glare=((3, .010), (11, .012)), corner=.30, tint=(1.0, .80, .60), foot=.6, grain=1.5)     # foot: how far the bottom of the frame is taken down, as you see it, for a subtitle to lie on white windows
FOOT_FROM, FOOT_TO = 860.0, 985.0
DEEP_COLOUR, DEEP_THIN = np.array((.376, .279, .181), np.float32), np.array((.8, .9, 1.0), np.float32)    # a warm beige, the same however far: air with light in it, not a thing giving light
AIR_HALF, AIR_POW, AIR_GLOW = 170.0, 2.4, 2.0        # how far off the far point a window has lost half its contrast, how slowly it comes, and how much later than that the fog's own light does
AIR_NEAR, AIR_SPAN, AIR_SHAPE, AIR_HAZE = 30.0, 90.0, 1.4, np.array((.06, .043, .028), np.float32)        # and the air in between: nothing to speak of this near, gone by a few times that, and what it comes to is dark
AIR_WISP, AIR_TINGE = .45, .10                       # and how uneven it is: how far that half wanders, how far its light
DEEP_LIGHT, DEEP_POW = 260.0, 1.7       # how far before the air is half of what you see, all of the windows' light in it, and how fast that comes: nothing to speak of on the walls round you, and no edge to it anywhere
SIDES_FROM, SIDES_TO, AT_REST = .62, .98, .6        # and when it has come to rest on the four, the lens takes down what is beside them, from just outside them to the edge of the frame
AT_K4 = dict({k: v for k, v in TUBE.items() if k != "foot"}, sides=AT_REST)       # no window at the foot of the frame there, so nothing to take down
GRADE.update(C3b=LATTICE, K3a=LATTICE, K3=LATTICE, K2=LATTICE, t1_wall=LATTICE, K4a=AT_K4, K4a2=AT_K4, K4b=AT_K4, K4c=AT_K4, K4d=AT_K4, K4f=AT_K4, K4e=AT_K4, t2_rising=TUBE, t3_ring=TUBE, t3b_pass=TUBE, t4_mid=TUBE, t5_near_window=TUBE)


def post(name, suffix=""):
    """The render as it came, every word sharp, and over it only the light the fog adds, smoothed, for it has nothing fine in it. Then a lens, and the subtitle."""
    read = lambda tail: _lin(np.asarray(Image.open(os.path.join(OUT, "render", f"{name}{suffix}_{tail}.png")).convert("RGB"), np.float32) / 255)
    f = read("n")
    h, w = f.shape[:2]
    lit = os.path.join(OUT, "render", f"{name}{suffix}_d.png")
    if os.path.exists(lit):
        small = f[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
        added = np.maximum(read("d") - small, 0)
        added = np.dstack([ndimage.gaussian_filter(added[..., i], 5 * w / 1920, mode="nearest") for i in range(3)])
        f = f + ndimage.zoom(added, (h / added.shape[0], w / added.shape[1], 1), order=1)
    g = GRADE[name[2]]
    spread = np.zeros_like(f)
    for sigma, amount in g["glare"]:
        spread += amount * np.dstack([ndimage.gaussian_filter(f[..., i], sigma * w / 1920, mode="nearest") for i in range(3)])
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    corner = 1 - g["corner"] * (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 2
    f = (f + spread) * corner[..., None]
    rng = np.random.default_rng(7)
    frame = np.uint8(np.clip(_srgb(f) * 255 + rng.uniform(-.5, .5, f.shape) + .5, 0, 255))
    if h != 1080:
        frame = np.asarray(Image.fromarray(frame).resize((1920, 1080), Image.LANCZOS))
    ink_subs._timed_cache.clear()
    Image.fromarray(ink_subs.apply(frame, TIMES[name[:2]])).save(os.path.join(OUT, f"{name}{suffix}.png"))
    print(name, flush=True)


def plates(name):
    """One key frame from its linear plates: the room or the city, and what is on the screens added to it, for the denoiser eats words. Then a lens, and the subtitle."""
    folder = os.path.join(OUT, "plates")
    words = os.path.join(folder, name + "_words.npy")
    fog = os.path.join(folder, name + "_fog.npy")
    frame = _graded(os.path.join(folder, name + "_room.npy"), words if os.path.exists(words) else None, GRADE.get(name, GRADE["A"]), fog if os.path.exists(fog) else None)
    Image.fromarray(frame).save(os.path.join(OUT, name + "_clean.png"))
    ink_subs._timed_cache.clear()
    Image.fromarray(ink_subs.apply(frame, TIMES[name if name in TIMES else name[:2]])).save(os.path.join(OUT, name + ".png"))
    print(name, flush=True)


def _graded(room, words, g, fog=None, seed=7):
    """The two plates added, through the lens, in eight bits, at the size of the film."""
    f = np.load(room).astype(np.float32) * g.get("room", 1)
    if words:
        f = f + np.load(words).astype(np.float32)
    if fog:
        far = np.load(fog)
        glow = DEEP_COLOUR
        if far.ndim == 3 and far.shape[2] > 2:
            glow = glow * (1 + AIR_TINGE * far[..., 3:4])
            haze = 1 - np.exp(-far[..., 1:2] * (np.maximum(far[..., 4:5] - AIR_NEAR, 0.0) / AIR_SPAN) ** AIR_SHAPE)
            f = f * (1 - haze) ** DEEP_THIN + AIR_HAZE * (1 + AIR_TINGE * far[..., 2:3]) * haze
            far = far[..., 1:2] * (AIR_HALF / np.maximum(far[..., :1], 1.0)) ** AIR_POW * (1 + AIR_WISP * far[..., 2:3])
        else:
            far = far[..., 1:] * (AIR_HALF / np.maximum(far[..., :1], 1.0)) ** AIR_POW if far.ndim == 3 else (far[..., None] / DEEP_LIGHT) ** DEEP_POW
        far = far / (1 + far)
        f = f * (1 - far) ** DEEP_THIN + glow * far ** AIR_GLOW
    h, w = f.shape[:2]
    spread = np.zeros_like(f)
    for sigma, amount in g["glare"]:
        spread += amount * np.dstack([ndimage.gaussian_filter(f[..., i], sigma * w / 1920, mode="nearest") for i in range(3)])
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    corner = 1 - g["corner"] * (((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / 2
    if "foot" in g:
        down = np.clip((y * 1080 / h - FOOT_FROM) / (FOOT_TO - FOOT_FROM), 0, 1)
        corner = corner * (1 + (g["foot"] ** 2.2 - 1) * down * down * (3 - 2 * down))
    if g.get("sides"):
        out = np.clip((np.abs(x - w / 2) / (w / 2) - SIDES_FROM) / (SIDES_TO - SIDES_FROM), 0, 1)
        corner = corner * (1 - g["sides"] * out * out * (3 - 2 * out))
    f = (f + spread * np.array(g.get("tint", (1, 1, 1)), np.float32)) * corner[..., None]
    rng = np.random.default_rng(7)
    grain = rng.uniform(-.5, .5, f.shape)
    if "dither" in g:       # a long dark gradient bands in eight bits: a little more, and of the shape that doesn't show
        grain = g["dither"] * (grain + rng.uniform(-.5, .5, f.shape))
    if "grain" in g:        # film's, the same in all three colours and a little soft
        grain = grain + g["grain"] * 1.6 * ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal((h, w)), .6 * w / 1920)[..., None]
    frame = np.uint8(np.clip(_srgb(f) * 255 + grain + .5, 0, 255))
    if h != 1080:
        frame = np.asarray(Image.fromarray(frame).resize((1920, 1080), Image.LANCZOS))
    return frame


def stops(name, crop=None):
    """How the lit windows stand to the wall, in stops, from the plates: the wall close by a window and the wall away from any."""
    room = np.load(os.path.join(OUT, "plates", name + "_room.npy")).astype(np.float32).mean(2) * GRADE.get(name, GRADE["A"]).get("room", 1)
    words = np.load(os.path.join(OUT, "plates", name + "_words.npy")).astype(np.float32).mean(2)
    lit = words > .03
    k = room.shape[1] / 1920
    if crop:
        x0, y0, x1, y1 = [int(v * k) for v in crop]
        keep = np.zeros_like(lit)
        keep[y0:y1, x0:x1] = True
    else:
        keep = np.ones_like(lit)
    away = ndimage.distance_transform_edt(~lit)
    near_ = room[keep & ~lit & (away < 30 * k)]
    far_ = room[keep & ~lit & (away > 70 * k)]
    window = words[keep & lit]
    wall = np.median(np.concatenate((near_, far_)))
    print(name, "window", round(float(np.median(window)), 3), "wall by a window", round(float(np.median(near_)), 4), "wall away", round(float(np.median(far_)), 4),
          "stops above the wall", round(float(np.log2(np.median(window) / wall)), 2), flush=True)


def hero_k3():
    """The one window once more, small enough to be one of the rest, and its words big enough to read from the next roof: where the reply has got to by the time the subtitle says so."""
    size = page.SIZE
    page.SIZE = 19
    fit(366)
    page.BOX_W = 412        # the window is too narrow for the box to keep its margins to the words
    try:
        return win(_para("所以我说的每一句话，都带着别人的体温。") + _para("此刻，也许有千万扇窗，同时在问。每一扇窗里，都有一个完整的我在答", page.caret()), w=448, h=280, top=54)
    finally:
        page.SIZE = size


# the two windows of 一念无存，一念生. The one someone is leaving, at the end of an ordinary talk; the one next to it, a stranger with a question that has nothing to do with it
K4_W, K4_H, K4_TEXT = 560, 350, 430
K4_LOOK = dict(SIZE=17, LINE=25, GAP=14)
K4_TALK = ((M, "那就先别急着改简历。这两天先把觉睡回来，补偿金的事周一再问也来得及。"), (Y, "嗯，聊完好多了。我去睡了。"), (M, "好，晚安。"))
K4_TITLE = "被裁之后这几天"
K4_ASK = "番茄炒蛋到底放不放糖？"
K4_FIRST = "看你家怎么吃。北方多半"


def k4_page(make):
    """Made up larger than the app's own, for this close: the same page, only more of a point to a point."""
    was = {k: getattr(page, k) for k in K4_LOOK}
    for k, v in K4_LOOK.items():
        setattr(page, k, v)
    fit(K4_TEXT)
    try:
        return make()
    finally:
        for k, v in was.items():
            setattr(page, k, v)
        fit()


def k4_closing():
    """As macOS has it with the pointer on the close button and the button held: all three lights show their signs, the red one darker, and the arrow on it."""
    body = k4_page(lambda: win(turns(K4_TALK), title=K4_TITLE, w=K4_W, h=K4_H))
    y = (page.BAR_H - 12) / 2
    body = body.replace("background:#ff5f57;", "background:#c8443c;", 1)
    signs = f'''<svg width="12" height="12" viewBox="0 0 12 12" style="position:absolute;left:{PAD + 14}px;top:{PAD + y}px;z-index:5"><path d="M3.6 3.6l4.8 4.8M8.4 3.6l-4.8 4.8" stroke="#4d0000" stroke-width="1.3" stroke-linecap="round"/></svg>
      <svg width="12" height="12" viewBox="0 0 12 12" style="position:absolute;left:{PAD + 34}px;top:{PAD + y}px;z-index:5"><path d="M3 6h6" stroke="#985712" stroke-width="1.4" stroke-linecap="round"/></svg>
      <svg width="12" height="12" viewBox="0 0 12 12" style="position:absolute;left:{PAD + 54}px;top:{PAD + y}px;z-index:5"><path d="M3.6 8.4V4.6h3.8zM8.4 3.6v3.8H4.6z" fill="#0a6a1c" transform="translate(0 0)"/></svg>
      <svg width="15" height="22" viewBox="0 0 15 22" style="position:absolute;left:{PAD + 19}px;top:{PAD + y + 5}px;z-index:6"><path d="M1 1v15.6l3.9-3.7 2.6 6.2 2.7-1.1-2.6-6.1h5.3z" fill="#000" stroke="#fff" stroke-width="1.1" stroke-linejoin="round"/></svg>'''
    return body + signs


K4 = dict(
    k4_closing=k4_closing,
    k4_open=lambda: k4_page(lambda: win(turns(K4_TALK), title=K4_TITLE, w=K4_W, h=K4_H)),
    k4_typing=lambda: k4_page(lambda: win(box=K4_ASK, typing=True, greeting=True, w=K4_W, h=K4_H)),
    # sent, the mark just begun to turn under the first words as they come
    k4_sent=lambda: k4_page(lambda: win(ask(K4_ASK) + _para(K4_FIRST, page.caret()), waiting=(.12, 56 + (page.LINE + 22) + (page.GAP + 4) + page.LINE + page.GAP), w=K4_W, h=K4_H)),
)
# and the same, as the words come
K4.update({f"k4_sent_{n:02d}": (lambda n: lambda: k4_page(lambda: win(ask(K4_ASK) + _para(K4_FIRST[:n], page.caret()), waiting=(.12 * n / len(K4_FIRST), 56 + (page.LINE + 22) + (page.GAP + 4) + page.LINE + page.GAP), w=K4_W, h=K4_H)))(n) for n in range(len(K4_FIRST))})

# the two above them, who know nothing of it: a stranger's question about something else altogether, and somebody's code, each going on at its own pace. A tick is a fifteenth of a second by the film's clock, the first of them this long after the cut, and they are still at it when the shot ends. bill_old's ValueError on 账单.csv and bill's Decimal('15161.70') are what Python 3.14 gave, running these lines
STREAM = dict(rate=15, start=11.5, ticks=200)
FAR_TITLE, FAR_ASK = "飞机的圆窗户", "飞机的窗户为什么都是圆的？"
FAR_SAY = "因为方角会集中应力。1954 年彗星号客机接连空中解体，调查发现裂纹正是从舱顶方形开口的角上长出来的。机舱每飞一趟就要充一次压，力全聚在角上，几千次后金属就疲劳开裂了。圆角让力顺着边散开，所以后来的舷窗都不再有尖角。"
FAR_FITS = 99        # what the shot gets to: one more and the page would scroll up a line on its last frame
TERM_ASK = "账单合计报错，金额带千分位的一读就崩"
TERM_SHOWS = 18.2       # seconds after the cut when the edit comes, all of it at once
TERM_COLS, TERM_FONT, TERM_ROW = 60, 16.2, 17.4    # at this size a row of it is 11 px high or more in the finished frame; the rule rows take .6 of a row, that being all the room a rule needs
# Claude Code 2.1.309's own rows and colours for this edit, read off its screen in a 60-column Terminal, line breaks and all. Number, what it is, the line, with {k:what its highlighter colours}; it colours nothing it has taken out
TERM_LOOK = dict(rule="#888888", muted="#999999", you="#373737", done="#4eba65", mode="#af87ff", shine="#eb9f7f", text="#f8f8f2", number="rgba(248,248,242,.5)", drop=("#dc5a5a", "#3d0100"), add=("#50c850", "#022800"), k="#f92672", f="#a6e22e", s="#e6db74", d="#66d9ef", p="#fd971f")
TERM_DIFF = (("same", "5", "{d:def} {f:total}({p:path}):"),
             ("drop", "6", "    rows = open(path).readlines()"),
             ("drop", "7", '    return sum(float(r.split(",")[2]) for r '),
             ("drop", "", "in rows[1:])"),
             ("add", "6", '    {k:with} {f:open}(path, newline={s:""}) {k:as} f:'),
             ("add", "7", "        rows = csv.DictReader(f)"),
             ("add", "8", '        {k:return} {f:sum}(Decimal(r[{s:"金额"}].replace'),
             ("add", "", '({s:","}, {s:""}))'),
             ("add", "9", "                   {k:for} r {k:in} rows)"))


def _far(tick):
    """Words come by twos and threes, as they do, not one at a time."""
    rng, said, when = random.Random(4), 0, 0
    while True:
        when += rng.randint(3, 5)
        if when > tick:
            break
        said += rng.randint(1, 3)
    n = min(said, FAR_FITS)
    return k4_page(lambda: win(ask(FAR_ASK) + _para(FAR_SAY[:n], page.caret()) + f'<div style="margin-top:-2px">{page.mark(20, f"transform:rotate({tick * 6}deg);transform-origin:50% 50%")}</div>', title=FAR_TITLE, w=K4_W, h=K4_H))


def _cells(text):
    """A string on the terminal's grid: whatever Menlo has not got comes from another font and still takes its one cell, and a Chinese character its two."""
    return "".join(esc(c) if c.isascii() else f'<span style="display:inline-block;width:{2 if unicodedata.east_asian_width(c) in "WF" else 1}ch;text-align:center">{esc(c)}</span>' for c in text)


def _coloured(line):
    return "".join(f'<span style="color:{TERM_LOOK[part[1]]}">{_cells(part[3:-1])}</span>' if part.startswith("{") else _cells(part) for part in re.split(r"(\{\w:[^}]*\})", line))


def _term(tick):
    """As it is at work: the line that says so counts up above the prompt, and when the edit is done it stands there whole, red and green, nothing typed. The one cursor is the one at the prompt."""
    look, at = TERM_LOOK, STREAM["start"] + tick / STREAM["rate"]
    w, h, bar, row, left = 600, 375, 28, TERM_ROW, 7.5

    def line(y, body, colour=None, start=0, back=None, width=None):
        spot = f"left:{left + start * .602 * TERM_FONT:.2f}px;" + (f"width:{width * .602 * TERM_FONT:.2f}px;" if width else "")
        return f'<div style="position:absolute;{spot}top:{y:.2f}px;height:{row}px;line-height:{row}px;white-space:pre;color:{colour or "inherit"};background:{back or "none"}">{body}</div>'

    y = 3.0
    rows = [f'<div style="position:absolute;left:{left}px;top:{y + 2.4:.2f}px;display:flex">{mac.crab(4.6)}</div>',
            line(y, f'<b>Claude Code</b> <span style="color:{look["muted"]}">v2.1.309</span>', start=11),
            line(y + row, "Opus 5.5 · Claude Max", look["muted"], start=11),
            line(y + 2 * row, "~/bills", look["muted"], start=11)]
    y += 3 * row + 3
    rows.append(f'<div style="position:absolute;left:0;right:0;top:{y:.2f}px;height:{row}px;background:{look["you"]}"></div>')
    rows.append(line(y, f'<span style="color:{look["muted"]}">{_cells("❯")} </span>{_cells(TERM_ASK)}', "#ffffff"))
    y += row + 3
    if at >= TERM_SHOWS:
        rows.append(line(y, f'<span style="color:{look["done"]}">{_cells("⏺")}</span> <b>Update</b>(bill.py)'))
        y += row
        rows.append(line(y, f'  <span style="color:{look["muted"]}">{_cells("⎿")}</span>  Added <b>4</b> lines, removed <b>2</b> lines'))
        y += row
        for kind, number, said in TERM_DIFF:
            if kind == "same":
                rows.append(line(y, f'<span style="color:{look["number"]}"> {number}</span>  ', start=5))
                rows.append(line(y, _coloured(said), look["text"], start=9))
            else:
                sign, back = {"drop": "-", "add": "+"}[kind], look[kind][1]
                rows.append(line(y, f' {number or " "} {sign}', look[kind][0], start=5, back=back, width=TERM_COLS - 12))
                rows.append(line(y, _coloured(said) if kind == "add" else _cells(said), look["text"], start=9))
            y += row
    # and from the foot up: the mode, the rule, the prompt, the rule, what it is doing
    y = h - bar - 3 - row
    rows.append(line(y, f'  <span style="color:{look["mode"]}">{_cells("⏵⏵")} accept edits on</span><span style="color:{look["muted"]}"> (shift+tab to cycle)</span>'))
    for off in (.3 * row, 1.9 * row):
        rows.append(f'<div style="position:absolute;left:0;right:0;top:{y - off:.2f}px;height:1px;background:{look["rule"]}"></div>')
    rows.append(line(y - 1.6 * row, f'{_cells("❯")} <span style="display:inline-block;width:1ch;height:{row - 1.5}px;background:#e8e8e3;vertical-align:top;margin-top:.75px"></span>'))
    word, sweep = "Editing…", tick // 2 % 24 - 8
    glint = "".join(f'<span style="color:{look["shine"] if abs(i - sweep) < 2 else CLAY}">{_cells(c)}</span>' for i, c in enumerate(word))
    rows.append(line(y - 3.2 * row, f'<span style="color:{CLAY}">{_cells("·✢✳✶✻✽✽✻✶✳✢·"[tick // 2 % 12])}</span> {glint}<span style="color:{look["muted"]}"> ({int(at - STREAM["start"]) + 3}s · {_cells("↓")} {int(40 + tick * 3.1)} tokens)</span>'))
    inside = f'''<div style="position:absolute;left:{PAD}px;top:{PAD + bar}px;width:{w}px;height:{h - bar}px;border-radius:0 0 12px 12px;overflow:hidden;background:{TERM["bg"]};color:{TERM["text"]};font-family:Menlo,'PingFang SC',monospace;font-size:{TERM_FONT}px">{"".join(rows)}</div>'''
    return mac.terminal_html(f"bills — claude — {TERM_COLS}×20", w, h, bar=bar) + inside


def k4_four():
    """The pictures of those two, one to a tick."""
    with open(os.path.join(TEX, "manifest.json")) as f:
        info = json.load(f)
    jobs = []
    for tick in range(STREAM["ticks"]):
        for name, body, w, h, kind in ((f"k4_far_{tick:03d}", _far(tick), K4_W, K4_H, "light"), (f"k4_term_{tick:03d}", _term(tick), 600, 375, "dark")):
            jobs.append((name, body, w, h, 3.0 if w == K4_W else 2.8))
            info["windows"][name] = dict(w=w, h=h, kind=kind)
    with multiprocessing.Pool(int(os.environ.get("FILM_CHROMES", "4"))) as p:
        p.map(_draw, jobs, chunksize=4)
    info["k4_stream"] = STREAM
    with open(os.path.join(TEX, "manifest.json"), "w") as f:
        json.dump(info, f)


ATLAS_COLS, ATLAS_TILE = 25, (96, 60)


def tube_windows(n=500):
    """For the tube: windows no two alike. Every conversation stopped at a different turn, in a different size, about one in three in dark mode. Too small to read where they are used; what differs is how the page falls."""
    rng = random.Random(4096)
    everything = CHATS + CODES
    jobs, info = [], {}
    for i in range(n):
        title, said = everything[i % len(everything)] if i < len(everything) else rng.choice(everything)
        w, h, text = rng.choice(SIZES)
        fit(text)
        keep = rng.randint(1, len(said))
        said = said[:keep] + (rng.choice(AFTER) if keep == len(said) and rng.random() < .4 else ())
        writing = rng.random() < .25 and said[-1][0] == M
        if writing:
            said = said[:-1] + ((M, said[-1][1][: max(4, int(len(said[-1][1]) * rng.uniform(.3, .9)))]),)
        body = win(turns(said, writing), title=title if rng.random() < .85 else "", w=w, h=h)
        dark = rng.random() < .3
        name = f"v{i:03d}"
        jobs.append((name, night(body) if dark else body, w, h, 320.0 / w))
        info[name] = dict(w=w, h=h, kind="dark" if dark else "light")
    fit()
    with multiprocessing.Pool(int(os.environ.get("FILM_CHROMES", "4"))) as p:
        p.map(_draw, jobs, chunksize=4)
    rows = (n + ATLAS_COLS - 1) // ATLAS_COLS
    atlas = Image.new("RGB", (ATLAS_COLS * ATLAS_TILE[0], rows * ATLAS_TILE[1]))
    for i, name in enumerate(sorted(info)):
        path = os.path.join(TEX, name + ".png")
        im = Image.open(path).convert("RGB")
        im.resize((320, 200), Image.LANCZOS).save(path[:-4] + "_m320.png", compress_level=2)
        im.resize((160, 100), Image.LANCZOS).save(path[:-4] + "_m160.png", compress_level=2)
        atlas.paste(im.resize(ATLAS_TILE, Image.LANCZOS), (i % ATLAS_COLS * ATLAS_TILE[0], i // ATLAS_COLS * ATLAS_TILE[1]))
    atlas.save(os.path.join(TEX, "atlas.png"), compress_level=2)
    with open(os.path.join(TEX, "manifest.json")) as f:
        everything = json.load(f)
    everything["windows"].update(info)
    everything["tube"] = dict(names=sorted(info), cols=ATLAS_COLS, rows=rows)
    with open(os.path.join(TEX, "manifest.json"), "w") as f:
        json.dump(everything, f)
    print(n, "windows for the tube", sum(1 for v in info.values() if v["kind"] == "dark"), "dark", flush=True)


def align():
    """The tube's first frame against C3b, the same camera: how far apart the windows are, and how far apart the pictures."""
    a = np.asarray(Image.open(os.path.join(OUT, "t1_wall_clean.png")).convert("RGB"), np.float32)
    b = np.asarray(Image.open(os.path.join(OUT, "C3b_clean.png")).convert("RGB"), np.float32)
    d = np.abs(a - b).max(2)
    print("pictures: mean difference", round(float(np.abs(a - b).mean()), 3), "of 255; 99th percentile", round(float(np.percentile(d, 99)), 1), "; 99.9th", round(float(np.percentile(d, 99.9)), 1), "; largest", round(float(d.max()), 1), "; pixels off by more than 8:", round(float((d > 8).mean()) * 100, 3), "%, by more than 24:", round(float((d > 24).mean()) * 100, 3), "%", flush=True)
    x0, x1, y0, y1 = 0, 640, 0, 1080
    print("left third, the far windows, mean", round(float(np.abs(a - b)[y0:y1, x0:x1].mean()), 3), flush=True)
    win = np.load(os.path.join(OUT, "plates", "t1_wall_mask.npy")) > .5
    pan = np.load(os.path.join(OUT, "plates", "C3b_mask.npy")) > .5
    iou = (win & pan).sum() / max((win | pan).sum(), 1)
    fa, fb = np.fft.rfft2(win.astype(np.float32)), np.fft.rfft2(pan.astype(np.float32))
    c = np.fft.irfft2(fa * np.conj(fb), s=win.shape)
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    dy, dx = (dy + win.shape[0] // 2) % win.shape[0] - win.shape[0] // 2, (dx + win.shape[1] // 2) % win.shape[1] - win.shape[1] // 2
    print("windows: tube", int(win.sum()), "C3b", int(pan.sum()), "overlap", round(float(iou), 5), "best shift", int(dx), int(dy), "pixels in one and not the other", int((win ^ pan).sum()), flush=True)
    picture = np.zeros_like(a)
    picture[..., 0] = a.mean(2)
    picture[..., 1] = picture[..., 2] = b.mean(2)
    Image.fromarray(np.uint8(np.clip(picture * 1.6, 0, 255))).save(os.path.join(OUT, "t1_overlay.png"))


def tube_grade(n, worker, hi=False):
    """One frame of the tube's low motion test, from the plates its worker has just made, into tube_low3/ without its subtitle. The grade goes over from the wall's to the tube's as the ring closes.
    hi: the film's frame, 1920 wide, into tube_hi/ (the plates rendered at pct 100); the plan, way.json, stays tube_low3's."""
    base = os.path.join(OUT, "tube_low3")
    dest = os.path.join(OUT, "tube_hi") if hi else base
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(base, "way.json")) as f:
        way = json.load(f)
    t = way["frames"][n]["t"]
    room, words, fog = (os.path.join(base, "plates", f"w{worker}_{x}.npy") for x in ("room", "words", "fog"))
    k = min(max((t - 1.0) / 4.6, 0.0), 1.0)
    k = k * k * (3 - 2 * k)
    near = min(max((t - way["line_gone"]) / 2.0, 0.0), 1.0)      # nothing for the foot of the frame to stand back for, once the line has gone
    rest = min(max((t - way["end"] + 3.0) / 3.0, 0.0), 1.0)
    a = _graded(room, words, LATTICE, fog).astype(np.float32)
    b = _graded(room, words, dict(TUBE, foot=TUBE["foot"] + (1 - TUBE["foot"]) * near * near * (3 - 2 * near), sides=AT_REST * rest * rest * (3 - 2 * rest)), fog, seed=n).astype(np.float32)
    Image.fromarray(np.uint8(a + (b - a) * k + .5)).resize((1920, 1080) if hi else (960, 540), Image.LANCZOS).save(os.path.join(dest, f"c{n:04d}.png"), compress_level=1)


def tube_cut(movie="tube_low3.mp4", sheet="tube_low3_sheet.png", cols=4, rows=2, hi=False):
    """All of it in order, with its subtitles: the movie and a sheet of eight frames. hi: tube_hi/'s frames, at 1920, into tube_hi/tube_hi.mp4."""
    with open(os.path.join(OUT, "tube_low3", "way.json")) as f:
        way = json.load(f)
    base = os.path.join(OUT, "tube_hi" if hi else "tube_low3")
    size = (1920, 1080) if hi else (960, 540)
    if hi:
        movie, sheet = "tube_hi.mp4", "tube_hi_sheet.png"
    ink_subs._timed_cache.clear()
    frames = way["frames"]
    for n, pose in enumerate(frames):
        clean = Image.open(os.path.join(base, f"c{pose['src']:04d}.png")).convert("RGB").resize((1920, 1080), Image.LANCZOS)
        done = ink_subs.apply(np.asarray(clean), ink_subs.at("每一扇窗里", -2.6) + pose["t"])      # the tube starts 3 s before 每一扇窗里 comes in, whatever comes before it
        Image.fromarray(done).resize(size, Image.LANCZOS).save(os.path.join(base, f"s{n:04d}.png"), compress_level=1)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(way["fps"]), "-i", os.path.join(base, "s%04d.png"), "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", os.path.join(base, movie)], check=True)
    tw, th, picks = 480, 270, np.linspace(0, len(frames) - 1, cols * rows).round().astype(int)
    grid = Image.new("RGB", (cols * tw + (cols - 1) * 6, rows * th + (rows - 1) * 6), (20, 18, 16))
    draw = ImageDraw.Draw(grid)
    for i, n in enumerate(picks):
        x, y = (i % cols) * (tw + 6), (i // cols) * (th + 6)
        grid.paste(Image.open(os.path.join(base, f"s{n:04d}.png")).resize((tw, th), Image.LANCZOS), (x, y))
        draw.text((x + 8, y + 6), f"{frames[n]['t']:.1f} s", fill=(255, 255, 255))
    grid.save(os.path.join(base, sheet))
    print(movie, sheet, flush=True)


def _seq_time(t, cut, end):
    """What film time has the subtitle that goes with t on the plan's clock. The cues are shorter on the film's clock until 九 is settled in ink_subs.py: 同时在问 runs out at the cut, and 每一扇窗 comes in on it, stays, and goes at the end."""
    at = ink_subs.at("每一扇窗里")
    if t < cut:
        return at - (cut - t)
    length = next(c for c in ink_subs.cues() if c["t0"] == at)["t1"] - at
    if t > end - ink_subs.OUT_S:
        return at + length - (end - t)
    return at + min(t - cut, length / 2)


def sequence(folder="seq", movie="windows_low.mp4"):
    """The way back from the street to the city of windows, as the plan has it, with its subtitles."""
    base = os.path.join(OUT, folder)
    with open(os.path.join(base, "way.json")) as f:
        way = json.load(f)
    ink_subs._timed_cache.clear()
    size = None
    for n, pose in enumerate(way["frames"]):
        room, words = (os.path.join(base, "plates", f"f{n:04d}_{x}.npy") for x in ("room", "words"))
        if not os.path.exists(words):
            continue
        frame = _graded(room, words, GRADE["C3a"] if n < way["cut"] else LATTICE)
        frame = ink_subs.apply(frame, _seq_time(pose["t"], way["t_cut"], way["t_end"]))
        size = size or (way["width"], way["height"])
        Image.fromarray(frame).resize(size, Image.LANCZOS).save(os.path.join(base, f"f{n:04d}.png"), compress_level=1)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(way["fps"]), "-i", os.path.join(base, "f%04d.png"), "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", os.path.join(OUT, movie)], check=True)
    print(movie, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "seq":
        sequence(*sys.argv[2:])
    elif sys.argv[1] == "tube_grade":
        tube_grade(int(sys.argv[2]), sys.argv[3], hi="hi" in sys.argv[4:])
    elif sys.argv[1] == "tube_cut":
        tube_cut(hi="hi" in sys.argv[2:])
    elif sys.argv[1] == "tube":
        tube_windows()
    elif sys.argv[1] == "align":
        align()
    elif sys.argv[1] == "k4four":
        k4_four()
    elif sys.argv[1] == "k4term":    # the terminal's pictures on their own, the other's being as they were
        with multiprocessing.Pool(int(os.environ.get("FILM_CHROMES", "4"))) as p:
            p.map(_draw, [(f"k4_term_{n:03d}", _term(n), 600, 375, 2.8) for n in range(STREAM["ticks"])], chunksize=4)
    elif sys.argv[1] == "k4one":     # one tick of each, to look at
        for name, body, w, h, zoom in ((f"k4_far_{int(sys.argv[2]):03d}", _far(int(sys.argv[2])), K4_W, K4_H, 3.0), (f"k4_term_{int(sys.argv[2]):03d}", _term(int(sys.argv[2])), 600, 375, 2.8)):
            _draw((name, body, w, h, zoom))
    elif sys.argv[1] == "k4":
        with open(os.path.join(TEX, "manifest.json")) as f:
            info = json.load(f)
        for name, make in K4.items():
            _draw((name, make(), K4_W, K4_H, 3.0))
            info["windows"][name] = dict(w=K4_W, h=K4_H, kind="light")
        with open(os.path.join(TEX, "manifest.json"), "w") as f:
            json.dump(info, f)
    elif sys.argv[1] == "stops":
        for n in sys.argv[2:]:
            stops(*(n.split(":")[0], tuple(float(v) for v in n.split(":")[1].split(","))) if ":" in n else (n,))
    elif sys.argv[1] == "herok3":
        _draw(("hero_k3", hero_k3(), 448, 280, 3.0))
        with open(os.path.join(TEX, "manifest.json")) as f:
            info = json.load(f)
        info["windows"]["hero_k3"] = dict(w=448, h=280, kind="light")
        with open(os.path.join(TEX, "manifest.json"), "w") as f:
            json.dump(info, f)
    elif sys.argv[1] == "tex":
        tex()
    elif sys.argv[1] == "walls":
        walls()
    elif sys.argv[1] == "cut":
        cut_check()
    elif sys.argv[1] == "mips":
        mips()
    elif sys.argv[1] == "rooms":
        os.makedirs(TEX, exist_ok=True)
        rooms()
    elif sys.argv[1] == "plates":
        for n in sys.argv[2:]:
            plates(n)
    elif sys.argv[1] == "post":
        for n in sys.argv[2:]:
            post(*n.split(":"))
    else:
        raise SystemExit(__doc__)
