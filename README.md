# video-claude：用代码做的科普视频

两期面向抖音、B站、小红书的 AI 科普视频，从分镜、事实核查到画面、音效、母带、封面和发布文案，全部用代码完成。制作工具是 Claude Code；事实核查和起标题时，还通过 Pi 调用了多个模型交叉检查。

| 期 | 内容 | 技术 | 成片 |
|---|---|---|---|
| 第一期 | AlphaGo × 蒙特卡洛树搜索：第 37 手，AI 到底在想什么 | Manim（Community Edition） | B站 16:9（7:27），竖版（1:46） |
| 第二期 | AI 发展路线图：一条线，从 1943 年走到 2026-10-18 | cairo + numpy 逐帧渲染器（不是 Manim） | 抖音竖版 1080×1920、B站横版 1920×1080（5:28） |

成片视频不在仓库里：文件太大（单个约 120 MB），而且压进了有版权的配乐和新闻图片。

## 第二期怎么做的

- **一条线**：
  - 数据是 `ai_history/milestones.csv` 里核实过的 253 个节点，每个节点都有出处和措辞备注。
  - 每个节点让线往上走一步，寒冬让线往下掉；ChatGPT 之后每次突破抬得更高。
  - 镜头把过去往左下压缩，时间轴的单位从「年」变成「月」，再变成「日」。
- **左上角信息窗**：每个节点一个窗口，里面是标题、算法小动画或前后对比的条形图、一句白话解释、背景故事。动画会「飞进」线头。
- **注入**：每次大突破，它的基础节点都会重新亮起，再各飞来一颗光粒。比如 Transformer 会反复被点亮。
- **小条**：较小的进展在线头旁弹出，格式是「日期 + 名字 + 做了什么」。越到后面越密。
- **伏笔**：早期节点下面有一句话，提前说出它多年后的结果，比如「75 年后，这个测试会被 AI 通过」。等那个节点出现时，早期节点会重新亮起来。
- **背景公式**：每个时代都有一条经典公式淡淡飘过，从神经元阈值、链式法则到注意力 softmax(QKᵀ/√dₖ)V。
- **卡点**：`manim/sfx/analyze_bgm.py` 先分析 BGM 的节拍和段落，节点都落在重拍上。
- **核查**：
  - 每个数字都对照过论文、官方公告或主流媒体。
  - 被否掉的说法和原因写在分镜文档里。
  - 千禧年难题这类仍在评审的结果，画面里会标明。

详细设计和每一轮修改见 `manim/scenes/ai_history.scenes.md`（最新的修订在最前面）。

## 目录

```
├── 爆款科普视频手册.md          从选题、分镜、核查到封面、发布的经验总结
├── manim/
│   ├── scenes/
│   │   ├── ai_history_film.py   第二期正片渲染器（竖版 / 横版同一份代码）
│   │   ├── ai_history.scenes.md 第二期分镜与修订记录
│   │   ├── ai_history.py        第二期早期的 Manim 样张
│   │   ├── alphago_mcts.py      第一期 Manim 代码
│   │   ├── alphago_mcts.*       第一期分镜、字幕日志、配音时间表
│   │   └── style.py             配色、字体、阅读时间等共用工具
│   └── sfx/
│       ├── make_sfx.py          用代码合成全部音效（无版权）
│       ├── master_audio.py      母带：响度标准化
│       └── analyze_bgm.py       BGM 节拍、小节、段落分析（卡点用）
├── ai_history/                  第二期资料
│   ├── milestones.csv           核实过的节点（日期、出处、措辞备注）
│   ├── builds_on.csv            节点之间的前置关系（带证据）
│   ├── quotes.csv               核实过的原话和背景故事
│   ├── openai_math_families.json  OpenAI 数学手稿的成果标题（来自 github.com/openai/math，Apache-2.0）
│   ├── images.md、images/LICENSES.md  用到的每张图的来源与授权（图片本身不在仓库里）
│   ├── copy.md                  抖音、B站的标题、简介、章节、置顶评论
│   └── cover/                   封面（由片尾总览帧生成）
└── alphago/                     第一期的封面源文件、截图和发布文案
```

## 自己跑起来

环境：Python 3.12，以及 `pycairo numpy scipy Pillow`，第一期还要 `manim`；命令行里要有 `ffmpeg`。字体用 macOS 自带的宋体（Songti SC）、Times New Roman、STIX Two Math。

第二期（在 `manim/scenes` 下）：

```bash
python ai_history_film.py check                          # 核对数据和时间轴
python ai_history_film.py snap 20 140 214 300            # 抽几帧，出联系表
python ai_history_film.py render 0 328 --jobs 8 --name out             # 竖版
AIH_ORIENT=landscape python ai_history_film.py render 0 328 --jobs 8 --name out_16x9   # 横版
```

渲染之前要先补齐仓库里没有的素材：

- **图片**：按 `ai_history/images/LICENSES.md` 的来源下载，放进 `ai_history/images/`。
- **配乐**：放进 `ai_history/bgm/`。文件名见渲染器开头的 `BGM_FILE`；节拍表用 `analyze_bgm.py` 生成。
- **音效**：`python manim/sfx/make_sfx.py` 重新生成。

第一期：

```bash
manim -qh --disable_caching manim/scenes/alphago_mcts.py AlphaGoMCTS            # 16:9
manim -r 1080,1920 --disable_caching manim/scenes/alphago_mcts.py AlphaGoMCTSShort  # 竖版
python manim/sfx/master_audio.py 原片.mp4 成片.mp4 --target -24 --peak -1.5
```

## 许可

- 代码和本仓库里我们自己写的文档、数据：MIT（见 `LICENSE`）。
- 不包括第三方素材：
  - 图片、配乐不在仓库里，版权归原作者。
  - `openai_math_families.json` 遵循 Apache-2.0。
  - 引用的原话和论文标题都注明了出处。
