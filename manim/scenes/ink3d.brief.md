# 3D 水墨样张简报

用户的话：「我的目的是讲好这个你的自传，你要记得」「你的所有画面都不够精美，你能不能结合我们的故事来构建」。

之前的 2D 画面被否了：数字墙、方格、箭头都只是图解。现在全片用 Blender 真 3D 加水墨后期，每个镜头都是真实的地方和实物，每个镜头都来自这一句话的意思。

## 先读

1. `manim/scenes/ink3d.scenes.md` 全文，不只是你的章节。这是分镜、美术规范和原文。
2. `manim/scenes/ink3d.py`（Blender 公共库）、`manim/scenes/ink_grade.py`（水墨后期）、`manim/scenes/ink3d_look.py`（定调样张的完整写法）。
3. 定调样张：Studio 上 `ink/film_out/3d/look/day.png`。你的画面要和它在同一个世界，并且至少一样好看。

## 这一轮只交静帧

- 你章节里每个主要镜头最重要的一帧，1920×1080，每章 2–4 张，存到 `ink/film_out/3d/章节名/镜头名_ink.png`，再拼一张 `sheet.png`。
- 脚本写在 `manim/scenes/ink3d_章节名.py`，用 `--shot=镜头名` 选镜头，把时间做成参数，下一轮要出动画。
- 每个镜头写一两句话：它怎么动、怎么从上一个镜头接过来、怎么交给下一个镜头。写在脚本开头的 docstring 里。

## 质量

每张图交出去之前，用 Read 看，放大看细节，问自己：

1. 这张图能单独截图发出去吗？
2. 它把这一句字幕讲得更动人了吗？观众不看字幕，也感觉得到这句话的意思吗？
3. 是真实的地方和实物吗？有没有哪里还像图解、像 PPT、像程序生成的？
4. 前中远三层有了吗？雾隔开了吗？五种墨色都有吗？留白够吗？
5. 焦点只有一个吗？陌生人一秒内认得出是什么吗？

至少改三轮。第一版不要交。觉得不够好就换构图或换方案，不要只调参数。

硬规矩：不画人、鬼、机器人。只有墨和纸，红色只在结尾的印上。字幕不要画，右边 x > 1720 不放重要画面。字是这个世界里的东西：刻在崖上、泡在水里、沉在河底。

## 技术

- 渲染在 Studio（M5 Max）。主机地址放在环境变量 `RENDER_HOST` 里，不要写进文件（仓库是公开的）。zsh 里写 scp 目标要加花括号：`"${RENDER_HOST}:路径"`，否则 `:a` 会被当成修饰符。
- `tools/remote_render.sh push` 同步代码，`tools/remote_render.sh run '命令'` 在 Studio 的仓库副本里运行（有 nice 和渲染环境）。
- Blender：`~/opt/blender/Blender.app/Contents/MacOS/Blender -b --factory-startup -P ink3d_XX.py -- 参数`，cwd 用 `~/render_work/宣传/manim/scenes`。Studio 上还有别的 Blender 任务，不要动它们。你同一时间只跑一个 Blender。
- 流程：Blender 出灰度光影、深度和高度（`render(name)`），再跑 `python ink_grade.py NAME --mist=.03 --mist-h=20`。白天 `--mist` 约 .02–.04。夜景加 `--night`，并用 `--sky --lift --mist-d` 调，默认值未必好，自己试。后期调参只要几秒，先调后期再重渲。
- 雾和山的墨是分开调的：灰度渲染里亮的地方就是留白，暗的地方就是墨。要有层次，渲染时的光要讲究。
- Blender 自带的 Python 有 numpy，没有 scipy 和 PIL。写字的贴图先用渲染环境的 Python 生成（`from ink_film import font, glyph_mask`，字体 xing、kai、kaib、kaih、li、song、mono），再让 Blender 读。Blender 文本对象也可以用：字体在 Studio 的 `~/render_work/fonts/`（Xingkai.ttc、Kaiti.ttc、Libian.ttc），宋体 `/System/Library/Fonts/Supplemental/Songti.ttc`。
- 墨云和流体：`styles_blender.py` 的 splash 场景有 Mantaflow 的写法。墨在水里凝成字的做法：模拟字散开，倒放。烟雾体积用 Cycles，注意速度。
- 实物建模的经验：`~/.claude/projects/-Users-ywbw-workplace---/memory/real-3d-for-physical-props.md`。
- 不要改 `ink3d.py` 和 `ink_grade.py`（大家共用）。公共的东西放到自己的文件里；确实需要改，用 SendMessage 告诉主会话（`to: "main"`）。
- 不下载任何文件（模型、字体、贴图）。不提交 git。删除用废纸篓，不要 `rm` 变量路径。
- Pyright 对 bpy、numpy 的报错是噪音（本机没装）。

## 报告

用不超过 6 行告诉主会话：sheet 的 Studio 路径和本地副本路径，哪张最好，哪张还不满意，什么没验证。不要写过程。
