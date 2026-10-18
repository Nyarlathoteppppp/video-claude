# AlphaGo 是怎么“想”棋的：蒙特卡洛树搜索 — storyboard (public video)

Audience: general viewers on Bilibili / Xiaohongshu who have heard of AlphaGo but know no RL. Goal: after one viewing they can say
“AlphaGo 每一步前都在长一棵树：反复‘选、长、试、传’，树会自己偏向好棋；神经网络让它只长有希望的枝、一眼判断形势。”

Two deliverables from one script file (`alphago_mcts.py`):
- `AlphaGoMCTS` — Bilibili master, 16:9 1920×1080, 3–4 min.
- `AlphaGoMCTSShort` — Xiaohongshu cut, vertical 9:16 (1080×1920), 60–80 s. Same visual language and helpers, re-laid out for portrait (big type, one idea per screen, top third free for a title strip).

Public release rules: no course slides, no photos, no DeepMind/AlphaGo logos; draw the Go board and stones abstractly. Every number shown is computed in the script or listed under “Facts” below.

## Facts (verified, use exactly)

- Go: about 250 legal moves per position, about 150 moves per game → about 250^150 ≈ 10^360 possible games (log10 250 × 150 = 359.7). Atoms in the observable universe ≈ 10^80.
- March 2016: AlphaGo beat Lee Sedol 4 : 1. Game 2, move 37: AlphaGo estimated a human would play it with probability about 1 in 10,000.
- AlphaGo (2016): MCTS guided by a policy network (which moves are worth looking at) and a value network (how good is this position); leaf positions were evaluated by the value network together with a fast rollout. AlphaGo Zero (2017) learned from self-play only and used the network alone.

## Toy problem (computed in the script)

Tic-tac-toe, cells 0–8 row-major. Board: X at 0, O at 2 and 4 (`"X.O.O...."`), **X to move**. O threatens the 2-4-6 diagonal, so the only non-losing move is **6 (bottom-left)**.
UCT MCTS with c = √2, random rollouts, reward 1 / 0.5 / 0 (win / draw / loss) from the mover's view, `random.Random(7)`, 2000 iterations. Reference run (`scratchpad/ttt/mcts.py`):

| iterations | visits per candidate move {1,3,5,6,7,8} |
|---|---|
| 20 | 3, 2, 4, **4**, 3, 4 |
| 50 | 6, 9, 7, **14**, 7, 7 |
| 300 | 25, 54, 35, **125**, 34, 27 |
| 2000 | 63, 118, 87, **1584**, 83, 65 → move 6 gets 79 %; its win-rate ≈ 0.48 (a draw), the others ≤ 0.22 |

The script must re-run this itself and assert: at 20 iterations no move has more than 25 % of visits; at 2000 move 6 has the most visits and > 70 %.

## Bilibili master `AlphaGoMCTS` — beats

Process strip (top): ① 为什么难 ② 随便下完 ③ 一棵树 ④ 选谁 ⑤ 越长越偏 ⑥ AlphaGo

0. **Cold open (≈8 s).** A 19×19 grid fades in, stones appear one by one quickly, then freeze. Big text: “2016 · AlphaGo 4 : 1 李世石”. Hook question replaces it: “它每下一步之前，脑子里在想什么？”
1. **Why brute force fails (≈20 s).** From one stone a tree explodes: 250 branches, each 250 again… A counter climbs 250 → 250² → … → 250^150 ≈ 10^360, next to “宇宙原子 ≈ 10^80” shown as a tiny bar against an enormous one (log scale, labelled). Caption: “把每一种下法都算一遍？宇宙毁灭也算不完”.
2. **A small game first (≈12 s).** Zoom into the tic-tac-toe board above. Highlight O’s threatening diagonal in red. Six candidate moves appear as ghost X’s. Caption: “先看个小例子：X 走，哪一步最好？”
3. **Idea 1 — just play it out (≈20 s).** From one candidate, a ghost game plays out at random to the end (fast), result ✓/✗/平 drops into a tally. Repeat a few times for two candidates. Caption: “不会算，就随便下完一盘，看输赢”. Then the flaw: “随便下，对手也会犯傻——太粗糙”.
4. **Idea 2 — a tree that remembers (≈45 s).** The board becomes the root node; children are the six moves (mini boards). One iteration in slow motion with the four step labels lighting up in turn (persistent side panel):
   - 选择 Selection: walk down from the root picking the best-scoring child;
   - 扩展 Expansion: add one new child (one more mini board);
   - 模拟 Simulation: random playout to the end (the mini ghost game from beat 3);
   - 回传 Backpropagation: the result flows back up the path, each node’s “访问 n / 赢 w” counter ticks.
   Caption per step; then two more iterations at medium speed so the loop is felt.
5. **Idea 3 — whom to try next (≈30 s).** Each child shows a score bar = 胜率 (teal) + 探索加分 (yellow). Formula appears piece by piece, coloured like the bars: score = w/n + c·√(ln N / n). Rarely tried children get tall yellow bonus parts. Caption: “胜率高的多看，看得少的也给机会：乐观面对未知”.
6. **Fast forward — the tree tilts (≈30 s).** Iterations run 20 → 50 → 300 → 2000 with a counter; the tree grows, the branch under move 6 gets thicker and deeper while others stay stubby; a bar chart of visits per move morphs through the table above. Aha (hold 2 s): move 6 lights up — 79 %. Caption: “没人告诉它要堵——搜索自己发现了唯一的正解”. Then: the move actually played = the most-visited child.
7. **AlphaGo’s upgrade (≈35 s).** Back to the 19×19 board: the same tree would be hopeless (250 children everywhere). Two networks appear as simple glowing boxes:
   - 策略网络 policy net: “凭直觉，只看几手” — the 250 children fade to a handful (prior);
   - 价值网络 value net: “一眼看形势” — a dial replaces the long random playout.
   The tree becomes narrow and deep. Caption: “神经网络给搜索装上直觉”. One line: “后来的 AlphaGo Zero 只靠自己和自己下棋学会了这一切”.
8. **Ending (≈15 s).** “第 37 手”: a single stone drops on the abstract board with a glow; text “人类棋手下出这一手的概率：约万分之一”. Final card: “搜索 = 一棵会自己长偏的树” + small line “反复：选 → 长 → 试 → 传”.

## Xiaohongshu cut `AlphaGoMCTSShort` (vertical, 60–80 s)

Beats 0 (hook, 5 s) → 2 (toy board, 6 s) → 4 compressed (the four steps as four big icons cycling once, 15 s) → 6 (the tilting tree and the 79 % bar — the visual centrepiece, 20 s, hold 2 s) → 7 (two networks, 12 s) → 8 (move 37 + final card, 10 s). Big captions (≥ 40 px at 1080 width), top title strip “AlphaGo 在想什么？”, nothing important in the bottom 15 % (platform UI).

## Style

Same palette and rules as the course videos (`style.py`, `可视化技巧/01–03`): colour = meaning (X/agent orange, O/opponent blue, win-rate teal, exploration bonus yellow, wrong/threat red), transform don’t replace, one focus at a time, fast-fast-slow rhythm with holds on the ahas, persistent skeleton (the tree / board stays), clean summary card. Narration sync uses the same optional `voice(key)` mechanism as `log_trick.py`, keyed to `alphago_mcts.voice.json` / `alphago_mcts_short.voice.json`.

## Update (user direction, overrides the beats above where they conflict)

**Audience: ordinary viewers, not students. Keep it easy.** No formulas on screen except plain numbers (10^360, 3.14, 79 %). No jargon on screen: use 局面 / 分支 / 走法 / 看了几次 / 赢了几次 / 从现在的局面出发 instead of 节点 / 访问次数 / 根 / UCB. The four steps are 选 → 长 → 试 → 传 (选择 / 扩展 / 模拟 / 回传 may appear only as small secondary labels).

**New opening (beat 0, ≈40 s, narration h01–h07):**
1. 0–3 s: a single black stone slams onto an abstract 19×19 board (with a short flash/ripple), freeze. Text: “2016 · AlphaGo vs 李世石 · 第二局”.
2. 3–10 s: a question mark over the stone; text cards “解说员：这是失误吗？” → “李世石十几分钟后才回应” → big “人类下出这一手的概率 ≈ 1/10000（AlphaGo 自己的估计）”. Hook question “它到底在想什么？”.
3. 10–14 s: transition card “答案要从 70 多年前的一副纸牌说起” — cards fan out.
4. 14–30 s: Monte Carlo origin, drawn abstractly (no portraits): “1946 · 数学家乌拉姆 · 养病时玩单人纸牌” → cards being dealt into a win/total tally → then the computable demo: random points rain into a square with an inscribed quarter circle; the running estimate 4·(inside/total) converges toward 3.14 (compute it with a fixed seed, show the real running value, e.g. after 10, 100, 1000, 5000 points).
5. 30–36 s: “后来用于核武器研究 → 代号取自赌城：蒙特卡洛” (dice / roulette icon, no casino logo).
6. 36–40 s: “随机试验 + 搜索树 = 蒙特卡洛树搜索” → “AlphaGo 用的正是它” → into beat 1. (MCTS predates AlphaGo; never say AlphaGo invented it.)

**Beat 5 (whom to try next) becomes formula-free:** each candidate move gets a two-colour bar: teal “胜率” + yellow “好奇心加分” (bonus large when 看得少). Compute the bar lengths from the real UCT terms (w/n and c·√(ln N/n)) but do not show the formula.

**Ending (beat 8):** call back to move 37 — the same stone from the opening, now with a tree drawn behind it whose thickest branch leads to that stone (schematic, labelled “示意”), then the final card.

**Vertical cut:** first 3 s = stone + big text “人类万分之一概率才会下的一步”; then “它的秘诀来自 70 年前的一副纸牌：算不出来，就随机试很多次” with 2 s of the point-rain → straight to tic-tac-toe (narration s00, s00b, s01…).

**Fact wording (checked 2026-10-18, overrides any earlier phrasing):** sources disagree on whether Lee Sedol left the room *because of* move 37 (some say he stepped out to smoke around move 36), so never show 离席 as a reaction. Use “李世石十几分钟后才回应” (12–15 min is consistent across sources). 10^360 is a rough estimate of game sequences (“粗略一算”), not “all possibilities”. The 79 % is from the tic-tac-toe demo; never caption it as a Go result. The tree behind move 37 is schematic (“示意”). AlphaGo plays the most-visited move; that is the honest link to move 37.


## As built (2026-10-18, `alphago_mcts.py`) — deviations and implementation notes

- **No voiceover.** Captions/cards carry the story. `Base.play()` enforces reading time for every caption or text card (a "track"): fully visible ≥ `style.reading_time(text)` + 0.5 × animation time under it (+ 1.5–2 s on the ahas), one reading target at a time; every track is logged to `alphago_mcts.captions.csv` / `alphago_mcts_short.captions.csv` and asserted. The `voice(key)` hooks are kept (no-ops without WAVs). Runtime is therefore caption-bound: 16:9 ≈ 4:55, vertical 79 s.
- **Opening (both cuts)** follows the coordinator's first-10-seconds brief: frame 0 = dim board crop with the move-37 point at 60 %/50 % (vertical 50 %/45 %); stone slam; "这是失误吗？" → "李世石十几分钟后才回应" (one clock sweep around the stone) → odds rolling 1/10 → 1/10000 ("AlphaGo 自己的估计" footnote) → teaser: the real 2000-round tree erupts from the stone (labelled 示意) and folds back → "答案，要从一副纸牌说起" → Ulam (养病时玩单人纸牌) → 100 deals (illustrative, labelled 示意) → π darts (seed 2016: 3.600, 3.000, 3.124, 3.146 → "≈ 3.14") → Monte Carlo (dice) → "随机试验 + 搜索树 = 蒙特卡洛树搜索" → "AlphaGo 用的正是它".
- **Go board** is abstract: 10 + 10 stones, move 37 drawn as a fifth-line shoulder hit next to a fourth-line white stone; not the game record. The callback uses the same composition with a schematic tree (示意) whose thickest branch ends at the stone.
- **10^360 vs 10^80** is drawn as digit strings (1 followed by 80 zeros vs 1 followed by 360 zeros) instead of log-scale bars; caption "粗略一算：约 10^360 种下法".
- **Counters** under each branch read "看 n / 赢 w" (X's view, draw = half; note "平局算半次赢"). Children appear as the root's ghost X's fly down (rounds 1–6 of the real run expand them in the order 5, 8, 1, 6, 3, 7); round 7 is the slow 选/长/试/传 demo (select 1 → expand O@7 → play-out → O wins), rounds 8–9 fast.
- **Beat 5** has no formula: teal 胜率 = w/n, yellow 好奇心加分 = √2·√(ln N / n) from the snapshot after round 9; rounds 10–12 then pick the tallest bar live (8, 6, 3 — asserted).
- **Beat 6**: 16:9 uses a tilting tree (branch slot width ∝ visits, min width; nodes shown if ≥ 2 % of visits; edge width ∝ visits) plus a horizontal visit chart fed by the counters; the vertical keeps equal slots with bars aligned under the branches. The move-6 bar climbs to 79 % in a slower last step (success sfx on that frame), caption "井字棋里，没人教它堵——它自己找到了". The win-rate labels were dropped (not narrated).
- **Re-hooks** (16:9): "那 AlphaGo 是怎么想的？", "可怎么知道哪里有希望？", "但还有个问题：先试哪一步？", "可围棋大得多——怎么办？".
- **Sound**: `style.sfx` with the README rules enforced (≤ 5 accents ≥ 20 s apart, events ≥ 0.8 s apart, no small sound within 0.5 s of an accent, ticks ≤ 4/s); ambient bed via `make_sfx.write_bed`. Render with `--disable_caching`.
- **Vertical** ends by returning to the opening composition (stone on the dim board) so replays loop.
- **Open fact question:** "70 多年前 / 70 年前" is relative to 2016 (1946 → 2016); from 2026 it is 80 years. The 16:9 card avoids it ("答案，要从一副纸牌说起"); the vertical caption "秘诀来自 70 年前的一副纸牌" still uses it.

### Round 2 (2026-10-18, coordinator review for virality)

- **16:9 opening (0–15 s):** hero layout. The board fills the left ~62 % (bleeds off left/top/bottom, smooth gradient into the right panel); move 37 lands at 36 % / 52 %. Right panel, left-aligned: tag (26 px) + "这是失误吗？" (96 px, Semibold, red ？) → "李世石 / 十几分钟后才回应" (72 px) → rolling hero number "≈ 1/10 → 1/10000" (160 px) with label (40 px) and footnote (24 px) → teaser "它在想什么？" (84 px) + "看完这棵树你就懂了" (44 px) while the real tree erupts from the stone. No bottom caption band before 15 s. The callback uses the same composition.
- **History detour ≈ 38.5 s:** 100-deals grid removed; 1946 card held only its reading time; π demo ≈ 10 s; nuclear card "后来用于核武器研究，代号：蒙特卡洛（一座赌城）"; "随机试验 + 搜索树 = 蒙特卡洛树搜索" ≤ 5 s, then "AlphaGo 用的正是它".
- **Fractal hairline fans** (250 lines, each tip fans again ×3 then ×2, opacity 0.2/0.11/0.08) out of the stone in "why brute force fails" and on a half-frame 19×19 board in the AlphaGo section; the policy network keeps 5 bright branches and fades the rest to 0.06; the narrow, deep AlphaGo tree is labelled 示意.
- **Counters** (看/赢): only the active branch is at full strength, the others dim to 0.3; grandchild counters disappear after their round.
- **Beat 3:** the two quick O-wins after move 1 play first, then the flaw game with its own caption.
- **Closing card** "这算思考，还是只是统计？" in both cuts (vertical: before the loop-back).
- **Vertical:** hook 84 px in the top third; the next two lines in the same hero style (72 px); "秘诀来自 1946 年的一副纸牌"; the tree folds back before the π square grows; the end card drops the "AlphaGo 的思考：" line (the title strip says it) to stay under 80 s.
- **Asserts added:** legibility floor (16:9 ≥ 24 px except 示意 at 20 px; vertical ≥ 32 px), no caption held more than 1 s past its need (except the closing card), banding check on decoded frames (luma steps ≤ 3/255). Gradients are single dithered RGBA images.

### Round 3 (2026-10-18, opening frame and cover polish)

- Glows are single smooth radial images (`glow_image`: same colour, outer radius 2.37 r and centre opacity as the old 7-disc stack, smoothstep falloff, 1/255 dither); the network-box glows are one soft signed-distance image each. The translucent tube around the teaser trunk is gone.
- Board dimming darkens stones by colour (they stay opaque), so grid lines and star points no longer show through the white stones.
- 16:9 hero blocks (and the move-37 callback block) are centred on the frame's vertical middle; the small tag glides above the two-line Lee Sedol card. "这是失误吗？" is one Text with a red full-width ？ (`t2c`); the text helpers assert there is no half-width "?".
- The first fan level ramps each line's opacity from ~0.05 at the root to full at 30 % of its length (11-stop stroke gradient per line, oriented along the line), so the 250 lines no longer merge into a solid core at the stone.
- Caption review: "一局牌能玩通的概率？直接算太难", "随机撒点：撒得越多，越接近 3.14", "这局赢了，可对手明明能赢" (shown when O misses its win), "一看胜率…", "二看好奇心…", "总分最高的先试，一样高就轮流", "策略网络：像棋感，先想好棋". Vertical: tag and footnote around the hook, "秘诀来自 1946 年的纸牌：/算不出来，就随机试很多次", "棋感：先想值得下的位置" and "大局观：一眼估计谁胜算大" as two separate captions, "井字棋：轮到 X 走 / 对手下一步就能连成斜线" as one two-line caption, 示意 visible on the callback tree from its first caption.

### Round 4 (2026-10-18, clarity pass from the user's own viewing; 16:9 has no length cap, vertical aims for ~95 s)

- **Stakes beat (16:9, new, after the teaser):** the tree folds back into the stone; the right hero panel shows “对手：李世石 / 拿过 18 个世界冠军” with the 18 counting up live (blue = the opponent, as O later). The block then shrinks to a dim header and a timeline appears: 1997 (“电脑赢了 / 国际象棋世界冠军”) → 2016 → a grey arrow +10 years (“专家预计：还要十年”); the arrow snaps back to 2016, the label is struck through, and a big “4 : 1” (4 orange, 1 blue) lands. Panel captions (42 px, left column): “1997 年，电脑就赢了国际象棋世界冠军” → “可围棋难得多：此前普遍认为，电脑赢职业棋手至少还要十年” → “2016 年 3 月，首尔：AlphaGo 4 : 1 获胜” → “第二局的第 37 手，是它最出人意料的一步” (ripple + halo on the stone), then the cards come out of the stone with “答案，要从一副纸牌说起”.
- **Vertical stakes:** after the hook, three top-third hero lines (64 px): “对手是拿过 18 个世界冠军的李世石” (18 counts up) → “此前普遍认为：电脑赢职业棋手还要十年” → “结果 AlphaGo 4 : 1 获胜” (big 4 : 1 in the lower band), then the teaser tree.
- **Solitaire → Monte Carlo (16:9):** “1946 年，数学家乌拉姆养病时，玩一种单人纸牌” (7-pile tableau) → “和电脑里自带的『纸牌』接龙是一类：把牌全部接完就算赢” (four foundation piles fill A→K from the tableau, ✓) → “能不能接完，很大程度看洗牌的运气” (deck riffle-shuffles twice, two deals side by side: one comes out ✓, one gets stuck ✗; cards are glyph-free) → “他想知道：随便洗一次，能玩通的概率是多少？” (big ？) → “可 52 张牌有约 8×10^67 种洗法，一种种算不完” (52 card backs appear one by one while the number of orders k! grows digit by digit, then becomes ≈ 8 × 10^67; `assert 8.0e67 <= 52! < 8.1e67`) → “他想到：不如真的洗 100 次、玩 100 局，数数成功几局” (10 × 10 ✓/✗ grid tagged 示意, counter “成功 n / 100”; n comes from `random.Random(1946)` with p = 0.3 and is never captioned as a real rate) → “这就是蒙特卡洛方法：算不出来，就随机试很多次” (the grid's border becomes the π square, the counter becomes the dart counters) → the π rain → “随机撒点：撒得越多，越接近 3.14”. The old cards “一局牌能玩通的概率？直接算太难” / “那就随机试很多次，数一数” are gone.
- **Vertical solitaire:** “秘诀来自 1946 年的纸牌接龙：洗法多到算不完” (a deck comes out of the stone and riffles three times, ~2 s) → “不如随机玩 100 局，数数成功几局” (the ✓/✗ grid fills, 示意) → the grid becomes the π square and the rain runs under the same caption to “圆周率 ≈ 3.14”.
- **Whose turn (both cuts):** a legend “✕ X：我们 / ○ O：对手” stays beside the board through the whole search section; a big “✕ 轮到 X” badge pops, pulses once and docks (16:9: above the legend; vertical: top right). 16:9 captions: “先看个小例子：井字棋” → “现在轮到 X 下” → “O 已经斜着连了两个：X 不堵，O 下一步就赢” (red threat line; the old red side label is gone) → “X 有 6 个空位可选，哪一步最好？” (ghost X's). In the tree the badge glides beside the root and turns into the root label “现在的局面（轮到 X）”, which leaves when X finally plays (move 6). Vertical captions: “现在轮到 X 下（我们）” → “O 斜着连了两个：不堵就输”; the badge sits beside the root and leaves when X plays.
- **Counters 看 → 试:** every counter reads “试 N / 赢 W” (two rows, W to one decimal, draw = 0.5). Callouts with a thin leader line, each its own caption step: round 1 “试 1 = 这一步模拟过 1 盘”; the first draw “赢 0.5 = 平局，算半盘”. Deviation: with seed 7, round 1 (move 5) is a loss (赢 0), so the 0.5 callout sits on round 3 (move 1, the first draw) instead of round 1; both are asserted against the counter values. The old note “平局算半次赢” is gone.
- **Renamed captions:** “每一步记一笔：试了几盘、赢了几盘”, “一看胜率：赢的盘数 ÷ 试的盘数”, “二看好奇心：试得少，多加分” (legends “赢得多，多试试” / “试得少，加分多”, “只试过 1 盘”), “跑起来：一开始每一步都试得差不多”, “试得越多，左下角的分支越粗”, chart title “每一步试了几盘”, new “79% 的尝试都给了这一步” before the aha, “最后：下试得最多的那一步”, “它总下树里试得最多的那一步”; vertical “一开始，每一步试得差不多 / 试得越多，左下角分支越粗” (merged into one two-line caption to save time), “它总下试得最多的那一步：第 37 手就是这样选出来的”. Remaining 看 are not counts (看完, 看输赢, 先看个小例子, 一看 / 二看, 看洗牌的运气, 不看人类棋谱).
- **Lengths:** 16:9 5:47 (346.8 s, no cap). Vertical 98.6 s after trims: hero swaps cross-fade (t_on at 60 %), the teaser tree folds while the deck comes out, π rain in three steps, the two tilt captions merged, aha on the 79 % caption 1.3 s, shorter loop-back. Reaching 95 s would need dropping a caption (e.g. the end card “一棵会自己长偏的树”, about 4.8 s).
- Cover plates unchanged (teaser tree and stone did not move), so the 4K plates were not re-exported.

### Round 5 (2026-10-18, user viewing of v4: slower play-outs, why move 37 was found, a beginner's pass)

- **Slow, readable play-outs.** Every slow 试 (想法一's six games, tree rounds 1–3, the round-7 选/长/试/传 demo, the vertical's 试 step) plays the real seeded game one move per 0.45 s in the X/O colours. On an enlarged copy of the leaf (rounds 1–3: 2.0 units = 2.2× the child; round 7: 1.75 units; vertical: 2.4×, all linked by a thin dashed line), or on the big board in 想法一. At the end a rounded capsule is drawn around the winning three (teal when X wins, red when O wins; round 7's last O move completes two lines, so two capsules), a draw shows the 平 stamp and no capsule, then the result stamp, then a 1.2 s hold on the full final board. The board then shrinks to a thumbnail beside the leaf (想法一: next to its tally chip) and stays through 传; it leaves when the next round starts (想法一: when that candidate's games are done). Capsule cells are asserted to be a real line of the winner on the final board computed from the seeded trace. Rounds 4–6 and the fast-forward stay fast. A caption that has been read leaves during a long play-out (`drop_read_cap`, `hold`), so the over-hold assert still holds.
- **π, why × 4 (16:9).** After the 5000 darts: the dots dim, the square and the quarter disc are shaded, "π/4" on the disc, caption "四分之一圆占正方形面积的 π/4"; then "所以：落进圆里的比例 × 4 ≈ π" with the live line "3933 ÷ 5000 × 4 = 3.146" (numbers fly from the counters; asserted against the seeded darts), then "随机撒点：撒得越多，越接近 3.14". Vertical: "落进圆里的比例 × 4 ≈ π" over its rain.
- **The tilt (16:9 beat 6), cause made visible.** The right-hand list is gone: each child has a column under it (visits bar with "试 N", and "胜率 xx%" under the child), live, from the counters. Four stops, one caption each, values asserted against the snapshots: 20 "一开始：每一步都试得差不多"; 50 "可其他 5 步都没堵住斜线：O 下一步就赢，几乎盘盘输" (a red O completes the diagonal on move 1's mini board; the five readouts are 8–22 %); 300 "左下角堵住了：大多下成平局，胜率最高" (move 6's readout boxed, 46 %); 2000 "胜率高 → 加分高 → 试得更多 → 越试越多" (move 6's column glows, 试 1584, 79 % beside the bar, 胜率 48 %; riser climbs through the step, success lands on it), then "井字棋里，没人教它堵——它自己找到了". Then the columns clear, the tree tilts and grows (300 → 2000) with the legend "橙点：X 的下一步 / 蓝点：O 的下一步" and "试得多的分支，就往下想得更深：X 堵住后，O 怎么应，X 再怎么下"; the first two nodes of the main path become mini boards (O blocks at 3, X blocks at 5; asserted to be the most-tried nodes). Payoff: "最后：下试得最多的那一步" — the real board appears big on the right with the red threat line; X lands bottom-left, the line breaks, "正是唯一不会输的那一步". That board becomes the Go board in beat 7. The old "79% 的尝试都给了这一步" caption is replaced by the 2000 stop.
- **Vertical tilt:** bars stay aligned under the branches, with teal win-rate readouts under the baseline; captions "一开始，每一步试得差不多" → "其他 5 步没堵住：几乎盘盘输" → "左下角堵住了：胜率最高，越试越多" (with the last step).
- **The two networks (16:9 beat 7).** A 选 → 长 → 试 → 传 badge row sits top right. "策略网络：看过 3000 万个人类高手的局面，学会猜高手会下哪" → "它升级了「选」：好奇心加分优先给看起来像好棋的走法 → 树变窄" (选 lights, the fan dims to five bright lines) → "价值网络：从大量自我对弈里学会，看一眼局面就估出谁赢面大" (the long rollout line) → "它升级了「试」：不用每次都随便下完，一眼估胜率 → 能想得更深" (试 lights, the rollout is replaced by the gauge; the "2016 版：价值网络 + 快速试下一起估" note stays) → "树变得又窄又深" → AlphaGo Zero. Facts: SL policy net on 30 million KGS positions; value net on 30 million self-play positions. Vertical: "棋感 = 学过人类高手棋谱：升级「选」", "大局观 = 一眼估胜率：升级「试」".
- **Why move 37 was found (16:9 callback).** After "回到第二局的第 37 手", a schematic panel (示意) on the right reuses the 选 bars (teal 胜率 + yellow 好奇心加分) for 走法 A, 走法 B and 第 37 手, with a "总共试了 N 盘" counter; the move-37 branch on the board starts hair-thin. "策略网络觉得：人类几乎不会下这里（约万分之一）" (a small "≈ 1/10000" echo of the hook flashes) → "可试的总盘数越多，好奇心加分越大：冷门的一步也会被试到" (N climbs, the yellow bar grows past the others, the branch is tried) → "一试才发现：眼前局部看似吃亏，往后推演，胜率反而更高" (a local patch dims, "看似吃亏"; a soft teal glow spreads over the board, "全局"; the teal bar rises) → "胜率高 → 试得更多 → 这根树枝越长越粗" (the branch thickens) → "它总下树里试得最多的那一步" → … as before. No side, territory count or centre is named. Vertical: "策略网络觉得人类几乎不会下（约万分之一）" and "但好奇心加分让它去试：一试，胜率反而更高" over the thin-then-thick branch. Move 37 is never given the nickname that belongs to Lee Sedol's move 78 in game 4.
- **Vertical length** (cap ~105 s): to make room, the short cut drops the end card "一棵会自己长偏的树", the aha "井字棋里…" (the 2000-stop caption carries the point) and the intro "围棋大得多：加两样直觉" (the board morph says it); the policy caption now arrives with the fan; two riffles instead of three; tighter hero cross-fades.
- **Lengths:** 16:9 7:27 (447.2 s, no cap); vertical 106.3 s. Also fixed: the "2016 版：价值网络 + 快速试下一起估" note was clipped at the right edge (now right-aligned inside the frame).
