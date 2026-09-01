# hot-news — RAG 加持的热点营销内容生成任务

`llamaindex-pse` 的一个 task，把「热点新闻 → 合规营销文案」跑通在 PSE 流水线里：
**Planner(选题) → Specialist(写作) → Evaluator(首轮 LLM 评审) + 每轮确定性合规核查 → Fix 循环**。

## 为什么是 llamaindex-pse

- **RAG grounding 防编造**：新闻目录先建成向量索引，写作前把真实新闻片段作为上下文喂给 Specialist，
  模型从源头就「看不到原文之外的内容」——这是热点场景（外部易变输入）对抗幻觉的最优形态。
- **verify_fn 双保险**：合规核查（违禁词/平台格式/AI 标注/事实对照）作为确定性函数每轮硬查，
  不依赖 LLM 自觉；语义擦边类问题才交给 Evaluator 的 LLM 首轮评审。

## 目录结构

```
hot-news/
├── run.py            # 入口：建索引 → build_workflow → verify_fn → 保存
├── compliance.py     # 确定性合规引擎（verify_fn 实现）：违禁词/格式/AI标注/事实对照
│                     #   同时是「平台限值 + 拉黑词」的单一数据源（发布端/抓取端共用）
├── prompts/
│   ├── planner.md     # 选题规划（含真实锚定规则）
│   ├── specialist.md  # 写作（含防编造铁律，从 crewai-pse specialist 思路改造）
│   ├── evaluator.md   # 评审（首轮 LLM，文风/结构/合规初判）
│   └── fix.md         # 修正（Fix 循环规则；无此文件时 workflow 回退 legacy 简历规则）
├── news/            # fetch_news.py 抓取落盘（weibo/qbitai/infoq/kr36/sspai 子目录，RAG grounding 源；其中 qbitai+infoq 为 AI 专属源，RSS 摘要过短时自动抓取文章正文）
├── articles/        # 成稿输出目录（文件名带生成时间戳，见下）
├── scheduler/       # launchd 定时刷新（refresh.sh + install.sh + plist 模板，见「定时刷新」）
├── publish/         # 三个平台的发布器（Playwright）+ 共享 common.py + 品牌多图 cover_gen.py
├── logs/            # 定时抓取日志
└── README.md
```

## 用法

```bash
# 标准：提供抓取好的新闻目录（RAG grounding 最强，事实可对照）
uv run python run.py --topic "AI 新规落地" \
    --news-dir /path/to/news --platform xiaohongshu --category tech_ai

# 降级：仅给主题，无新闻源（verify 会警告，无事实对照）
uv run python run.py --topic "AI 新规落地" --platform douyin --provider agnes
```

### 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--topic` | 热点主题（留空则自动选题：**优先**微博热搜中「语料有据」的 AI/科技相关话题；若全部热搜在语料无据则回退 **qbitai/infoq/kr36/sspai** 新闻标题兜底） | 空（自动选题，AI 优先且语料有据优先） |
| `--selection` | 自动选题策略：`random`(默认, 有据热搜中随机，同快照产出不同选题) / `top`(有据热搜中最高热度) | `random` |
| 拉黑词 | 自动选题/兜底/抓取三层统一过滤：标题命中「鸿蒙/HarmonyOS/Harmony」即跳过（改 `run.py` 的 `EXCLUDED_TOPICS` 与 `fetch_news.py --exclude`） | 鸿蒙系列 |
| `--news-dir` | 已抓取新闻目录，RAG grounding 源 | 空（降级） |
| `--provider` | `deepseek` / `agnes` / `scnet-kimi` / `scnet-minimax` | `agnes` |
| `--platform` | `xiaohongshu` / `douyin` / `zhihu` / `toutiao` | `xiaohongshu` |
| `--category` | `tech_ai`(默认最宽松) / `beauty` / `food` / `education` / `finance` / `medical` / `ecommerce` | `tech_ai` |
| `--top-k` | RAG 检索 top-k | `8` |
| `--rebuild` | 强制重建索引（默认无需：索引按新闻文件集指纹自动失效重建） | 否 |
| `--out-dir` | 成稿输出目录 | `tasks/hot-news/articles` |

### 成稿命名与元信息

- 输出文件名带生成时间戳，避免覆盖历史成稿：
  `articles/hot_news_{platform}_{provider}_{YYYYMMDD_HHMMSS}.md`
- 文件头部自动写入 YAML frontmatter。`topic` 跟随成稿实际标题（保证与内容一致）；
  初始热搜话题若与之不同则记入 `heat_topic` 供追溯。
  自动选题来自「新闻标题兜底」时（或显式 `--source-url`）会写入 `source_url`，
  供发布端「内容来源」标注使用：

  ```markdown
  ---
  topic: "交给AI主编20分钟，多份成果齐全"
  platform: xiaohongshu
  category: tech_ai
  provider: agnes
  generated_at: "2026-08-30 21:25:02"
  source_url: "https://www.qbitai.com/2026/08/481372.html"
  heat_topic: "华为新三折叠"
  ---
  ```

## 合规性（核心）

`compliance.py` 实现确定性核查，分四类：

1. **违禁词**（按品类合并通用+定向词表）：极限词（最/第一/绝对/百分百）、导流词（微信/加我/私信领）、
   虚假诱导（秒杀/倒计时/错过再无）；金融/医疗为高危品类，词表从严。
2. **平台格式**：小红书标题 ≤20 字、正文 ≤1000 字（各平台阈值见 `PLATFORM_LIMITS`）。
3. **AI 标注**：AI 生成内容须含「AI 辅助创作」标记（2026 新规强制），但**不写入本 task 成稿正文**——
   由发布端浏览器自动化层在发布时自动勾选（见「发布端」章节），故合规核查中该标记不作为硬性失败项。
4. **事实对照**：数字/日期断言须在新闻原文（`--news-dir` 语料）中出现，否则判编造。

语义暗示类擦边（如「性价比较优」、无数据支撑拉踩）交给 Evaluator 的 LLM 首轮评审，不在确定性引擎内。

**话题标签结构按平台分治**：只有小红书/抖音类「文末 #标签 + 互动收尾」是硬性结构检查；
知乎/头条的话题由编辑器/发布端单独管理（知乎编辑侧「添加话题」、头条正文首行 `#话题#` 闭环），
合规核查不强制文末标签，避免生成 loop 为无关平台写虎头蛇尾收尾。

**内容来源标注**：正文尾部按平台轻重标注——小红书用轻标注「资料参考 @量子位」，
头条/其他带原文链接；知乎正文不放来源行（交给「创作声明」人工声明）。
数据取自成稿 frontmatter 的 `source_url`（run.py 自动选题兜底或 `--source-url` 写入）。

## 发布端（小红书，草稿优先 + 绝不自动发布）

`publish/publish_xiaohongshu.py` 是 Playwright 发布器：headful 驱动本机 Chrome，从
「发布图文笔记」入口进入真实图文编辑器，自动生成并上传**品牌多图**，再填标题/正文/话题，
**绝不自动点「发布」**。登录 cookie 用 `storageState` 持久化到 `publish/storage/`（已 .gitignore）。

**图片（`publish/cover_gen.py`，基于 Pillow + 系统苹方字体）**：自动产出 1 张品牌封面
（渐变底 + 标题 / 话题 / `erishen.cn · AI 观察` 署名）+ 至多 3 张正文要点卡，文案取自
**已经过合规校验的文章原句**，不引入新内容；多图更贴近小红书图文氛围。
`COVER=c.jpg` 可用你自己的图替换封面（要点卡照常生成）。

```bash
# 0) llamaindex-pse/.env 配好待发布文章：
#    HOT_NEWS_ARTICLE=tasks/hot-news/articles/xxx.md   （之后所有命令缺省读它，ARTICLE= 可覆盖）

# 1) 首次：扫码登录一次
make hot-news-publish-login

# 2) 离线校验标题(≤20字)/正文字数/话题
make hot-news-publish-check

# 3) 填好内容停编辑器，人工核对后在浏览器里自己决定发布（默认最安全）
make hot-news-publish
#    SAVE_AS_DRAFT=1   点「发布笔记」走官方存草稿机制（往下看）
#    DRY_RUN=1         填好后标注 dryrun 截图并停在编辑器（与默认行为一致，便于排障留痕）
#    COVER=c.jpg       用你自己的 3:4 图替换封面
```

**关于「存草稿」的重要事实（2026-08 实测）**：小红书新版图文编辑器**没有「保存草稿」按钮**。
官方草稿机制是：点「发布笔记」时若「AI 内容来源声明」未勾选，会被拦截并**自动存入草稿箱**——
`SAVE_AS_DRAFT=1` 就是利用这一机制（脚本先尽力勾选 AI 声明；找不到则拒绝，除非 `ALLOW_NO_AI=1`）。
默认（不带 `SAVE_AS_DRAFT`）则完全不点任何按钮，填好后浏览器停留等人工。

两点须知：

- **行文以平台实测为准**：小红书发布页是频繁改版的 SPA，脚本选择器都集中在
  `publish_xiaohongshu.py` 顶部常量区；改版后只需维护那里。直连 `/publish/publish` 会命中错误页，
  必须走「首页 → 发布 → 发布图文笔记」入口（见文件头注释）。
- **风控与声明兜底**：遇到验证码/滑块在浏览器里人工完成（默认 headful）；发布 AI 内容前
  请人工确认「内容来源声明」（脚本尽力自动处理，失败时有截图留痕）。

## 发布端（知乎，文章格式，品牌图全自动）

`publish_zhihu.py` 走知乎文章编辑器 `zhuanlan.zhihu.com/write`：扫码登录后自动
**填标题 + 填正文 + 上传 3 张要点卡入正文 + 封面区单独上传品牌封面 + 添加首要话题 +
尽力勾选「创作声明」（AI 来源）**，浏览器保屏交人工发布。

```bash
make hot-news-publish-zhihu-login   # 知乎App/微信扫码或手机验证码，一次
make hot-news-publish-zhihu-check   # 离线校验（知乎标题上限 100 字，建议 ≤30）
make hot-news-publish-zhihu         # 填稿+插图+封面+话题后保持页面，人工按「发布」
```

**知乎是三个平台里自动化完成度最高的**：正文图片按钮 → 本地 file input 直传要点卡、
封面走常驻封面区的 `input[type="file"][class*="UploadPicture"]`（可直接 set_input_files）、
话题用「添加话题」联想自动选中。注意几点（实测 2026-08）：

- **风控保护**：脚本带「精扫菜单项（≤80）+ 操作间隔」节奏，避开整页大范围扫描触发
  「请求存在异常（40362）」。若仍被风控，网页登录会被临时限制——等 30 分钟~数小时、
  App 端验证账号后，用 `make hot-news-publish-zhihu-login` 重新登录再跑。
- **正文不放来源行**：来源信息由人工在发布时通过「创作声明」声明（脚本尽力自动勾选；
  未命中时会打印菜单候选日志，供校准选择器）。来源链接存于文章 frontmatter `source_url` 仅供追溯。
- **草稿是会话内本地保存**，账号草稿箱不落盘 → 填完别关页面，发布动作始终人工。
- `^C` 接管：按下后 5 秒内按回车可保持浏览器等人发布，否则自动关闭退出。
- 排障：`make hot-news-publish-zhihu-cover-probe`（或 `python tasks/hot-news/publish/publish_zhihu.py cover-probe`）
  会打开写作页并转储封面区真实 DOM，用于校准 `set_cover`（封面结构变动后用它摸底）。

## 发布端（今日头条，可用但能力有限）

`publish_toutiao.py` 走头条「文章(图文)」编辑器：扫码登录（头条App）后直达
`/profile_v4/graphic/publish`，自动填标题（textarea，2~30 字）与正文（contenteditable），
**填好保屏交人工，绝不自动发布**（页面底部只有「定时发布 / 预览并发布」）。

```bash
make hot-news-publish-toutiao-login   # 头条App扫码一次
make hot-news-publish-toutiao-check   # 离线校验（头条标题建议 ≤30 字）
make hot-news-publish-toutiao         # 填好标题/正文保持浏览器，人工自己发布
```

**实测限制（2026-08）**：
- 头条「文章」不提供本地图片上传（封面「单图/三图」走头条图库、正文无插图按钮）→ 品牌多图无法自动放置，封面/插图只能人工在图库挑；
- 编辑器的“草稿将自动保存”在自动化输入下不落草稿箱（实测关闭后 0 条）→ **禁止关闭该浏览器**，否则内容丢失；
- 话题转用头条格式：仅把首个话题以 `#话题#` 闭环放正文首行（小红书式文末 `#a #b` 在头条不参与聚合）；
- 我们的 ~600 字短文对头条文章偏薄，且 AI 内容声明需人工在发布时处理。

结论：头条更适合深度长文 + 平台图库素材的形态；营销短文自动发布仍以小红书为主力推手。

## 定时刷新（launchd）

微博热搜是瞬时数据（15 分钟就换），固定人手抓会忘。仓库内置 launchd 调度：

- **`weibo`**：每 **15 分钟**抓一次微博热搜，覆盖写 so 旧标题不累积（`refresh.sh` 默认模式）。
- **`full`**：每天 **09:00** 全源抓取（weibo+kr36+sspai+qbitai+infoq），RSS 按 `--keep-days 7` 清理 7 天前旧文、按 `--max-age-days 2` 丢弃发布超 2 天的旧闻。

```bash
make hot-news-cron-install      # 生成 plist 并 launchctl 加载；weibo 立即跑首轮，full 明早 9 点首次
make hot-news-cron-status       # 查看两个 job 状态 / 最近退出码
make hot-news-cron-uninstall    # 卸载（保留已生成的 plist 与日志）
make hot-news-refresh           # 手动跑一遍（REFRESH_MODE=full 全源）
```

日志：`tasks/hot-news/logs/refresh-{mode}-YYYYMMDD-HHMMSS.log`（每模式保留最近 30 份）+ launchd 收尾日志。
调度器脚本：`tasks/hot-news/scheduler/{refresh.sh, install.sh, *.plist.tpl}`。安装脚本会自动探测 `uv` 与 `bash` 路径并写入 plist 环境（launchd 默认 PATH 不含它们，勿手改模板占位符）。

## 注意

- 自动选题只选「语料有据」的热搜（weibo 纯标题不算），避免选中无信源话题后转写别的故事、导致 topic 与内容错位；`--topic` 显式指定时不强制，靠 planner/specialist 的「无据换角度」规则 + frontmatter 的 `heat_topic` 兜底。
- `news/` 快照由 launchd 定时刷新（见上），无需每次手动 `make hot-news-fetch`；首次使用或修改了抓取开关后记得 `make hot-news-cron-install`。
- 纯 `--topic` 降级模式无事实对照，仅适合练手；正式产出务必提供 `--news-dir`。
- 浏览器自动化发布处于平台 ToS 灰色地带，有封号风险，建议前期「生成 → 人工确认 → 手动发」。
