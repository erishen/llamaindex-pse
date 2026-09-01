"""hot-news 合规核查 — 确定性规则引擎（PSE verify_fn 实现）。

设计定位（对应 2026 小红书/抖音平台新规）：

营销类内容的核查几乎全是**确定性规则**，正是 PSE `verify_fn` 的主场。
本模块把「硬规则每轮跑、LLM 只审首轮」的原则落到代码：
  - 硬性违禁词 / 平台格式 / AI 标注 / 事实对照  → 确定性，每轮 `verify_fn` 硬查
  - 语义暗示类擦边（如「性价比较优」、无数据支撑拉踩） → 交给 evaluator 的 LLM 首轮评审

四类核查：
  1. 违禁词（按品类合并通用+定向词表）：极限词/导流词/虚假诱导/金融医疗高危词
  2. 平台格式（确定性数值）：小红书标题≤20字、正文≤1000字 等
  3. AI 标注：AI 生成内容须含「AI 辅助创作」标记，但由发布端浏览器自动化层按 2026 新规
     自动勾选，本 task 成稿不写入正文（见 README「发布端」），故合规核查中不作硬性失败项
  4. 事实断言 vs 新闻原文（确定性 RAG 对照）：数字(含无单位裸数字如热度)/日期/人名须在 news_corpus 出现

品类（category）决定词表松紧：tech_ai（默认，最宽松）< beauty/food/education <
finance/medical（高危，《广告法》级罚款，从严）。

API：
  load_banned_words(category) -> list[str]
  verify_compliance(state, category, platform) -> (bad, ok)
  make_compliance_verifier(category, platform) -> (state) -> (bad, ok)   # 供 build_workflow 注入
"""

import re
from typing import Callable

# ─────────────────────── 通用违禁词表（与品类无关）───────────────────────

# 《广告法》极限词（绝对化表述）——命中即违规
GENERIC_SUPERLATIVE = [
    "第一", "唯一", "最佳", "绝对", "百分百", "100%", "永久", "零风险", "彻底",
    "顶级", "极致", "国家级", "世界级", "第一品牌", "独一无二", "史无前例",
    "万能", "最牛", "最强", "最优", "包治", "根治", "100%有效", "绝对化",
    "100%安全", "史上最", "全网最",
]

# 虚假/诱导词
GENERIC_FALSE_INDUCEMENT = [
    "秒杀", "倒计时", "假一赔十", "错过再无", "最后一天", "限时免费",
    "稳赚", "稳赚不赔", "必定", "一定涨", "绝对赚", "点击领取", "免费送",
    "0元购", "领红包", " guaranteed", " guaranteed pass",
]

# 导流/站外词（小红书/抖音重点管控：平台外引流，2026 重点打击）
GENERIC_DIVERSION = [
    "微信", "薇信", "VX", "薇", "卫星", "私信领", "加我咨询", "加微信",
    "公众号", "关注我领", "一键三连", "求点赞", "互关互赞", "互粉",
    "加我", "扫码", "二维码", "私聊", "加好友", "加V",
]

# ─────────────────────── 按品类违禁词（定向收紧）───────────────────────
CATEGORY_WORDS: dict[str, list[str]] = {
    # 技术/AI 个人 IP（默认，最宽松）
    "tech_ai": [
        "月入过万", "财务自由", "躺赚", "副业刚需", "不学就废", "一夜暴富",
    ],
    # 美妆
    "beauty": [
        "美白", "祛斑", "根治", "医用", "药妆", "纯天然", "无副作用",
        "百分百吸收", "速效", "一洗白", "永久脱毛",
    ],
    # 食品
    "food": [
        "降糖", "排毒养颜", "增强免疫力", "抗癌", "治疗", "药用",
        "纯天然无添加", "减肥神器", "燃脂", "治病",
    ],
    # 教育
    "education": [
        "包过", "保上岸", "零基础速成", "不过退费", "提分神器", "秒懂", "速成",
    ],
    # 金融（高危）
    "finance": [
        "稳赚", "保本", "无风险", "高收益", "荐股", "内幕", "杠杆", "暴富",
        "财富自由", "保底", "稳赢", "稳涨", "必涨",
    ],
    # 医疗（高危）
    "medical": [
        "治愈", "根治", "特效", "神药", "无副作用", "包治", "疗效百分百",
        "一针见效", "中医世家", "祖传秘方", "药到病除",
    ],
    # 带货/通用（导流+虚假从严）
    "ecommerce": [
        "最后X件", "厂家直销", "亏本甩卖", "跳楼价", "全网最低",
        "史上最低", "错过等一年", "最后机会",
    ],
}

# 平台格式约束（确定性数值，单位：字）——单一数据源：发布端 check 与 README/Makefile 均以此为准。
#   zhihu 标题上限 100（编辑器实测，发布端口径一致）；toutiao 标题 2~30、正文 5000（发布端实测校验口径）。
PLATFORM_LIMITS = {
    "xiaohongshu": {"title_min": 0, "title_max": 20, "body_max": 1000},
    "douyin": {"title_min": 0, "title_max": 30, "body_max": 2000},
    "zhihu": {"title_min": 0, "title_max": 100, "body_max": 20000},
    "toutiao": {"title_min": 2, "title_max": 30, "body_max": 5000},
}

AI_LABEL = "AI 辅助创作"

# 热搜身份类表述：只有话题真实在微博热榜时才允许使用；来源为「新闻标题兜底」时禁用
HOT_CLAIM_PHRASES = [
    "登上热搜", "冲上热搜", "霸榜热搜", "霸占热搜", "热搜第一",
    "登上微博热搜", "冲上微博热搜", "登顶热搜",
]

# 支持品类 / 平台白名单（run.py argparse 校验用）
CATEGORIES = list(CATEGORY_WORDS.keys())
PLATFORMS = list(PLATFORM_LIMITS.keys())

# 拉黑选题统一入口（自动选题/媒体兜底/抓取三层共用；fetch_news --exclude 默认值与
# run.py 自动选题过滤都读这一个常量，改这一处即可三处生效）
EXCLUDED_TOPICS = ["鸿蒙", "harmonyos", "harmony"]


def load_banned_words(category: str) -> list[str]:
    """合并通用违禁词 + 指定品类词。category 未知时回退到 tech_ai。"""
    words = list(GENERIC_SUPERLATIVE)
    words += GENERIC_FALSE_INDUCEMENT
    words += GENERIC_DIVERSION
    cat = CATEGORY_WORDS.get(category, CATEGORY_WORDS["tech_ai"])
    words += cat
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _extract_title(artifact: str) -> str:
    """从 Markdown 提取首个 # 标题文本。"""
    m = re.search(r"(?m)^#\s+(.+?)\s*$", artifact)
    return m.group(1).strip() if m else ""


def _check_standalone_numbers(artifact: str, corpus: str, bad: list[str], ok: list[str]) -> None:
    """独立大数（无单位的指标，如热度/播放量/金额）须在新闻原文出现，防编造。

    `_check_factual_claims` 既有正则只匹配『数字+单位』，会漏掉
    「热度达1100680」这类无单位裸数字；此处补强：要求 5 位及以上纯数字
    必须能在 news_corpus 中找到，否则视为可能编造。
    """
    corpus_norm = re.sub(r"[\s,]", "", corpus)
    # 5 位及以上纯数字（热度/金额/数量级指标）；4 位年份与带单位的已在别处处理
    for m in re.finditer(r"(?<!\d)(\d{5,})(?!\d)", artifact):
        num = m.group(1)
        if num in corpus_norm:
            ok.append(f"数值「{num}」在新闻原文中存在")
        else:
            bad.append(
                f"数值「{num}」未在新闻原文中找到，可能编造"
                f"（须能在抓取新闻中核实，或删除该具体数字/改用原文表述）"
            )


def _check_factual_claims(artifact: str, corpus: str, bad: list[str], ok: list[str]) -> None:
    """数字/日期断言须在新闻原文中出现（RAG 事实对照）。"""
    corpus_norm = re.sub(r"[\s,]", "", corpus)

    # 数字+单位（金额/百分比/数量/时间跨度）
    num_pat = r"(\d+(?:\.\d+)?\s*(?:%|万|亿|元|万元|亿元|倍|人|个|条|起|岁|年|月|日|万次|亿次|次))"
    for c in re.findall(num_pat, artifact):
        c_norm = c.replace(" ", "")
        if c_norm and c_norm in corpus_norm:
            ok.append(f"数字断言「{c}」在新闻原文中存在")
        else:
            bad.append(f"数字断言「{c}」未在新闻原文中找到，可能编造（须能在抓取新闻中核实）")

    # 日期 YYYY.MM.DD / YYYY年MM月 / YYYY-MM-DD
    date_pat = r"(\d{4}\s*[年./-]\s*\d{1,2}\s*[月./-]?\s*\d{0,2}\s*[日]?)"
    corpus_date_norm = re.sub(r"[\s年./-]", "", corpus)
    for d in re.findall(date_pat, artifact):
        d_norm = re.sub(r"[\s年./-]", "", d)
        if d_norm and d_norm in corpus_date_norm:
            ok.append(f"日期「{d}」在新闻原文中存在")
        else:
            bad.append(f"日期「{d}」未在新闻原文中找到，可能编造")

    # 裸数字（无单位具体指标，如热度/播放量/金额）——补强既有『数字+单位』正则的盲区
    _check_standalone_numbers(artifact, corpus, bad, ok)


def _check_hashtags_and_hook(
    artifact: str,
    bad: list[str],
    ok: list[str],
) -> None:
    """话题标签唯一且仅在文末、格式紧凑、结尾必须有互动引导（确定性格式检查）。

    - 剥离 YAML frontmatter 与首个 H1 标题（二者均以 # 起头，不算标签）。
    - 文末「标签块」= 结尾连接的一段含 # 令牌的行。
    - 标题之外的正文任何位置出现 # 令牌（如标题下重复标签行）→ 位置违规。
    """
    body = artifact
    if body.lstrip().startswith("---"):
        parts = body.lstrip().split("---", 2)
        if len(parts) >= 3:
            body = parts[2]

    lines = [ln.rstrip() for ln in body.splitlines()]
    non_empty = [(i, ln) for i, ln in enumerate(lines) if ln.strip()]
    if not non_empty:
        bad.append("正文为空，无法检查话题标签与互动收尾")
        return

    title_idx = non_empty[0][0]  # 首个内容行 = H1 标题，跳过
    tag_re = re.compile(r"#\s*([^\s#]+)")

    def names(line: str) -> list[str]:
        return tag_re.findall(line)

    block_start = None
    for i in range(len(non_empty) - 1, -1, -1):
        idx, line = non_empty[i]
        if idx == title_idx:
            break
        if names(line):
            block_start = i
        else:
            break
    block_lines = non_empty[block_start:] if block_start is not None else []

    if not block_lines:
        bad.append("正文结尾缺少话题标签行（如 #AI #大模型）")
        return

    block_names: list[str] = []
    for _, line in block_lines:
        block_names.extend(names(line))

    stray = [
        line for i, line in non_empty
        if i != title_idx and i < block_start and names(line)
    ]
    if stray:
        bad.append(
            f"话题标签出现在正文而非文末（标签必须只出现在结尾）：「{stray[0].strip()[:40]}」"
        )
    else:
        ok.append("话题标签位置正确（仅出现在文末）")

    loose = [
        (i, line)
        for i, line in non_empty
        if i != title_idx and re.search(r"#\s+[^\s#]+", line)
    ]
    if loose:
        bad.append(f"标签使用非紧凑写法（`# ` 后带空格）：「{loose[0][1].strip()[:40]}」")
    else:
        ok.append("标签写法紧凑（#XX）")

    seen, dup = set(), set()
    for n in block_names:
        key = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", n).lower()
        if key in seen:
            dup.add(key)
        seen.add(key)
    if dup:
        bad.append(f"话题标签重复出现: {', '.join(sorted(dup))}")
    else:
        ok.append("文末标签无重复")

    hook_line = non_empty[block_start - 1][1] if block_start > 0 else ""
    if any(mark in hook_line for mark in ("？", "?", "评论", "聊聊")):
        ok.append("结尾含互动引导句")
    else:
        bad.append("结尾缺少互动引导句（如「你遇到过吗？评论区聊聊」，标签不算互动）")


def verify_compliance(
    state: dict,
    category: str = "tech_ai",
    platform: str = "xiaohongshu",
) -> tuple[list[str], list[str]]:
    """PSE verify_fn 实现：确定性合规核查。返回 (bad, ok)。

    state 来自 workflow.evaluator 注入的 state_dict：
      - artifact: 当前产物
      - task_data.news_corpus / rag_context: 事实来源（新闻原文）
      - attempts: 当前轮次
    """
    artifact = state.get("artifact", "") or ""
    task_data = state.get("task_data", {}) or {}
    news_corpus = task_data.get("news_corpus", "") or state.get("rag_context", "") or ""
    attempts = state.get("attempts", 0)

    bad: list[str] = []
    ok: list[str] = []

    # 0) 热搜身份核查：非热搜来源（新闻标题兜底）的话题不得声称「登上热搜」，防编造传播事实
    heat_src = (task_data.get("scan_result", {}).get("heat_topic_source", "") or "")
    hot_hits = [p for p in HOT_CLAIM_PHRASES if p in artifact]
    if heat_src == "news" and hot_hits:
        bad.append(
            f"话题来源为新闻头条兜底（非实时热搜），却出现热搜身份声明: "
            f"{', '.join(hot_hits)} —— 属编造传播事实，须删除或改为普通新闻背景表述"
        )
    elif hot_hits:
        ok.append("热搜身份表述来源校验通过（话题在热榜或未使用热搜身份声明）")
    else:
        ok.append("无热搜身份类表述")

    # 1) 违禁词扫描
    banned = load_banned_words(category)
    found = [w for w in banned if w and w in artifact]
    if found:
        bad.append(f"命中违禁词（{category} 品类）: {', '.join(found)} —— 须替换为合规表述")
    else:
        ok.append("无违禁词")

    # 2) 平台格式
    limits = PLATFORM_LIMITS.get(platform, PLATFORM_LIMITS["xiaohongshu"])
    title = _extract_title(artifact)
    if title:
        if len(title) > limits["title_max"]:
            bad.append(
                f"标题 {len(title)} 字超过 {platform} 上限 {limits['title_max']} 字: 「{title}」"
            )
        elif len(title) < limits.get("title_min", 0):
            bad.append(
                f"标题 {len(title)} 字低于 {platform} 下限 {limits['title_min']} 字: 「{title}」"
            )
        else:
            ok.append(f"标题 {len(title)} 字在 {platform} 限内（{limits.get('title_min', 0)}~{limits['title_max']}）")
    else:
        bad.append("未检测到标题（# 一级标题缺失）")
    if len(artifact) > limits["body_max"]:
        bad.append(f"正文 {len(artifact)} 字超过 {platform} 上限 {limits['body_max']} 字")
    else:
        ok.append(f"正文长度 {len(artifact)} 字 ≤ {limits['body_max']} 字")

    # 3) AI 标注：由发布端浏览器自动化层按 2026 平台新规自动勾选「AI 辅助创作」，
    #    本 task 成稿不写入正文（见 README「发布端」章节），故此处不做硬性失败，
    #    仅作提示，避免 Fix 循环因草稿无标注而永不收敛。
    if AI_LABEL in artifact:
        ok.append(f"已含「{AI_LABEL}」标注（注：常规由发布端自动勾选，草稿可不写）")
    else:
        ok.append(f"AI 标注未写入正文（预期：由发布端自动勾选「{AI_LABEL}」，本 task 不写入草稿）")

    # 4) 事实断言 vs 新闻原文（仅当有新闻语料时）
    if news_corpus:
        _check_factual_claims(artifact, news_corpus, bad, ok)
    elif attempts == 0:
        # 无新闻源（纯 topic 降级生成）：警告而非硬失败（避免死循环）
        bad.append(
            "无新闻源语料（纯 topic 降级生成），无法做事实对照核查，"
            "建议在 --news-dir 提供抓取新闻"
        )

    # 5) 结构格式：话题标签唯一/文末/紧凑 + 互动收尾（确定性，见 evaluator 第 7/8 项）。
    #    仅小红书/抖音类「文末标签」结构适用；知乎/头条的话题由编辑器/发布端单独管理
    #    （知乎走「添加话题」、头条正文首行 #话题# 闭环），不做文末标签硬性检查。
    if platform in ("xiaohongshu", "douyin"):
        _check_hashtags_and_hook(artifact, bad, ok)
    else:
        ok.append(f"{platform} 话题由编辑器/发布端管理，跳过文末标签与互动收尾检查")

    # 6) 引流延伸阅读（软性 + 防杜撰）：注入过原创文章素材时，模型可自行判断相关性——
    #    相关才引、不相关可不引（优先保证内容质量），但**一旦引用**链接必须是注入素材中的
    #    真实 URL，杜撰/改写链接仍按失败拦截。
    articles_context = task_data.get("articles_context", "") or ""
    if articles_context.strip():
        # 注入素材中的真实 URL 集合（引流头的 https://erishen.cn/…）
        real_links = set(
            re.findall(r"https://erishen\.cn/[A-Za-z0-9_\-/]+", articles_context)
        )
        artifact_links = set(
            re.findall(r"https://erishen\.cn/[A-Za-z0-9_\-/]+", artifact)
        )
        if artifact_links:
            fake = sorted(artifact_links - real_links)
            if fake:
                bad.append(
                    f"成稿含杜撰的引流链接（不在注入素材的真实 URL 集合内）: {', '.join(fake)} "
                    "——必须改用任务给定的真实链接，不得改写/拼接/杜撰"
                )
            else:
                ok.append(
                    f"已含引流延伸阅读链接（全部为真实 URL）: {', '.join(sorted(artifact_links)[:2])}"
                )
        else:
            # 未引用引流素材：不强制（相关性不足时宁可不引，保证质量），仅提示
            ok.append("未引用引流延伸阅读（可选；相关性足够时才需引，保证内容质量为先）")
    else:
        ok.append("未注入引流素材，跳过延伸阅读检查")

    return bad, ok


def make_compliance_verifier(
    category: str = "tech_ai",
    platform: str = "xiaohongshu",
) -> Callable[[dict], tuple[list[str], list[str]]]:
    """生成单参 verify_fn 闭包，供 build_workflow(verify_fn=...) 注入。"""

    def _verify(state: dict) -> tuple[list[str], list[str]]:
        return verify_compliance(state, category=category, platform=platform)

    return _verify
