#!/usr/bin/env python3
"""把 news/ 下散落的各平台热点 .md 渲染成一份自包含 HTML 总览。

设计约束：
- 产物落在 news/ 的**父目录**（任务根），绝不写进 news/ 内部——
  run.py 用 news_dir.rglob('*') 把所有 .md 读进 RAG 语料，内部塞汇总会污染 grounding。
- 纯标准库，无第三方依赖；可被 fetch_news.py 在抓取后自动调用，也可单独 `python news_digest.py` 重跑。

用法：
  python news_digest.py                 # 读 news/ 生成 ../hot-news-overview.html
  python news_digest.py /path/to/news  # 指定目录
"""
from __future__ import annotations

import html as _html
import re
import sys
from datetime import datetime
from pathlib import Path

# 展示元数据（颜色 / 中文名 / 品类）。与 fetch_news.SOURCES 的 key 对齐。
SOURCE_META: dict[str, dict] = {
    "weibo": {"label": "微博热搜", "color": "#e6162d", "kind": "热榜"},
    "kr36": {"label": "36氪", "color": "#0a7cff", "kind": "科技"},
    "sspai": {"label": "少数派", "color": "#f3591b", "kind": "数字生活"},
    "qbitai": {"label": "量子位", "color": "#7c3aed", "kind": "AI"},
    "infoq": {"label": "InfoQ", "color": "#1f9d55", "kind": "工程"},
}
# 排序展示时的源顺序
SOURCE_ORDER = ["weibo", "qbitai", "infoq", "kr36", "sspai"]

_FM = re.compile(r"^---\s*\n(.*?)\n---", re.S)
_SUMMARY = re.compile(r"^>\s*摘要[:：]\s*(.+)$", re.M)


def _parse(p: Path) -> dict | None:
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    fm: dict[str, str] = {}
    m = _FM.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip()
    else:
        return None
    title = fm.get("title", "")
    if not title:
        return None
    sm = _SUMMARY.search(text)
    summary = sm.group(1).strip() if sm else ""
    raw_hot = fm.get("hot", "").strip()
    hot = 0
    if raw_hot:
        try:
            hot = int(raw_hot)
        except ValueError:
            hot = 0
    url = fm.get("url", "").strip()
    source = fm.get("source", p.parent.name)
    published = fm.get("published_at", "").strip()
    fetched = fm.get("fetched_at", "").strip()
    return {
        "source": source,
        "title": title,
        "hot": hot,
        "url": url,
        "summary": summary,
        "published": published,
        "fetched": fetched,
        "file": p.name,
    }


def _fts(s: str) -> int:
    """抓取/发布时间的秒级时间戳，用于去重时保留最新一份。"""
    if not s:
        return 0
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(s, fmt).timestamp())
        except ValueError:
            continue
    return 0


def _norm(title: str) -> str:
    """标题归一（去空白+小写），用于跨次运行去重。"""
    return re.sub(r"\s+", "", title).lower()


def _dedup(items: list[dict]) -> list[dict]:
    """同标题只留最新一份（按 fetched_at / published_at 取大）。"""
    best: dict[str, dict] = {}
    for it in items:
        k = _norm(it["title"])
        if not k:
            continue
        cur = best.get(k)
        if cur is None or _fts(it["fetched"]) > _fts(cur["fetched"]) or (
            _fts(it["fetched"]) == _fts(cur["fetched"]) and _fts(it["published"]) > _fts(cur["published"])
        ):
            best[k] = it
    return list(best.values())


def _load(news_dir: Path) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = {}
    for p in sorted(news_dir.rglob("*.md")):
        if not p.is_file():
            continue
        item = _parse(p)
        if not item:
            continue
        by_source.setdefault(item["source"], []).append(item)
    # 多次运行会累积近似重复（RSS 源只追加不清旧），先按标题去重保留最新
    for src in list(by_source.keys()):
        by_source[src] = _dedup(by_source[src])
    # 每源内部排序：有热度按热度降序；无热度按发布时间降序（空置后）
    for src, items in by_source.items():
        items.sort(
            key=lambda x: (x["hot"], _ts(x["published"]), _fts(x["fetched"])),
            reverse=True,
        )
    return by_source


def _ts(s: str) -> int:
    if not s:
        return 0
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(s, fmt).timestamp())
        except ValueError:
            continue
    return 0


def _esc(s: str) -> str:
    return _html.escape(str(s), quote=True)


def _fmt_hot(n: int) -> str:
    if n >= 10000:
        return f"{n / 10000:.1f}w"
    return str(n)


_CSS = """
:root {
  --bg: #f6f7f9;
  --card: #ffffff;
  --ink: #1f2329;
  --muted: #8a919f;
  --line: #ececf0;
  --accent: #2b6cff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  line-height: 1.5;
}
.wrap { max-width: 960px; margin: 0 auto; padding: 24px 16px 64px; }
header.top {
  background: linear-gradient(135deg, #2b6cff, #6a3df0);
  color: #fff;
  border-radius: 16px;
  padding: 22px 24px;
  box-shadow: 0 6px 20px rgba(43, 108, 255, .25);
}
header.top h1 { margin: 0 0 6px; font-size: 22px; }
header.top .sub { opacity: .9; font-size: 13px; }
.chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255, 255, 255, .18);
  border: 1px solid rgba(255, 255, 255, .35);
  color: #fff; border-radius: 999px; padding: 4px 12px; font-size: 12px;
}
.chip .dot { width: 8px; height: 8px; border-radius: 50%; background: #fff; }
h2.section { font-size: 17px; margin: 30px 0 12px; display: flex; align-items: center; gap: 8px; }
h2.section .count { color: var(--muted); font-weight: 400; font-size: 13px; }
.section-card {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 6px 4px; overflow: hidden;
}
.src-head {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 16px; border-bottom: 1px solid var(--line);
}
.src-head .tag {
  font-size: 12px; color: #fff; border-radius: 6px; padding: 2px 8px; font-weight: 600;
}
.src-head .kind { color: var(--muted); font-size: 12px; }
.src-head .n { margin-left: auto; color: var(--muted); font-size: 12px; }
ul.list { list-style: none; margin: 0; padding: 0; }
ul.list li {
  display: flex; gap: 12px; align-items: flex-start;
  padding: 12px 16px; border-bottom: 1px solid var(--line);
}
ul.list li:last-child { border-bottom: none; }
.rank {
  flex: 0 0 26px; height: 26px; border-radius: 8px; font-size: 13px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  background: #f0f2f5; color: #6b7280;
}
.rank.top { background: var(--accent); color: #fff; }
.main { flex: 1 1 auto; min-width: 0; }
.title { font-size: 15px; font-weight: 600; word-break: break-word; }
a.title-link { color: var(--ink); text-decoration: none; }
a.title-link:hover { color: var(--accent); text-decoration: underline; }
.meta { margin-top: 4px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }
.hot { color: #e6162d; font-weight: 600; }
.bar { height: 4px; border-radius: 2px; background: #f0f2f5; margin-top: 6px; overflow: hidden; }
.bar > i { display: block; height: 100%; background: linear-gradient(90deg, #ff8a3d, #e6162d); }
.summary {
  margin-top: 6px; font-size: 13px; color: #4b5563;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
footer { margin-top: 36px; text-align: center; color: var(--muted); font-size: 12px; }
"""


def build_digest(news_dir: Path) -> str:
    by_source = _load(news_dir)
    total = sum(len(v) for v in by_source.values())
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    gen_time = now

    # 综合热榜：跨源有热度的 Top 20（按标题全局去重，避免同选题多平台重复）
    seen: set[str] = set()
    ranked: list[tuple[str, dict]] = []
    for s, items in by_source.items():
        for it in items:
            if it["hot"] <= 0:
                continue
            k = _norm(it["title"])
            if k in seen:
                continue
            seen.add(k)
            ranked.append((s, it))
    ranked.sort(key=lambda x: x[1]["hot"], reverse=True)
    top = ranked[:20]
    max_hot = top[0][1]["hot"] if top else 1

    sources_sorted = [s for s in SOURCE_ORDER if s in by_source]
    sources_sorted += [s for s in by_source if s not in sources_sorted]

    parts: list[str] = []
    parts.append(
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>热点新闻总览 · {gen_time}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        "<div class=\"wrap\">"
    )
    # header
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:{_esc(m["color"])}"></span>'
        f"{_esc(m.get('label', s))} {len(by_source[s])}</span>"
        for s in sources_sorted
        if (m := SOURCE_META.get(s, {}))
    )
    parts.append(
        f'<header class="top"><h1>📰 热点新闻总览</h1>'
        f'<div class="sub">生成于 {_esc(gen_time)} · 共 {total} 条 · 覆盖 {len(by_source)} 个平台</div>'
        f'<div class="chips">{chips}</div></header>'
    )

    # 综合热榜
    if top:
        parts.append(
            f'<h2 class="section">🔥 综合热榜 <span class="count">Top {len(top)}（按热度）</span></h2>'
            '<div class="section-card"><ul class="list">'
        )
        for i, (s, it) in enumerate(top, 1):
            meta = SOURCE_META.get(s, {})
            label = _esc(meta.get("label", s))
            color = _esc(meta.get("color", "#888"))
            title_html = _title_html(it)
            w = max(4, int(it["hot"] / max_hot * 100))
            parts.append(
                f'<li><span class="rank {"top" if i <= 3 else ""}">{i}</span>'
                f'<div class="main">{title_html}'
                f'<div class="meta"><span class="tag" style="background:{color};color:#fff;'
                f'border-radius:4px;padding:1px 6px;font-size:11px;">{label}</span>'
                f'<span class="hot">🔥 {_fmt_hot(it["hot"])}</span></div>'
                f'<div class="bar"><i style="width:{w}%"></i></div></div></li>'
            )
        parts.append("</ul></div>")

    # 各源分区
    for s in sources_sorted:
        items = by_source[s]
        meta = SOURCE_META.get(s, {})
        label = _esc(meta.get("label", s))
        color = _esc(meta.get("color", "#888"))
        kind = _esc(meta.get("kind", ""))
        parts.append(
            f'<h2 class="section">{_esc(label)} <span class="count">{kind} · {len(items)} 条</span></h2>'
            '<div class="section-card">'
            f'<div class="src-head"><span class="tag" style="background:{color}">{label}</span>'
            f'<span class="kind">{kind}</span><span class="n">{len(items)} 条</span></div>'
            '<ul class="list">'
        )
        for i, it in enumerate(items, 1):
            title_html = _title_html(it)
            meta_bits = []
            if it["hot"] > 0:
                meta_bits.append(f'<span class="hot">🔥 {_fmt_hot(it["hot"])}</span>')
            if it["published"]:
                meta_bits.append(f'🕒 {_esc(it["published"])}')
            elif it["fetched"]:
                meta_bits.append(f'📥 {_esc(it["fetched"])}')
            meta_line = "".join(meta_bits)
            summary_html = (
                f'<div class="summary">{_esc(it["summary"])}</div>' if it["summary"] else ""
            )
            bar_html = ""
            if it["hot"] > 0:
                w = max(4, int(it["hot"] / max_hot * 100))
                bar_html = f'<div class="bar"><i style="width:{w}%"></i></div>'
            parts.append(
                f'<li><span class="rank">{i}</span><div class="main">{title_html}'
                + (f'<div class="meta">{meta_line}</div>' if meta_line else "")
                + summary_html
                + bar_html
                + "</div></li>"
            )
        parts.append("</ul></div>")

    parts.append(
        f'<footer>由 news_digest.py 生成 · 数据目录 {_esc(str(news_dir))} · {gen_time}</footer>'
        "</div></body></html>"
    )
    return "".join(parts)


def _title_html(it: dict) -> str:
    t = _esc(it["title"])
    if it["url"] and it["url"].lower().startswith("http"):
        return f'<a class="title-link" href="{_esc(it["url"])}" target="_blank" rel="noopener">{t}</a>'
    return f'<span class="title">{t}</span>'


def write_digest(news_dir: Path) -> Path:
    # resolve() makes the dir absolute (cwd-relative input included). The path
    # returned here is printed verbatim into the chat bubble ("📊 总览已生成:
    # …"), and only absolute .html paths get linkified into a clickable
    # preview, so never hand back a relative one.
    news_dir = Path(news_dir).resolve()
    out = news_dir.parent / "hot-news-overview.html"
    out.write_text(build_digest(news_dir), encoding="utf-8")
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (Path(__file__).resolve().parent / "news")
    path = write_digest(target)
    print(f"✅ 总览已生成: {path}  ({path.stat().st_size} bytes)")
