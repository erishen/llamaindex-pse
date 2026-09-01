#!/usr/bin/env python3
"""抓取多平台热点新闻，落盘为 Markdown 供 llamaindex-pse hot-news 任务做 RAG grounding。

新闻目录（默认 tasks/hot-news/news）即为 run.py 的 --news-dir：
run.py 用 rglob 递归读取其中的 .md/.txt/.pdf 建索引并做事实对照。

数据源（默认绕开系统代理直连，避免 clash 等代理对公开域名返回 TLS EOF）：
  - weibo : 微博热搜直连 JSON（选题向，纯标题+热度）
  - kr36  : 36氪 RSS（素材向，含正文摘要，grounding 质量高）
  - sspai : 少数派 RSS（同上）
  - qbitai: 量子位 RSS（AI 专属，中文 AI 资讯，tech_ai 品类主源）
  - infoq : InfoQ 中文 RSS（AI/软件工程向，含大量 Agent/大模型选题；摘要过短时自动抓正文）

用法:
  uv run python tasks/hot-news/fetch_news.py                  # 抓全部默认源
  uv run python tasks/hot-news/fetch_news.py --topic "AI"     # 仅留标题或正文含 AI 的
  uv run python tasks/hot-news/fetch_news.py --sources weibo  # 只抓微博
  uv run python tasks/hot-news/fetch_news.py --use-proxy      # 走系统代理（默认直连）
  uv run python tasks/hot-news/fetch_news.py --clean          # 清空旧快照再重写（干净快照，不累积）
  uv run python tasks/hot-news/fetch_news.py --out /tmp/news --keep-days 3

要加新源：在 SOURCES 加一项 {url, parser, headers?}，parser ∈ weibo_hot|rss|json。
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

# 拉黑词单一数据源：默认值取 compliance.EXCLUDED_TOPICS（与 run.py 自动选题同源）
from compliance import EXCLUDED_TOPICS

BASE = Path(__file__).resolve().parent

# 默认输出目录：与 hot-news 任务的 news-dir 消费约定对齐
DEFAULT_OUT = BASE / "news"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# 默认源：热榜(选题) + RSS(正文素材)。各源独立容错，单源失败不影响其他源。
# parser 类型: weibo_hot(微博热搜直连) | rss(标准 RSS/Atom) | json(通用聚合, 容错取字段)
SOURCES: dict[str, dict] = {
    "weibo": {
        "url": "https://weibo.com/ajax/side/hotSearch",
        "parser": "weibo_hot",
        "headers": {"Referer": "https://weibo.com/"},
    },
    "kr36": {"url": "https://www.36kr.com/feed", "parser": "rss"},
    "sspai": {"url": "https://sspai.com/feed", "parser": "rss"},
    "qbitai": {"url": "https://www.qbitai.com/feed", "parser": "rss"},
    "infoq": {"url": "https://www.infoq.cn/feed", "parser": "rss"},
}

# AI 专属源（tech_ai 品类优先从这些选题；比微博热搜的 AI 命中率高得多）
AI_SOURCES = ("qbitai", "infoq")

TIMEOUT = 10

# RSS 摘要过短时正文补抓策略：description 低于该长度视为无内容（如 InfoQ 仅「点击查看原文」），
# 改为抓取文章链接正文，避免 RAG grounding 只有空壳标题、逼 specialist 编造细节
BODY_MIN = 80            # description 低于该字数（缩略后）才触发正文抓取
BODY_MAX = 2000          # 抓到的正文最大保留字数
BODY_FETCH_LIMIT = 12    # 每源最多正文抓取条数，限速防骚扰站点
BODY_FETCH_WORKERS = 4   # 正文抓取并发路数（串行 12 条 × 10s 超时太慢，4 路折中）
POLITENESS_DELAY = 0.25  # 并发提交的请求间隔（秒），避免同刻打爆目标站点


def _fetch_text(url: str, use_proxy: bool = False, extra_headers: dict | None = None) -> str:
    # 默认绕开系统代理：抓公开热榜/RSS 不需要代理，且代理常对这类域名返回 TLS EOF
    handlers = [] if use_proxy else [urllib.request.ProxyHandler({})]
    opener = urllib.request.build_opener(*handlers)
    headers = {"User-Agent": UA}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    with opener.open(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", "ignore")


def _pick(d: dict, *keys: str) -> str:
    for k in keys:
        if k in d and d[k]:
            return str(d[k])
    return ""


def _parse_weibo(payload: dict) -> list[dict]:
    realtime = (payload.get("data") or {}).get("realtime") or []
    items: list[dict] = []
    for row in realtime:
        if not isinstance(row, dict):
            continue
        word = _pick(row, "word", "title")
        if not word:
            continue
        items.append(
            {
                "title": word,
                "hot": _pick(row, "num", "raw_hot", "hot"),
                "url": _pick(row, "url", "href"),
                "body": "",
            }
        )
    return items


def _parse_json_generic(payload: dict) -> list[dict]:
    """容错取 data 数组；适配 {code,data}/{data}/{result}/{list} 多种外壳。"""
    data = payload.get("data") or payload.get("result") or payload.get("list") or []
    if not isinstance(data, list):
        return []
    items: list[dict] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        title = _pick(row, "title", "word", "name", "query")
        if not title:
            continue
        items.append(
            {
                "title": title,
                "hot": _pick(row, "hot", "num", "heat", "score"),
                "url": _pick(row, "url", "link", "mobil_url", "href"),
                "body": "",
            }
        )
    return items


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "").strip()


def _html_to_text(html_doc: str) -> str:
    """HTML → 纯文本：去掉脚本/样式/导航壳，转义实体，压缩空白。

    尽力而为，不追求解析站点自定义结构（页眉页脚可能掺入，但不足以证伪）。
    """
    text = re.sub(
        r"(?is)<(script|style|head|noscript|nav|footer|aside)[^>]*>.*?</\1>",
        " ",
        html_doc or "",
    )
    text = _strip_tags(text)
    text = html_mod.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_body_one(it: dict, use_proxy: bool) -> None:
    """抓取单篇文章正文并写回 it['body']。失败仅打印告警，不影响其余条目。"""
    url = it.get("url")
    if not url:
        return
    try:
        raw = _fetch_text(url, use_proxy=use_proxy)
    except Exception as e:  # noqa: BLE001
        print(f"  · [正文] 抓取失败: {url} ({e})", file=sys.stderr)
        return
    text = _html_to_text(raw)
    if text:
        it["body"] = text[:BODY_MAX]
        print(f"  · [正文] {it['title'][:24]}：{len(text)} 字", file=sys.stderr)


def _enrich_bodies(items: list[dict], use_proxy: bool) -> list[dict]:
    """对摘要过短的 RSS 条目并发抓取文章正文，补齐 grounding。

    并发 BODY_FETCH_WORKERS 路 + 提交间隔 POLITENESS_DELAY 限速，
    避免串行 12 条 × 10s 超时拖慢全源，也不至于同刻打爆目标站点。
    单条失败仅跳过，不影响其余条目与整个源。
    """
    pending = [
        it for it in items
        if not (it.get("body") and len(it["body"]) >= BODY_MIN) and it.get("url")
    ][:BODY_FETCH_LIMIT]
    if not pending:
        return items
    with ThreadPoolExecutor(max_workers=BODY_FETCH_WORKERS) as pool:
        futures = []
        for it in pending:
            futures.append(pool.submit(_fetch_body_one, it, use_proxy))
            time.sleep(POLITENESS_DELAY)
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as err:  # noqa: BLE001
                print(f"  · [正文] 抓取线程异常: {err}", file=sys.stderr)
    return items


def _text_of(node: ET.Element, *tags: str) -> str:
    for child in node:
        if child.tag.split("}")[-1] in tags:
            # itertext 含后代文本，兼容 <description><p>...</p></description> 这类带子标签的情况
            return "".join(child.itertext()).strip()
    return ""


def _link_of(node: ET.Element) -> str:
    for child in node:
        if child.tag.split("}")[-1] == "link":
            # RSS: <link>text</link> ; Atom: <link href="..."/>
            if child.text and child.text.strip():
                return child.text.strip()
            href = child.get("href")
            if href:
                return href
    return ""


def _parse_date(node: ET.Element) -> datetime | None:
    """从 RSS/Atom item 节点提取发布时间，尽力解析多种格式。"""
    raw = _text_of(node, "pubDate", "published", "updated", "issued", "date")
    if not raw:
        for child in node:
            if child.tag.split("}")[-1] == "date" and child.text and child.text.strip():
                raw = child.text.strip()
                break
    if not raw:
        return None
    raw = raw.strip()
    # RSS pubDate: RFC 822 (Sat, 30 Aug 2026 10:00:00 GMT)
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt
    except (TypeError, ValueError, OverflowError):
        pass
    # Atom: ISO 8601
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _parse_rss(text: str) -> list[dict]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    items: list[dict] = []
    for node in root.iter():
        if node.tag.split("}")[-1] in ("item", "entry"):
            title = _text_of(node, "title")
            if not title:
                continue
            desc = _text_of(node, "description", "summary")
            items.append(
                {
                    "title": title,
                    "hot": "",
                    "url": _link_of(node),
                    "body": _strip_tags(desc)[:800],
                    "published": _parse_date(node),
                }
            )
    return items


def _safe_name(s: str, max_len: int = 60) -> str:
    s = re.sub(r'[\\/:*?"<>|#\n\r\t]', "_", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:max_len] or "untitled"


def fetch_source(
    name: str,
    spec: dict,
    limit: int,
    topic: str | None,
    use_proxy: bool,
    max_age_days: int = 0,
) -> list[dict]:
    # 默认直连；若直连失败且用户未强制走代理，则自动回退代理重试一次（适配部分域名直连不稳的网络）
    attempts = [use_proxy] if use_proxy else [False, True]
    text = None
    last_err: Exception | None = None
    for up in attempts:
        try:
            text = _fetch_text(spec["url"], use_proxy=up, extra_headers=spec.get("headers"))
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if not use_proxy:
                print(f"  · {name}: 直连失败，回退代理重试", file=sys.stderr)
    if text is None:
        print(f"  ⚠️ {name} 抓取失败: {last_err}", file=sys.stderr)
        return []
    parser = spec.get("parser", "json")
    try:
        if parser == "weibo_hot":
            items = _parse_weibo(json.loads(text))
        elif parser == "rss":
            items = _parse_rss(text)
        else:
            items = _parse_json_generic(json.loads(text))
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ {name} 解析失败: {e}", file=sys.stderr)
        return []
    if topic:
        key = topic.lower()
        items = [
            it
            for it in items
            if key in it["title"].lower()
            or (it.get("body") and key in it["body"].lower())
        ]
    if max_age_days and max_age_days > 0:
        now = datetime.now()
        kept: list[dict] = []
        dropped = 0
        dated = 0
        for it in items:
            pub = it.get("published")
            if pub is not None:
                dated += 1
                if (now - pub) > timedelta(days=max_age_days):
                    dropped += 1
                    continue
            kept.append(it)
        if dropped:
            print(f"  · {name}: 过滤 {dropped} 条 {max_age_days}天前旧文", file=sys.stderr)
        if dated == 0:
            print(f"  · {name}: 条目均无发布时间，--max-age-days 不生效（保留全部）", file=sys.stderr)
        items = kept
    # RSS 摘要过短时补抓文章正文（weibo_hot 纯标题不补，避免热榜链接无正文）
    if parser == "rss":
        _enrich_bodies(items, use_proxy)
    return items[:limit]


def _write(out_root: Path, source: str, items: list[dict], keep_days: int) -> int:
    sub = out_root / source
    sub.mkdir(parents=True, exist_ok=True)
    # 微博热榜是瞬时数据：本次若抓到新榜单，先清空子目录旧文件，避免旧标题累积成「过期热搜」
    if source == "weibo" and items:
        for old in sub.glob("*.md"):
            try:
                old.unlink()
            except OSError:
                pass
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = 0
    for i, it in enumerate(items, 1):
        fname = f"{i:02d}_{_safe_name(it['title'])}.md"
        body = it.get("body") or ""
        content = (
            "---\n"
            f"source: {source}\n"
            f"title: {it['title']}\n"
            f"hot: {it['hot']}\n"
            f"url: {it['url']}\n"
            f"published_at: {it.get('published').strftime('%Y-%m-%d %H:%M') if it.get('published') else ''}\n"
            f"fetched_at: {ts}\n"
            "---\n\n"
            f"# {it['title']}\n\n"
            f"- 平台: {source}\n"
            f"- 热度: {it['hot']}\n"
            f"- 链接: {it['url']}\n\n"
        )
        if body:
            content += f"> 摘要：{body}\n\n"
        content += f"> 抓取时间: {ts}\n"
        (sub / fname).write_text(content, encoding="utf-8")
        n += 1
    return n


def _clean_old(out_root: Path, keep_days: int) -> int:
    if keep_days <= 0:
        return 0
    cutoff = datetime.now() - timedelta(days=keep_days)
    removed = 0
    for p in out_root.rglob("*.md"):
        try:
            if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def _clean_all(out_root: Path) -> int:
    """清空输出目录下所有已落盘的热点文件，用于 --clean 干净快照。

    仅删 out_root 内的 .md（不碰其他类型/父目录），且调用方保证已抓到数据才调用。
    """
    removed = 0
    for p in out_root.rglob("*.md"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取多平台热点新闻（落盘 Markdown 供 RAG grounding）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出根目录（默认 tasks/hot-news/news）")
    ap.add_argument("--sources", default="", help="逗号分隔平台，默认全部；可选: " + ",".join(SOURCES))
    ap.add_argument("--limit", type=int, default=30, help="每平台保留条数（默认 30）")
    ap.add_argument("--topic", default="", help="仅保留标题或正文含该关键词的热点")
    ap.add_argument(
        "--exclude",
        default=",".join(EXCLUDED_TOPICS),
        help="逗号分隔拉黑词，标题命中任一即剔除（忽略大小写/空格；默认 compliance.EXCLUDED_TOPICS）",
    )
    ap.add_argument(
        "--max-age-days",
        type=int,
        default=2,
        help="仅保留 N 天内的热点（过滤旧文；0=不过滤，默认 2）；weibo 热榜无发布时间不应用",
    )
    ap.add_argument(
        "--keep-days",
        type=int,
        default=7,
        help="清理 N 天前的旧文件（0=不清理，默认 7）",
    )
    ap.add_argument(
        "--use-proxy",
        action="store_true",
        help="走系统代理（默认直连，避开代理对公开域名的 TLS 干扰）",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="重写前清空 out 目录下所有已落盘热点（干净快照，不累积；仅当抓到数据时才清空，避免全失败时误删）",
    )
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    args.exclude = [k.strip().lower() for k in (args.exclude or "").split(",") if k.strip()]

    if args.sources:
        chosen = [s.strip() for s in args.sources.split(",") if s.strip()]
        unknown = [s for s in chosen if s not in SOURCES]
        if unknown:
            print(f"❌ 未知源: {', '.join(unknown)}；可用: {', '.join(SOURCES)}", file=sys.stderr)
            return 2
    else:
        chosen = list(SOURCES.keys())

    print(f"▶ 抓取热点新闻 → {out_root}" + ("（走系统代理）" if args.use_proxy else "（直连）"))
    topic = args.topic or None
    results: list[tuple[str, list[dict]]] = []
    total = 0
    for name in chosen:
        items = fetch_source(name, SOURCES[name], args.limit, topic, args.use_proxy, args.max_age_days)
        if args.exclude and items:
            before = len(items)
            items = [
                it for it in items
                if not any(k.lower() in str(it["title"]).lower().replace(" ", "") for k in args.exclude)
            ]
            if before - len(items):
                print(f"  · {name}: 排除 {before - len(items)} 条（拉黑词: {','.join(args.exclude)}）", file=sys.stderr)
        if items:
            print(f"  ✓ {name}: 抓取 {len(items)} 条")
            results.append((name, items))
            total += len(items)
        else:
            print(f"  · {name}: 0 条（可能源不可用或被关键词过滤）")

    # --clean：先抓到数据再清空旧快照，避免全失败时误删已有数据
    if args.clean and total > 0:
        wiped = _clean_all(out_root)
        print(f"  🧹 --clean 清空旧快照 {wiped} 个文件")

    for name, items in results:
        written = _write(out_root, name, items, args.keep_days)
        print(f"  ✓ {name}: {written} 条写入")

    removed = _clean_old(out_root, args.keep_days)
    print(f"✅ 完成：新增/更新 {total} 条，清理旧文件 {removed} 个 → {out_root}")
    if total == 0:
        print(
            "  ⚠️ 未抓到任何热点：直连不通可试 --use-proxy；仍失败请检查网络或 SOURCES 端点。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
