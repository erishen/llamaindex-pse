"""fetch_news.py 的纯逻辑单元测试：乱码清洗与正文抽取（不触网）。"""

from datetime import datetime
from pathlib import Path

import fetch_news as fn
import news_digest as nd


class TestCleanText:
    def test_strips_replacement_char(self):
        # 36氪 feed 自带 U+FFFD，源侧已丢失，只能清洗
        assert fn._clean_text("比亚\ufffd迪\ufffd涨价") == "比亚迪涨价"

    def test_strips_iconfont_private_use(self):
        # InfoQ 等站点 iconfont 用私用区码位承载图标，会混入正文显示为豆腐块
        assert fn._clean_text("更多 \ue607\ue604 写点什么") == "更多 写点什么"

    def test_strips_zero_width_and_bom(self):
        assert fn._clean_text("a\u200bb\ufeffc") == "abc"


class TestHtmlToText:
    def test_prefers_article_over_surrounding_chrome(self):
        doc = (
            "<header>首页 更多 \ue607 写点什么</header>"
            "<article><p>真正的正文内容</p></article>"
            "<footer>版权信息</footer>"
        )
        assert fn._html_to_text(doc) == "真正的正文内容"

    def test_picks_longest_article_block(self):
        doc = "<article>短</article><article>这是较长的一段正文</article>"
        assert fn._html_to_text(doc) == "这是较长的一段正文"

    def test_falls_back_when_no_article(self):
        doc = "<nav>导航</nav><div>正文段</div><script>x()</script>"
        assert fn._html_to_text(doc) == "正文段"

    def test_strips_iconfont_from_body(self):
        doc = "<article>AI \ue601时代</article>"
        assert fn._html_to_text(doc) == "AI 时代"


class TestWrite:
    def test_snapshot_has_no_junk_chars(self, tmp_path: Path):
        items = [
            {
                "title": "AI\ue601时代",
                "url": "https://x/y",
                "hot": 1,
                "published": datetime(2026, 10, 11, 9, 0),
                "body": "更多 \ue607 写点什么\ufffd",
            }
        ]
        fn._write(tmp_path, "infoq", items, keep_days=7)
        text = next((tmp_path / "infoq").glob("*.md")).read_text(encoding="utf-8")
        for bad in ("\ue601", "\ue607", "\ufffd"):
            assert bad not in text
        assert "AI时代" in text


class TestResolveOutDir:
    """--out 的相对路径以任务目录为基准，不跟随 cwd。

    否则在仓库根目录跑 `fetch_news.py --out news` 会把快照落到仓库根，
    news_digest 再把总览 HTML 写到 news 的父目录，一并污染工作区（2026-11 发生过）。
    """

    def test_relative_resolves_under_task_dir(self):
        assert fn.resolve_out_dir("news") == fn.BASE / "news"

    def test_relative_with_dot_prefix(self):
        assert fn.resolve_out_dir("./tmp/news") == (fn.BASE / "tmp" / "news").resolve()

    def test_absolute_passthrough(self, tmp_path: Path):
        assert fn.resolve_out_dir(str(tmp_path)) == tmp_path.resolve()


class TestResolveNewsDir:
    """单独跑 news_digest.py 时，相对目录同样以任务目录为基准。"""

    def test_relative_resolves_under_task_dir(self):
        assert nd.resolve_news_dir("news") == nd.BASE / "news"

    def test_absolute_passthrough(self, tmp_path: Path):
        assert nd.resolve_news_dir(str(tmp_path)) == tmp_path.resolve()
