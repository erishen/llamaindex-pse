"""hot-news 引流/人设 RAG 相关纯逻辑的单元测试。

覆盖对象全部为无 LLM/embedding 依赖的纯函数，测试不触发 LlamaIndex 索引构建。
"""

import os
import re
from pathlib import Path

import pytest

import run as hn  # noqa: I001 - 本地模块由 conftest 注入 path，放第三方后
from compliance import verify_compliance

# 本地引流索引校验用的文章库目录：通过环境变量显式指定，避免硬编码私有工作区路径。
# 未配置时指向仓库内 fixtures 占位目录（不存在则对应测试 skip）。
_ARTICLES_DIR_ENV = os.getenv("HOTNEWS_ARTICLES_DIR", "")
ZH_ARTICLES = (
    Path(_ARTICLES_DIR_ENV) / "zh"
    if _ARTICLES_DIR_ENV
    else Path(__file__).resolve().parent / "fixtures" / "articles" / "zh"
)


class TestParseArticleFrontmatter:
    def test_valid_slug_generates_url(self, tmp_path: Path):
        f = tmp_path / "a.md"
        f.write_text('---\ntitle: "你好"\nslug: hello-world\n---\n正文', encoding="utf-8")
        title, slug, url = hn._parse_article_frontmatter(str(f))
        assert title == "你好"
        assert slug == "hello-world"
        assert url == "https://erishen.cn/hello-world/"

    def test_chinese_slug_returns_empty_url(self, tmp_path: Path):
        f = tmp_path / "b.md"
        f.write_text("---\ntitle: 中文\ntags: [AI]\nslug: 使用-中文-标题\n---\n正文", encoding="utf-8")
        _t, slug, url = hn._parse_article_frontmatter(str(f))
        assert slug == ""
        assert url == ""

    def test_underscore_slug_rejected_by_frontmatter(self, tmp_path: Path):
        # 下划线 slug 由发布登记表显式提供 URL，frontmatter 不自行生成
        f = tmp_path / "c.md"
        f.write_text("---\ntitle: X\nslug: ai_workbench\n---\n正文", encoding="utf-8")
        _t, slug, url = hn._parse_article_frontmatter(str(f))
        assert slug == ""
        assert url == ""

    def test_no_frontmatter_returns_empty(self, tmp_path: Path):
        f = tmp_path / "d.md"
        f.write_text("# 只有标题没有 frontmatter", encoding="utf-8")
        title, slug, url = hn._parse_article_frontmatter(str(f))
        assert slug == ""
        assert url == ""
        assert title == "d"  # 标题退化为文件名

    def test_missing_file_returns_empty(self):
        _t, slug, url = hn._parse_article_frontmatter("/nonexistent/zz.md")
        assert slug == ""
        assert url == ""


class TestLoadPublishedArticleUrls:
    def test_loads_zh_urls_from_registry(self):
        urls = hn._load_published_article_urls()
        assert urls, "发布登记表应能加载出 zh 文章 URL"
        assert "lobster" in urls
        assert urls["lobster"] == "https://erishen.cn/building-ai-tool-server-lobster-architecture-cn/"
        # 全部为合法 https 链接
        for url in urls.values():
            assert url.startswith("https://erishen.cn/")

    def test_all_published_zh_have_indexable_file(self):
        """发布登记表中每个 zh URL 至少对应一个 zh 文章文件（无遗漏）。"""
        if not ZH_ARTICLES.exists():
            pytest.skip("未配置 HOTNEWS_ARTICLES_DIR，跳过本地引流索引校验")
        urls = hn._load_published_article_urls()
        # interview 与 fastapi-web 共享同一 URL，去重后即文件总数上限
        unique_zh = set(urls.values())
        indexed = {
            hn._resolve_article_url(f.stem, urls)
            for f in ZH_ARTICLES.glob("*.md")
            if hn._article_is_indexable(f.stem, urls)
        }
        missing = unique_zh - indexed
        assert not missing, f"登记表中以下已发布 zh 文章未被引流索引覆盖: {missing}"


class TestResolveArticleUrl:
    def test_override_takes_priority(self):
        urls = {"resolve-studio": "https://erishen.cn/resolve_studio/"}
        assert hn._resolve_article_url("resolve_studio-zh", urls) == urls["resolve-studio"]

    def test_normalized_match(self):
        urls = {"resolve-studio": "https://erishen.cn/resolve_studio/"}
        assert hn._resolve_article_url("resolve_studio-zh", urls) == urls["resolve-studio"]

    def test_underscore_kept_in_published_url(self):
        # erishen.cn 真实链接以下划线为准（上线验证 200），不允许归一化成连字符
        urls = {"ai-workbench": "https://erishen.cn/ai_workbench/"}
        assert hn._resolve_article_url("ai_workbench-zh", urls) == "https://erishen.cn/ai_workbench/"

    def test_special_slug_override(self):
        urls = {}
        assert (
            hn._resolve_article_url("ai-tool-server-lobster-architecture", urls)
            == "https://erishen.cn/building-ai-tool-server-lobster-architecture-cn/"
        )

    def test_unpublished_returns_empty(self):
        urls = {}
        assert hn._resolve_article_url("mac-brew-mysql-issue", urls) == ""


class TestArticleIsIndexable:
    def test_published_and_not_blacklisted(self):
        urls = {"resolve-studio": "https://erishen.cn/resolve_studio/"}
        assert hn._article_is_indexable("resolve_studio-zh", urls) is True

    def test_blacklisted_file_excluded(self):
        # 用户确认不引流的文件名黑名单
        urls = {"stock-analyzer": "https://erishen.cn/stock_analyzer/"}
        assert hn._article_is_indexable("china-developer-survey-2024", urls) is False
        assert hn._article_is_indexable("electron-react-desktop-app", urls) is False

    def test_unpublished_not_indexable(self):
        assert hn._article_is_indexable("privacy-policy", {}) is False

    def test_override_article_indexable(self):
        urls = {}
        assert hn._article_is_indexable("rag-smart-chat-app", urls) is True


class TestDedupArticleContext:
    def test_keeps_first_chunk_of_each_article(self):
        sample = (
            "[1] (score=0.47)\n"
            "📌 我的原创文章《Lobster 架构》：https://erishen.cn/a/\n"
            "正文1\n\n"
            "[2] (score=0.40)\n"
            "📌 我的原创文章《Lobster 架构》：https://erishen.cn/a/\n"
            "正文2（同篇另一 chunk）\n\n"
            "[3] (score=0.35)\n"
            "📌 我的原创文章《resolve_studio》：https://erishen.cn/b/\n"
            "正文3"
        )
        out = hn._dedup_article_context(sample)
        assert out.count("Lobster 架构") == 1
        assert out.count("resolve_studio") == 1
        assert "正文1" in out and "正文2（同篇另一 chunk）" not in out

    def test_empty_input_unchanged(self):
        assert hn._dedup_article_context("") == ""
        assert hn._dedup_article_context("  ") == "  "


class TestVerifyArticlesDrainage:
    """引流延伸阅读是 verify 的必过项：注入素材后成稿必须含 erishen.cn 链接。"""

    def _state(self, artifact: str, articles_context: str = "📌 我的原创文章《X》：https://erishen.cn/x/") -> dict:
        return {
            "artifact": artifact,
            "task_data": {"articles_context": articles_context, "news_corpus": ""},
            "attempts": 0,
        }

    def test_with_real_link_passes(self):
        bad, _ok = verify_compliance(
            self._state("正文...\n\n**延伸阅读**\n《X》：https://erishen.cn/x/"),
            category="tech_ai",
            platform="xiaohongshu",
        )
        assert not any("引流" in b for b in bad)

    def test_without_link_is_ok(self):
        # 引流是软性项：相关性不足时宁可不引（质量优先），不因无链接失败
        bad, _ok = verify_compliance(
            self._state("正文没有引流链接，只有#标签"),
            category="tech_ai",
            platform="xiaohongshu",
        )
        assert not any("引流" in b for b in bad)

    def test_fabricated_link_fails(self):
        # 模型杜撰不在注入素材里的假链接，必须被拦截
        bad, _ok = verify_compliance(
            self._state(
                "**延伸阅读**\n《假文》：https://erishen.cn/fake/debug-guide",
                articles_context="📌 我的原创文章《真文》：https://erishen.cn/real/",
            ),
            category="tech_ai",
            platform="xiaohongshu",
        )
        assert any("杜撰" in b for b in bad)

    def test_no_injection_skips_check(self):
        bad, ok = verify_compliance(
            self._state("正文", articles_context=""),
            category="tech_ai",
            platform="xiaohongshu",
        )
        assert not any("引流" in b for b in bad)
        assert any("未注入引流素材" in o for o in ok)


class TestIndexFingerprint:
    def test_fingerprint_changes_when_files_change(self, tmp_path: Path):
        f = tmp_path / "a.md"
        f.write_text("hello", encoding="utf-8")
        fp1 = hn._index_fingerprint(tmp_path)
        # 显式推进 mtime（模拟文件被更新），验证指纹对 mtime 敏感
        new_mtime = f.stat().st_mtime_ns + 1_000_000_000  # +1s
        os.utime(f, ns=(new_mtime, new_mtime))
        fp2 = hn._index_fingerprint(tmp_path)
        assert fp1 != fp2

    def test_same_content_same_fingerprint(self, tmp_path: Path):
        f = tmp_path / "a.md"
        f.write_text("hello", encoding="utf-8")
        fp1 = hn._index_fingerprint(tmp_path)
        fp2 = hn._index_fingerprint(tmp_path)
        assert fp1 == fp2

    def test_subdirs_scoping(self, tmp_path: Path):
        (tmp_path / "zh").mkdir()
        (tmp_path / "en").mkdir()
        (tmp_path / "zh" / "a.md").write_text("中文", encoding="utf-8")
        (tmp_path / "en" / "b.md").write_text("en", encoding="utf-8")
        fp_zh = hn._index_fingerprint(tmp_path, subdirs=("zh",))
        fp_all = hn._index_fingerprint(tmp_path)
        assert fp_zh != fp_all
        assert re.fullmatch(r"[0-9a-f]{32}", fp_zh)
