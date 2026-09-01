# ── hot-news 发布（知乎，文章编辑器，复用同一篇 .env 文章）──
# 流程：login 扫码一次 → check → publish（填稿+插图+封面+话题+尽力AI声明，停留，绝不自动发布）。
# 知乎标题上限 100 字（建议 ≤30），与 compliance.PLATFORM_LIMITS 同源；封面/话题/AI声明 仍由人工最终核对。
ZHIHU_CMD = $(PY) tasks/hot-news/publish/publish_zhihu.py
hot-news-publish-zhihu-login: ## 知乎App/微信扫码登录（存储到 publish/storage/zhihu_login.json）
	$(ZHIHU_CMD) login

hot-news-publish-zhihu-check: ## 知乎版离线校验（上限 100 字，建议 ≤30）
	$(ZHIHU_CMD) check

hot-news-publish-zhihu: ## 知乎：填好标题/正文/插图/封面/话题后保持页面，人工发布
	$(ZHIHU_CMD) publish $(if $(DRY_RUN),--dry-run,)

hot-news-publish-zhihu-cover-probe: ## 知乎：探测封面区 DOM（封面结构变动后校准 set_cover 用）
	$(ZHIHU_CMD) cover-probe

.PHONY: install lint clean resume-tailor resume-tailor-deepseek resume-tailor-scnet-kimi resume-tailor-scnet-minimax resume-recommend resume-recommend-deepseek resume-recommend-scnet-kimi resume-recommend-scnet-minimax resume-tailor-rebuild resume-tailor-rebuild-deepseek resume-tailor-rebuild-scnet-kimi resume-tailor-rebuild-scnet-minimax hot-news hot-news-topics hot-news-test hot-news-deepseek hot-news-fetch hot-news-refresh hot-news-cron-install hot-news-cron-uninstall hot-news-cron-status hot-news-publish hot-news-publish-login hot-news-publish-check hot-news-publish-toutiao hot-news-publish-toutiao-login hot-news-publish-toutiao-check hot-news-publish-zhihu hot-news-publish-zhihu-login hot-news-publish-zhihu-check hot-news-publish-zhihu-cover-probe help

PY := uv run python

install: ## 安装依赖（uv sync）
	uv sync

lint: ## 代码检查
	uv run ruff check src/ tasks/

clean: ## 清理缓存/构建产物
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ *.egg-info .index_cache/ tasks/hot-news/.index_cache/

# ── JD 定制模式 ──
# 用法: make resume-tailor JD=path/to/jd.md
#       make resume-tailor JD=path/to/jd.md DOCS=/path/to/docs
resume-tailor: ## JD 定制简历 - agnes（默认，需 JD= 参数）
	$(PY) tasks/resume-tailor/run.py --jd $(JD) $(if $(DOCS),--docs $(DOCS),)

resume-tailor-deepseek: ## JD 定制简历 - deepseek（需 JD= 参数，覆盖默认 agnes）
	$(PY) tasks/resume-tailor/run.py --jd $(JD) --provider deepseek $(if $(DOCS),--docs $(DOCS),)

# ── 自由推荐模式（无需 JD）──
# 根据你的经历 + 国内招聘行情，推荐最适合的岗位并定制简历
resume-recommend: ## 自由推荐模式 - agnes（默认，无需 JD）
	$(PY) tasks/resume-tailor/run.py --recommend $(if $(DOCS),--docs $(DOCS),)

resume-recommend-deepseek: ## 自由推荐模式 - deepseek（无需 JD，覆盖默认 agnes）
	$(PY) tasks/resume-tailor/run.py --recommend --provider deepseek $(if $(DOCS),--docs $(DOCS),)

# ── SCNet 模式（Kimi / MiniMax，需 .env 中 SCNET_* 配置）──
resume-recommend-scnet-kimi: ## 自由推荐模式 - SCNet Kimi（无需 JD）
	$(PY) tasks/resume-tailor/run.py --recommend --provider scnet-kimi $(if $(DOCS),--docs $(DOCS),)

resume-recommend-scnet-minimax: ## 自由推荐模式 - SCNet MiniMax（无需 JD，推荐）
	$(PY) tasks/resume-tailor/run.py --recommend --provider scnet-minimax $(if $(DOCS),--docs $(DOCS),)

resume-tailor-scnet-kimi: ## JD 定制模式 - SCNet Kimi（需 JD= 参数）
	$(PY) tasks/resume-tailor/run.py --jd $(JD) --provider scnet-kimi $(if $(DOCS),--docs $(DOCS),)

resume-tailor-scnet-minimax: ## JD 定制模式 - SCNet MiniMax（需 JD= 参数）
	$(PY) tasks/resume-tailor/run.py --jd $(JD) --provider scnet-minimax $(if $(DOCS),--docs $(DOCS),)

resume-tailor-rebuild: ## 强制重建分区 embedding 索引 - agnes（默认）
	rm -rf tasks/resume-tailor/.index_cache/
	$(PY) tasks/resume-tailor/run.py --recommend --rebuild

resume-tailor-rebuild-deepseek: ## 强制重建分区 embedding 索引 - deepseek（覆盖默认 agnes）
	rm -rf tasks/resume-tailor/.index_cache/
	$(PY) tasks/resume-tailor/run.py --recommend --rebuild --provider deepseek

resume-tailor-rebuild-scnet-kimi: ## 强制重建分区 embedding 索引 - SCNet Kimi
	rm -rf tasks/resume-tailor/.index_cache/
	$(PY) tasks/resume-tailor/run.py --recommend --rebuild --provider scnet-kimi

resume-tailor-rebuild-scnet-minimax: ## 强制重建分区 embedding 索引 - SCNet MiniMax
	rm -rf tasks/resume-tailor/.index_cache/
	$(PY) tasks/resume-tailor/run.py --recommend --rebuild --provider scnet-minimax

# ── 热点营销内容（RAG + 合规）──
# 用法: make hot-news TOPIC="AI 新规落地" NEWS_DIR=/path/to/news PLATFORM=xiaohongshu CATEGORY=tech_ai
#       默认 provider=agnes（对齐用户全局默认）；PROVIDER=deepseek 可覆盖
#       NEWS_DIR 留空则纯 topic 降级生成（事实对照缺失，合规风险高，不推荐）
PLATFORM ?= xiaohongshu
CATEGORY ?= tech_ai
NEWS_DIR ?= tasks/hot-news/news
OUT_DIR ?= tasks/hot-news/articles
MAX_AGE_DAYS ?= 2
SELECTION ?= random

hot-news: ## 热点新闻→RAG 生成→合规校对的营销内容（TOPIC= 留空则自动选题，默认 agnes）
	$(PY) tasks/hot-news/run.py $(if $(TOPIC),--topic "$(TOPIC)",) \
		$(if $(NEWS_DIR),--news-dir $(NEWS_DIR),) \
		$(if $(PROVIDER),--provider $(PROVIDER),) \
		$(if $(OUT_DIR),--out-dir $(OUT_DIR),) \
		--platform $(PLATFORM) --category $(CATEGORY) \
		--selection $(SELECTION) \
		$(if $(REBUILD),--rebuild,)

hot-news-topics: ## 列出微博热搜候选（标题+热度），供人工挑选 TOPIC
	$(PY) tasks/hot-news/run.py --list-topics $(if $(NEWS_DIR),--news-dir $(NEWS_DIR),)

hot-news-test: ## 运行 hot-news 单元测试（引流/人设/指纹纯逻辑，不触发 LLM/embedding）
	uv run pytest tasks/hot-news/tests/ -v

hot-news-deepseek: ## 同上 - 显式 deepseek provider（覆盖默认 agnes）
	$(MAKE) hot-news PROVIDER=deepseek $(if $(TOPIC),TOPIC="$(TOPIC)",) PLATFORM=$(PLATFORM) CATEGORY=$(CATEGORY) $(if $(NEWS_DIR),NEWS_DIR=$(NEWS_DIR),) $(if $(REBUILD),REBUILD=1,)

# ── 热点新闻抓取（落盘 tasks/hot-news/news，供 hot-news 的 --news-dir 消费）──
# 用法: make hot-news-fetch
#       make hot-news-fetch SOURCES=weibo,zhihu LIMIT=20 TOPIC="AI" KEEP_DAYS=3
#       make hot-news-fetch OUT=/tmp/news
#       make hot-news-fetch CLEAN=1            # 清空旧快照再重写（干净快照，不累积）
hot-news-fetch: ## 抓取多平台热点新闻到 tasks/hot-news/news
	$(PY) tasks/hot-news/fetch_news.py $(if $(OUT),--out $(OUT),) \
		$(if $(SOURCES),--sources $(SOURCES),) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(TOPIC),--topic "$(TOPIC)",) \
		$(if $(EXCLUDE),--exclude "$(EXCLUDE)",) \
		$(if $(KEEP_DAYS),--keep-days $(KEEP_DAYS),) \
		$(if $(MAX_AGE_DAYS),--max-age-days $(MAX_AGE_DAYS),) \
		$(if $(USE_PROXY),--use-proxy,) \
		$(if $(CLEAN),--clean,)

# ── hot-news 定时刷新（launchd）──
# 微博热搜每 15 分钟（覆盖写、不累积过期标题）、全源每天 9:00。
# 示例:
#   make hot-news-cron-install      # 安装并立即拉起 weibo
#   make hot-news-cron-status       # 查看两个 job
#   make hot-news-cron-uninstall    # 卸载但不删除已生成 plist
hot-news-refresh: ## 手动执行定时刷新（默认 weibo，REFRESH_MODE=full 全源）
	bash tasks/hot-news/scheduler/refresh.sh $(if $(REFRESH_MODE),$(REFRESH_MODE),weibo)

hot-news-cron-install: ## 安装 launchd 定时任务（weibo 每15分钟 + full 每天9点）
	bash tasks/hot-news/scheduler/install.sh install

hot-news-cron-uninstall: ## 卸载 launchd 定时任务
	bash tasks/hot-news/scheduler/install.sh uninstall

hot-news-cron-status: ## 查看 launchd 定时任务状态
	bash tasks/hot-news/scheduler/install.sh status

# ── hot-news 发布（小红书，草稿优先，绝不自动发布）──
# 流程：login 一次扫码存登录态 → check 离线校验 → publish（默认填好停编辑器，人工确认）。
# 默认文章来自 llamaindex-pse/.env 的 HOT_NEWS_ARTICLE；命令行传 ARTICLE= 可临时覆盖。
# 示例:
#   make hot-news-publish-login                        # 首次扫码，保存登录态到 publish/storage/
#   make hot-news-publish                               # 填好内容停在编辑器，人工核对后自己发
#   make hot-news-publish SAVE_AS_DRAFT=1               # 点「发布笔记」走官方 AI拦截→自动存草稿
#   make hot-news-publish ARTICLE=xxx.md COVER=c.jpg    # 覆盖文章 + 3:4 封面
PUBLISH_CMD = $(PY) tasks/hot-news/publish/publish_xiaohongshu.py
hot-news-publish-login: ## 小红书扫码登录（持久化 storageState）
	$(PUBLISH_CMD) login

hot-news-publish-check: ## 离线校验文章标题/字数/话题（不联网；ARTICLE 缺省读 .env）
	$(PUBLISH_CMD) check $(if $(ARTICLE),--article $(ARTICLE),)

hot-news-publish: ## 填好内容停编辑器供人工确认（SAVE_AS_DRAFT=1 走官方存草稿流程）
	$(PUBLISH_CMD) publish $(if $(ARTICLE),--article $(ARTICLE),) \
		$(if $(COVER),--cover $(COVER),) \
		$(if $(SAVE_AS_DRAFT),--save-as-draft,) \
		$(if $(ALLOW_NO_AI),--allow-no-ai-check,) \
		$(if $(DRY_RUN),--dry-run,)

# ── hot-news 发布（今日头条，复用同一篇 .env 文章 + 品牌多图）──
# 流程同小红书：login 扫码一次 → check → publish（default 填好停编辑器，绝不自动提交）。
TOUTIAO_CMD = $(PY) tasks/hot-news/publish/publish_toutiao.py
hot-news-publish-toutiao-login: ## 头条App扫码登录（存储到 publish/storage/toutiao_login.json）
	$(TOUTIAO_CMD) login

hot-news-publish-toutiao-check: ## 头条版离线校验（头条标题建议 ≤30 字）
	$(TOUTIAO_CMD) check

hot-news-publish-toutiao: ## 头条：填好标题/正文保持页面，人工发布（头条不存自动化草稿、不能自动配图）
	$(TOUTIAO_CMD) publish $(if $(DRY_RUN),--dry-run,)

help: ## 列出全部命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
