.PHONY: install lint clean resume-tailor resume-tailor-agnes resume-recommend resume-recommend-agnes resume-tailor-rebuild help

PY := uv run python

install: ## 安装依赖（uv sync）
	uv sync

lint: ## 代码检查
	uv run ruff check src/ tasks/

clean: ## 清理缓存/构建产物
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ *.egg-info .index_cache/

# ── JD 定制模式 ──
# 用法: make resume-tailor JD=path/to/jd.md
#       make resume-tailor JD=path/to/jd.md DOCS=/path/to/docs
resume-tailor: ## JD 定制简历 - deepseek（需 JD= 参数）
	$(PY) tasks/resume-tailor/run.py --jd $(JD) $(if $(DOCS),--docs $(DOCS),)

resume-tailor-agnes: ## JD 定制简历 - agnes（需 JD= 参数）
	$(PY) tasks/resume-tailor/run.py --jd $(JD) --provider agnes $(if $(DOCS),--docs $(DOCS),)

# ── 自由推荐模式（无需 JD）──
# 根据你的经历 + 国内招聘行情，推荐最适合的岗位并定制简历
resume-recommend: ## 自由推荐模式 - deepseek（无需 JD）
	$(PY) tasks/resume-tailor/run.py --recommend $(if $(DOCS),--docs $(DOCS),)

resume-recommend-agnes: ## 自由推荐模式 - agnes（无需 JD）
	$(PY) tasks/resume-tailor/run.py --recommend --provider agnes $(if $(DOCS),--docs $(DOCS),)

resume-tailor-rebuild: ## 强制重建 embedding 索引
	$(PY) tasks/resume-tailor/run.py --recommend --rebuild

help: ## 列出全部命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
