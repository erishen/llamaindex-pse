.PHONY: install lint clean resume-tailor resume-tailor-agnes help

PY := uv run python

install: ## 安装依赖（uv sync）
	uv sync

lint: ## 代码检查
	uv run ruff check src/ tasks/

clean: ## 清理缓存/构建产物
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ *.egg-info

# RAG 加持的简历定制
# 用法: make resume-tailor JD=path/to/jd.md
#       make resume-tailor JD=path/to/jd.md DOCS=/path/to/docs
resume-tailor: ## RAG 简历定制 - deepseek（需 JD= 参数）
	$(PY) tasks/resume-tailor/run.py --jd $(JD) $(if $(DOCS),--docs $(DOCS),)

resume-tailor-agnes: ## RAG 简历定制 - agnes（需 JD= 参数）
	$(PY) tasks/resume-tailor/run.py --jd $(JD) --provider agnes $(if $(DOCS),--docs $(DOCS),)

help: ## 列出全部命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
