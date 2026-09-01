#!/usr/bin/env bash
# hot-news 定时刷新入口（供 launchd / crontab 调用）
# 用法: refresh.sh weibo   # 只抓微博热搜（轻量，建议每 15 分钟）
#       refresh.sh full    # 全源抓取（weibo+kr36+sspai+qbitai+infoq，建议每天 1 次）
set -euo pipefail

MODE="${1:-weibo}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOGS_DIR="$PROJECT_DIR/tasks/hot-news/logs"
mkdir -p "$LOGS_DIR"

stamp="$(date '+%Y-%m-%d %H:%M:%S')"
log="$LOGS_DIR/refresh-${MODE}-$(date +%Y%m%d-%H%M%S).log"

case "$MODE" in
  weibo)
    cmd=(uv run python tasks/hot-news/fetch_news.py --sources weibo --limit 30 --keep-days 0)
    ;;
  full)
    # --exclude 缺省 = compliance.EXCLUDED_TOPICS（拉黑词单一数据源，勿在此重复硬编码）
    cmd=(uv run python tasks/hot-news/fetch_news.py --keep-days 7 --max-age-days 2)
    ;;
  *)
    echo "❌ 未知模式: $MODE（可用 weibo|full）" >&2
    exit 2
    ;;
esac

cd "$PROJECT_DIR"
if "${cmd[@]}" >"$log" 2>&1; then
  # 只保留每个模式最近 30 份日志，避免长期累积
  ls -1t "$LOGS_DIR"/refresh-${MODE}-*.log 2>/dev/null | tail -n +31 | xargs -r rm -f
  echo "[$stamp] OK mode=$MODE -> $log"
else
  echo "[$stamp] FAIL mode=$MODE (tail $log):"
  tail -n 20 "$log" >&2
  exit 1
fi