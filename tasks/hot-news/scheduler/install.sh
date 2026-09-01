#!/usr/bin/env bash
# 安装/卸载 hot-news 定时抓取 (macOS launchd)
# 用法:
#   install.sh install   # 由 .plist.tpl 生成正式 plist 并 launchctl bootstrap，立即拉起 weibo（full 明早 9 点首跑）
#   install.sh uninstall # launchctl bootout 并保留已生成的 plist
#   install.sh status    # launchctl print 查看两个 job
set -euo pipefail

ACTION="${1:-install}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOGS_DIR="$PROJECT_DIR/tasks/hot-news/logs"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"
UV_DIR="$(dirname "$(command -v uv || true)")"
UV_PATH="${UV_DIR:+$UV_DIR:}/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

mkdir -p "$LAUNCH_AGENTS" "$LOGS_DIR"

JOBS=(weibo full)

sed_plist() {
  # 将模板占位符替换为真实路径，生成 ~/Library/LaunchAgents/<name>.plist
  local tpl="$1" out="$2"
  sed \
    -e "s|__REFRESH_SCRIPT__|$SCRIPT_DIR/refresh.sh|g" \
    -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__USER_HOME__|$HOME|g" \
    -e "s|__LOG_DIR__|$LOGS_DIR|g" \
    -e "s|__UV_PATH__|$UV_PATH|g" \
    "$tpl" > "$out"
}

case "$ACTION" in
install)
    # 清空旧 launchd 日志，避免误读历史报错
    : > "$LOGS_DIR/launchd-weibo.out.log"; : > "$LOGS_DIR/launchd-weibo.err.log"
    : > "$LOGS_DIR/launchd-full.out.log";  : > "$LOGS_DIR/launchd-full.err.log"
    for j in "${JOBS[@]}"; do
      sed_plist "$SCRIPT_DIR/com.individular.hotnews-$j.plist.tpl" "$LAUNCH_AGENTS/com.individular.hotnews-$j.plist"
      launchctl bootout "gui/$UID_NUM" "$LAUNCH_AGENTS/com.individular.hotnews-$j.plist" 2>/dev/null || true
      launchctl bootstrap "gui/$UID_NUM" "$LAUNCH_AGENTS/com.individular.hotnews-$j.plist"
      echo "✅ 已加载 com.individular.hotnews-$j"
    done
    ;;
  uninstall)
    for j in "${JOBS[@]}"; do
      launchctl bootout "gui/$UID_NUM" "$LAUNCH_AGENTS/com.individular.hotnews-$j.plist" 2>/dev/null || true
      echo "✅ 已卸载 com.individular.hotnews-$j"
    done
    ;;
  status)
    for j in "${JOBS[@]}"; do
      echo "── $j ──"
      launchctl print "gui/$UID_NUM/com.individular.hotnews-$j" || echo "  (未加载)"
    done
    ;;
  *)
    echo "用法: install.sh [install|uninstall|status]" >&2
    exit 2
    ;;
esac