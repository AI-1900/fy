#!/usr/bin/env bash
# ==============================================================================
# 文件名: git_heartbeat_single_script.sh
# 目标: 一个脚本完成 Git 仓库每日 03:00 自动检查、拉取远端、写 heartbeat、提交并 push。
#
# 使用方式概览:
#   1) 修改【第 1 段：用户配置区】中的 REPO_DIR 为你要管理的本地仓库绝对路径。
#   2) chmod +x ./git_heartbeat_single_script.sh
#   3) ./git_heartbeat_single_script.sh --doctor        # 检查 Git / 仓库 / 远端 / 用户配置
#   4) ./git_heartbeat_single_script.sh --run           # 手动执行一次 heartbeat
#   5) ./git_heartbeat_single_script.sh --install-cron  # 安装每日 03:00 定时任务
#   6) ./git_heartbeat_single_script.sh --status        # 查看本地与远端状态
#
# 定时任务策略:
#   - 本脚本默认使用 cron 安装定时任务，不依赖额外 systemd service/timer 文件。
#   - cron 只保存一行命令，真正逻辑全部在本脚本内。
#
# 本地变更策略:
#   - 默认 LOCAL_CHANGE_MODE=stash。
#   - 如果本地存在未提交修改，先 git stash push -u，完成 heartbeat 后再 git stash pop。
#   - 不会默认把你的业务代码一起提交，避免误提交密钥、临时代码或 build 产物。
#
# 远端更新策略:
#   - git fetch --prune REMOTE
#   - 如果远端有更新，执行 git pull --rebase REMOTE BRANCH
#   - 然后写入 .heartbeat/heartbeat.log 与 .heartbeat/last.json
#   - 生成并 push 一条 heartbeat commit
# ===============================================================================

set -Eeuo pipefail

# ==============================================================================
# 第 1 段：用户配置区
# ===============================================================================
# 【当前待 Git 管理的仓库】
# 必须改成你的本地 repo 绝对路径，例如：
#   REPO_DIR="$HOME/work/my_repo"
# 或者运行时覆盖：
#   REPO_DIR=/abs/path/to/repo ./git_heartbeat_single_script.sh --run
# REPO_DIR="${REPO_DIR:-$HOME/work/my_repo}"
REPO_DIR="/home/luke/00meta/fy"

# 【远端设置】
# REMOTE 一般是 origin。
REMOTE="${REMOTE:-origin}"

# 【分支设置】
# 留空表示使用当前 checked-out 分支。
# 显式指定示例：BRANCH="main" 或 BRANCH="master"。
# BRANCH="${BRANCH:-}"
BRANCH="main"


# 【定时任务设置】
# cron 表达式：分 时 日 月 周。
# 默认每天 03:00 执行。
CRON_SCHEDULE="${CRON_SCHEDULE:-0 3 * * *}"

# 【heartbeat 文件设置】
# 这两个文件会被 git 管理并提交到远端。
HEARTBEAT_DIR="${HEARTBEAT_DIR:-.heartbeat}"
HEARTBEAT_LOG_FILE="${HEARTBEAT_LOG_FILE:-$HEARTBEAT_DIR/heartbeat.log}"
HEARTBEAT_JSON_FILE="${HEARTBEAT_JSON_FILE:-$HEARTBEAT_DIR/last.json}"

# 【本地未提交变更处理策略】
#   stash  : 推荐。先 stash 本地修改，完成 heartbeat 后恢复。
#   skip   : 如果本地有 dirty changes，直接跳过本次 heartbeat。
#   commit : 自动提交所有本地改动。不推荐，除非你明确知道自己在做什么。
LOCAL_CHANGE_MODE="${LOCAL_CHANGE_MODE:-stash}"

# 【拉取远端策略】
#   rebase : 推荐。git pull --rebase REMOTE BRANCH
#   merge  : git pull REMOTE BRANCH
PULL_MODE="${PULL_MODE:-rebase}"

# 【push 失败重试次数】
# push 期间如果远端刚好被别人更新，脚本会重新 fetch/pull 后重试。
PUSH_RETRY="${PUSH_RETRY:-2}"

# 【commit message 前缀】
COMMIT_PREFIX="${COMMIT_PREFIX:-chore: heartbeat}"

# 【日志文件】
LOG_FILE="${LOG_FILE:-$HOME/.cache/git-heartbeat/git-heartbeat.log}"

# 【可选：为当前仓库设置 git user】
# 如果为空，则使用已有的 repo/global git config。
# 示例：
#   GIT_USER_NAME="heartbeat-bot"
#   GIT_USER_EMAIL="heartbeat-bot@example.com"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

# ==============================================================================
# 第 2 段：内部变量与基础工具函数
# ===============================================================================
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
GIT_BIN="${GIT_BIN:-git}"
TARGET_BRANCH=""
LOCK_FILE=""
STASH_CREATED=0
STASH_RESTORED=0
STASH_NAME=""

now_iso() {
  date -Iseconds
}

log() {
  local msg="$*"
  mkdir -p "$(dirname "$LOG_FILE")"
  printf '[%s] %s\n' "$(now_iso)" "$msg" | tee -a "$LOG_FILE" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"
}

git_in_repo() {
  "$GIT_BIN" -C "$REPO_DIR" "$@"
}

shell_quote() {
  printf '%q' "$1"
}

json_escape() {
  # 简单 JSON 字符串转义，避免路径、host、branch 中的引号破坏 JSON。
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

sanitize_for_file() {
  printf '%s' "$1" | sed 's#[^A-Za-z0-9_.-]#_#g'
}

show_help() {
  cat <<USAGE
用法:
  $0 --run             手动执行一次：检查本地、检查远端、pull、写 heartbeat、commit、push
  $0 --install-cron    安装每日 03:00 cron 定时任务，时间由 CRON_SCHEDULE 控制
  $0 --uninstall-cron  删除本脚本安装的 cron 定时任务
  $0 --status          查看当前仓库、本地 dirty、远端 ahead/behind 状态
  $0 --doctor          检查依赖、仓库、remote、branch、git user 配置
  $0 --print-config    打印当前配置
  $0 --print-cron      打印将要安装的 cron 行
  $0 --help            显示帮助

最小示例:
  chmod +x $0
  REPO_DIR=/abs/path/to/repo $0 --doctor
  REPO_DIR=/abs/path/to/repo $0 --run
  REPO_DIR=/abs/path/to/repo $0 --install-cron

推荐写法:
  直接修改脚本顶部 REPO_DIR，然后执行：
    $0 --doctor
    $0 --run
    $0 --install-cron

当前配置:
  REPO_DIR             = $REPO_DIR
  REMOTE               = $REMOTE
  BRANCH               = ${BRANCH:-<当前分支>}
  CRON_SCHEDULE        = $CRON_SCHEDULE
  LOCAL_CHANGE_MODE    = $LOCAL_CHANGE_MODE
  PULL_MODE            = $PULL_MODE
  HEARTBEAT_LOG_FILE   = $HEARTBEAT_LOG_FILE
  HEARTBEAT_JSON_FILE  = $HEARTBEAT_JSON_FILE
  LOG_FILE             = $LOG_FILE
USAGE
}

# ==============================================================================
# 第 3 段：配置检查与 Git 环境检测
# ===============================================================================
resolve_target_branch() {
  local current_branch
  current_branch="$(git_in_repo branch --show-current 2>/dev/null || true)"

  if [[ -n "$BRANCH" ]]; then
    TARGET_BRANCH="$BRANCH"
  else
    TARGET_BRANCH="$current_branch"
  fi

  [[ -n "$TARGET_BRANCH" ]] || die "无法确定目标分支。请设置 BRANCH=main/master，或切到一个非 detached HEAD 分支。"
}

validate_common() {
  need_cmd "$GIT_BIN"
  need_cmd sed
  need_cmd date

  [[ -n "$REPO_DIR" ]] || die "REPO_DIR 为空，请在脚本顶部配置待管理仓库绝对路径。"
  [[ -d "$REPO_DIR" ]] || die "REPO_DIR 不存在: $REPO_DIR"
  [[ -d "$REPO_DIR/.git" || "$(git_in_repo rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] \
    || die "不是 Git 仓库: $REPO_DIR"

  resolve_target_branch

  if ! git_in_repo remote get-url "$REMOTE" >/dev/null 2>&1; then
    die "remote 不存在: $REMOTE。请先执行: git -C $(shell_quote "$REPO_DIR") remote add $REMOTE <url>"
  fi

  case "$LOCAL_CHANGE_MODE" in
    stash|skip|commit) ;;
    *) die "LOCAL_CHANGE_MODE 只能是 stash / skip / commit，当前是: $LOCAL_CHANGE_MODE" ;;
  esac

  case "$PULL_MODE" in
    rebase|merge) ;;
    *) die "PULL_MODE 只能是 rebase / merge，当前是: $PULL_MODE" ;;
  esac
}

validate_for_run() {
  validate_common
  need_cmd flock
  need_cmd hostname

  # 如果显式指定 BRANCH，确保当前工作区切到目标分支。
  local current_branch
  current_branch="$(git_in_repo branch --show-current 2>/dev/null || true)"
  if [[ "$current_branch" != "$TARGET_BRANCH" ]]; then
    log "当前分支是 ${current_branch:-detached}，切换到目标分支: $TARGET_BRANCH"
    git_in_repo switch "$TARGET_BRANCH"
  fi

  # 可选：设置当前 repo 的 git user，避免 cron 环境下 commit 失败。
  if [[ -n "$GIT_USER_NAME" ]]; then
    git_in_repo config user.name "$GIT_USER_NAME"
  fi
  if [[ -n "$GIT_USER_EMAIL" ]]; then
    git_in_repo config user.email "$GIT_USER_EMAIL"
  fi

  local user_name user_email
  user_name="$(git_in_repo config user.name || true)"
  user_email="$(git_in_repo config user.email || true)"
  [[ -n "$user_name" ]] || die "git user.name 未配置。执行: git -C $(shell_quote "$REPO_DIR") config user.name 'your-name'"
  [[ -n "$user_email" ]] || die "git user.email 未配置。执行: git -C $(shell_quote "$REPO_DIR") config user.email 'you@example.com'"

  local repo_key
  repo_key="$(sanitize_for_file "$REPO_DIR")"
  LOCK_FILE="${LOCK_FILE:-/tmp/git_heartbeat_${USER:-user}_${repo_key}.lock}"
}

print_config() {
  validate_common
  cat <<CONFIG
当前脚本配置:
  SCRIPT_PATH          = $SCRIPT_PATH
  REPO_DIR             = $REPO_DIR
  REMOTE               = $REMOTE
  BRANCH               = ${BRANCH:-<当前分支>}
  TARGET_BRANCH        = $TARGET_BRANCH
  CRON_SCHEDULE        = $CRON_SCHEDULE
  LOCAL_CHANGE_MODE    = $LOCAL_CHANGE_MODE
  PULL_MODE            = $PULL_MODE
  PUSH_RETRY           = $PUSH_RETRY
  HEARTBEAT_DIR        = $HEARTBEAT_DIR
  HEARTBEAT_LOG_FILE   = $HEARTBEAT_LOG_FILE
  HEARTBEAT_JSON_FILE  = $HEARTBEAT_JSON_FILE
  COMMIT_PREFIX        = $COMMIT_PREFIX
  LOG_FILE             = $LOG_FILE
  GIT_USER_NAME        = ${GIT_USER_NAME:-<使用已有 git config>}
  GIT_USER_EMAIL       = ${GIT_USER_EMAIL:-<使用已有 git config>}
CONFIG
}

run_doctor() {
  validate_common
  log "doctor: 依赖检查通过"
  log "doctor: repo=$(git_in_repo rev-parse --show-toplevel)"
  log "doctor: branch=$TARGET_BRANCH"
  log "doctor: remote.$REMOTE=$(git_in_repo remote get-url "$REMOTE")"

  local user_name user_email
  user_name="$(git_in_repo config user.name || true)"
  user_email="$(git_in_repo config user.email || true)"
  if [[ -z "$user_name" || -z "$user_email" ]]; then
    log "doctor: git user 未完整配置，commit 可能失败"
    log "doctor: 建议执行：git -C $(shell_quote "$REPO_DIR") config user.name 'your-name'"
    log "doctor: 建议执行：git -C $(shell_quote "$REPO_DIR") config user.email 'you@example.com'"
  else
    log "doctor: git user.name=$user_name"
    log "doctor: git user.email=$user_email"
  fi

  if git_in_repo ls-remote --exit-code --heads "$REMOTE" "$TARGET_BRANCH" >/dev/null 2>&1; then
    log "doctor: 远端分支存在: $REMOTE/$TARGET_BRANCH"
  else
    log "doctor: 注意：远端分支可能不存在或无权限访问: $REMOTE/$TARGET_BRANCH"
  fi

  log "doctor: 检查完成"
}

# ==============================================================================
# 第 4 段：状态查看，本地与远端差异检测
# ===============================================================================
print_status() {
  validate_common

  echo "仓库状态:"
  echo "  REPO_DIR      = $REPO_DIR"
  echo "  REMOTE        = $REMOTE"
  echo "  TARGET_BRANCH = $TARGET_BRANCH"
  echo "  HEAD          = $(git_in_repo rev-parse --short HEAD)"
  echo

  echo "本地工作区 dirty 检测:"
  if [[ -n "$(git_in_repo status --porcelain=v1)" ]]; then
    echo "  dirty = yes"
    git_in_repo status --short
  else
    echo "  dirty = no"
  fi
  echo

  echo "远端更新检测:"
  git_in_repo fetch --prune "$REMOTE" >/dev/null 2>&1 || {
    echo "  fetch failed: 请检查网络、SSH key、remote 权限"
    return 1
  }

  local remote_ref="$REMOTE/$TARGET_BRANCH"
  if git_in_repo rev-parse --verify "$remote_ref" >/dev/null 2>&1; then
    local counts ahead behind
    counts="$(git_in_repo rev-list --left-right --count "HEAD...$remote_ref")"
    ahead="$(printf '%s' "$counts" | awk '{print $1}')"
    behind="$(printf '%s' "$counts" | awk '{print $2}')"
    echo "  remote_ref = $remote_ref"
    echo "  local ahead remote  = $ahead commit(s)"
    echo "  local behind remote = $behind commit(s)"
  else
    echo "  remote_ref 不存在: $remote_ref"
  fi
}

# ==============================================================================
# 第 5 段：本地未提交改动处理
# ===============================================================================
handle_local_changes_before() {
  local status
  status="$(git_in_repo status --porcelain=v1)"

  if [[ -z "$status" ]]; then
    log "local: 工作区干净，无本地未提交修改"
    return 0
  fi

  log "local: 检测到本地未提交修改"
  git_in_repo status --short | tee -a "$LOG_FILE" >&2

  case "$LOCAL_CHANGE_MODE" in
    stash)
      STASH_NAME="git-heartbeat-auto-stash $(now_iso)"
      log "local: LOCAL_CHANGE_MODE=stash，先暂存本地改动: $STASH_NAME"
      git_in_repo stash push -u -m "$STASH_NAME"
      STASH_CREATED=1
      ;;
    skip)
      log "local: LOCAL_CHANGE_MODE=skip，跳过本次 heartbeat，不修改仓库"
      exit 0
      ;;
    commit)
      log "local: LOCAL_CHANGE_MODE=commit，自动提交本地全部改动。注意：该模式可能误提交临时代码或敏感文件。"
      git_in_repo add -A
      if git_in_repo diff --cached --quiet; then
        log "local: add 后没有可提交内容"
      else
        git_in_repo commit -m "chore: auto-save local changes before heartbeat $(date '+%Y-%m-%d %H:%M:%S %z')"
      fi
      ;;
  esac
}

restore_local_changes_after_success() {
  if [[ "$STASH_CREATED" -eq 1 && "$STASH_RESTORED" -eq 0 ]]; then
    log "local: heartbeat 完成，恢复之前 stash 的本地改动"
    if git_in_repo stash pop; then
      STASH_RESTORED=1
      log "local: stash 恢复完成"
    else
      log "local: stash pop 失败，可能发生冲突。请手动检查: git -C $(shell_quote "$REPO_DIR") status"
      log "local: 可查看 stash: git -C $(shell_quote "$REPO_DIR") stash list"
      exit 1
    fi
  fi
}

cleanup_on_exit() {
  local code=$?
  if [[ "$code" -ne 0 && "$STASH_CREATED" -eq 1 && "$STASH_RESTORED" -eq 0 ]]; then
    log "cleanup: 脚本异常退出，为避免二次冲突，本地改动仍保留在 stash 中"
    log "cleanup: 请手动恢复: git -C $(shell_quote "$REPO_DIR") stash pop"
  fi
}

# ==============================================================================
# 第 6 段：远端更新检测与同步
# ===============================================================================
fetch_and_pull_remote() {
  log "remote: fetch --prune $REMOTE"
  git_in_repo fetch --prune "$REMOTE"

  local remote_ref="$REMOTE/$TARGET_BRANCH"
  if ! git_in_repo rev-parse --verify "$remote_ref" >/dev/null 2>&1; then
    log "remote: 远端分支 $remote_ref 不存在。后续 push 会尝试创建/设置该分支。"
    return 0
  fi

  local local_sha remote_sha base_sha
  local_sha="$(git_in_repo rev-parse HEAD)"
  remote_sha="$(git_in_repo rev-parse "$remote_ref")"
  base_sha="$(git_in_repo merge-base HEAD "$remote_ref")"

  if [[ "$local_sha" == "$remote_sha" ]]; then
    log "remote: 本地与远端一致，无需 pull"
    return 0
  fi

  if [[ "$local_sha" == "$base_sha" ]]; then
    log "remote: 远端领先本地，开始同步远端更新到本地"
  elif [[ "$remote_sha" == "$base_sha" ]]; then
    log "remote: 本地领先远端，无需 pull，后续直接 push heartbeat"
    return 0
  else
    log "remote: 本地与远端发生分叉，按 PULL_MODE=$PULL_MODE 处理"
  fi

  if [[ "$PULL_MODE" == "rebase" ]]; then
    log "remote: git pull --rebase $REMOTE $TARGET_BRANCH"
    git_in_repo pull --rebase "$REMOTE" "$TARGET_BRANCH"
  else
    log "remote: git pull $REMOTE $TARGET_BRANCH"
    git_in_repo pull "$REMOTE" "$TARGET_BRANCH"
  fi
}

# ==============================================================================
# 第 7 段：生成 heartbeat 文件并提交
# ===============================================================================
write_heartbeat_files() {
  local now host user_name user_email head_short remote_url top_level
  now="$(now_iso)"
  host="$(hostname -f 2>/dev/null || hostname)"
  user_name="$(git_in_repo config user.name || true)"
  user_email="$(git_in_repo config user.email || true)"
  head_short="$(git_in_repo rev-parse --short HEAD)"
  remote_url="$(git_in_repo remote get-url "$REMOTE")"
  top_level="$(git_in_repo rev-parse --show-toplevel)"

  mkdir -p "$REPO_DIR/$HEARTBEAT_DIR"

  # 追加日志：保留历史 heartbeat 记录。
  {
    printf '%s | host=%s | user=%s <%s> | repo=%s | branch=%s | remote=%s | head_before=%s\n' \
      "$now" "$host" "$user_name" "$user_email" "$top_level" "$TARGET_BRANCH" "$REMOTE" "$head_short"
  } >> "$REPO_DIR/$HEARTBEAT_LOG_FILE"

  # 覆盖 last.json：保留最后一次 heartbeat 状态，便于脚本或监控读取。
  cat > "$REPO_DIR/$HEARTBEAT_JSON_FILE" <<JSON
{
  "timestamp": "$(json_escape "$now")",
  "host": "$(json_escape "$host")",
  "repo_dir": "$(json_escape "$top_level")",
  "remote": "$(json_escape "$REMOTE")",
  "remote_url": "$(json_escape "$remote_url")",
  "branch": "$(json_escape "$TARGET_BRANCH")",
  "head_before": "$(json_escape "$head_short")",
  "git_user_name": "$(json_escape "$user_name")",
  "git_user_email": "$(json_escape "$user_email")"
}
JSON

  log "heartbeat: 已写入 $HEARTBEAT_LOG_FILE 和 $HEARTBEAT_JSON_FILE"
}

commit_heartbeat() {
  git_in_repo add "$HEARTBEAT_LOG_FILE" "$HEARTBEAT_JSON_FILE"

  if git_in_repo diff --cached --quiet; then
    log "commit: 没有 heartbeat 差异，无需提交"
    return 0
  fi

  local msg
  msg="$COMMIT_PREFIX $(date '+%Y-%m-%d %H:%M:%S %z')"
  log "commit: $msg"
  git_in_repo commit -m "$msg"
}

# ==============================================================================
# 第 8 段：push 到远端，失败后 fetch/pull/retry
# ===============================================================================
push_with_retry() {
  local attempt=0
  local max_attempts=$((PUSH_RETRY + 1))

  while (( attempt < max_attempts )); do
    attempt=$((attempt + 1))
    log "push: 第 $attempt/$max_attempts 次 push 到 $REMOTE/$TARGET_BRANCH"

    if git_in_repo push -u "$REMOTE" "HEAD:$TARGET_BRANCH"; then
      log "push: 成功"
      return 0
    fi

    if (( attempt >= max_attempts )); then
      die "push 多次失败，请检查远端权限、分支保护、网络或冲突"
    fi

    log "push: 失败，重新 fetch/pull 后重试"
    fetch_and_pull_remote
  done
}

# ==============================================================================
# 第 9 段：主流程，一个 run 完成所有业务需求
# ===============================================================================
run_once() {
  validate_for_run
  trap cleanup_on_exit EXIT

  # 防止 cron 重入：如果上一次任务没结束，本次直接退出。
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "lock: 已有 heartbeat 任务正在运行，跳过本次执行。lock=$LOCK_FILE"
    exit 0
  fi

  log "run: start"
  log "run: repo=$REPO_DIR remote=$REMOTE branch=$TARGET_BRANCH schedule='$CRON_SCHEDULE'"

  handle_local_changes_before
  fetch_and_pull_remote
  write_heartbeat_files
  commit_heartbeat
  push_with_retry
  restore_local_changes_after_success

  log "run: done"
}

# ==============================================================================
# 第 10 段：cron 定时任务安装/卸载
# ===============================================================================
cron_tag() {
  printf '# git-heartbeat-single-script repo=%s script=%s' "$REPO_DIR" "$SCRIPT_PATH"
}

cron_command_line() {
  local cmd
  cmd="REPO_DIR=$(shell_quote "$REPO_DIR") REMOTE=$(shell_quote "$REMOTE") BRANCH=$(shell_quote "$BRANCH") LOCAL_CHANGE_MODE=$(shell_quote "$LOCAL_CHANGE_MODE") PULL_MODE=$(shell_quote "$PULL_MODE") HEARTBEAT_DIR=$(shell_quote "$HEARTBEAT_DIR") HEARTBEAT_LOG_FILE=$(shell_quote "$HEARTBEAT_LOG_FILE") HEARTBEAT_JSON_FILE=$(shell_quote "$HEARTBEAT_JSON_FILE") LOG_FILE=$(shell_quote "$LOG_FILE") $(shell_quote "$SCRIPT_PATH") --run >> $(shell_quote "$LOG_FILE") 2>&1"
  printf '%s %s' "$CRON_SCHEDULE" "$cmd"
}

print_cron() {
  validate_common
  echo "$(cron_tag)"
  echo "$(cron_command_line)"
}

install_cron() {
  validate_common
  need_cmd crontab

  if [[ ! -x "$SCRIPT_PATH" ]]; then
    log "install-cron: 当前脚本不可执行，自动 chmod +x $SCRIPT_PATH"
    chmod +x "$SCRIPT_PATH"
  fi

  local tmp
  tmp="$(mktemp)"

  # 删除旧的同类任务，再追加新的任务。
  # 约定：tag 行的下一行就是真正 cron 命令。
  if crontab -l >/dev/null 2>&1; then
    crontab -l | awk '
      /# git-heartbeat-single-script repo=/ {skip_next=1; next}
      skip_next==1 {skip_next=0; next}
      {print}
    ' > "$tmp"
  else
    : > "$tmp"
  fi

  {
    cat "$tmp"
    echo "$(cron_tag)"
    echo "$(cron_command_line)"
  } | crontab -

  rm -f "$tmp"

  log "install-cron: 已安装定时任务"
  log "install-cron: $(cron_tag)"
  log "install-cron: $(cron_command_line)"
}

uninstall_cron() {
  need_cmd crontab

  local tmp
  tmp="$(mktemp)"

  if crontab -l >/dev/null 2>&1; then
    crontab -l | awk '
      /# git-heartbeat-single-script repo=/ {skip_next=1; next}
      skip_next==1 {skip_next=0; next}
      {print}
    ' > "$tmp"
    crontab "$tmp"
    log "uninstall-cron: 已删除 git-heartbeat-single-script 相关 cron 任务"
  else
    log "uninstall-cron: 当前用户没有 crontab，无需删除"
  fi

  rm -f "$tmp"
}

# ==============================================================================
# 第 11 段：入口参数分发
# ===============================================================================
main() {
  local action="${1:---help}"

  case "$action" in
    --run)
      run_once
      ;;
    --install-cron)
      install_cron
      ;;
    --uninstall-cron)
      uninstall_cron
      ;;
    --status)
      print_status
      ;;
    --doctor)
      run_doctor
      ;;
    --print-config)
      print_config
      ;;
    --print-cron)
      print_cron
      ;;
    --help|-h|help)
      show_help
      ;;
    *)
      show_help
      die "未知参数: $action"
      ;;
  esac
}

main "$@"
