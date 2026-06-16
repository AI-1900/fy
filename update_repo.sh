#!/usr/bin/env bash
set -e

# ============================================================
# 最简单版：自动提交本地所有修改，并推送到远端 main 分支
# 使用前只需要修改 REPO_DIR 为你的本地 git 仓库路径
# ============================================================

# 1. 当前待 git 管理的本地仓库路径
REPO_DIR="/home/luke/00meta/fy"

# 2. 远端仓库名，一般是 origin
REMOTE="origin"

# 3. 目标分支：远端 main 分支
BRANCH="main"

# 4. commit message
COMMIT_MSG="auto commit: $(date '+%Y-%m-%d %H:%M:%S')"

# 5. 进入仓库
cd "$REPO_DIR"

# 6. 确保当前是 git 仓库
git rev-parse --is-inside-work-tree >/dev/null

# 7. 切换到 main 分支
git checkout "$BRANCH"

# 8. 拉取远端 main 最新代码，避免直接 push 冲突
# 如果有冲突，这一步会失败，需要手动解决冲突
git pull --rebase "$REMOTE" "$BRANCH"

# 9. 添加所有修改，包括新增、修改、删除文件
git add -A

# 10. 如果没有任何修改，则直接退出
if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

# 11. 提交本地所有修改
git commit -m "$COMMIT_MSG"

# 12. 推送到远端 main 分支
git push "$REMOTE" "$BRANCH"

echo "Auto commit and push to $REMOTE/$BRANCH done."