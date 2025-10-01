#!/usr/bin/env bash
# git-smart-push.sh
# A tiny helper to add, commit, and push — with interactive recovery on push rejects.

set -o pipefail

# ---------- helpers ----------
say() { printf "%b\n" "$*"; }
ask() { read -r -p "$1" REPLY; printf "%s" "$REPLY"; }
die() { say "❌ $*"; exit 1; }

need_git() {
  command -v git >/dev/null 2>&1 || die "git not found. Please install git."
}

ensure_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Not inside a git repository."
}

current_branch() {
  git rev-parse --abbrev-ref HEAD
}

ensure_origin() {
  if ! git remote get-url origin >/dev/null 2>&1; then
    say "No 'origin' remote is set."
    url="$(ask '→ Enter remote URL (e.g., git@github.com:user/repo.git): ')"
    [ -n "$url" ] || die "Remote URL required."
    git remote add origin "$url" || die "Failed to add remote."
    say "✅ Added origin: $url"
  fi
}

have_changes() {
  # any changes staged or unstaged?
  if [[ -n "$(git status --porcelain)" ]]; then
    return 0
  else
    return 1
  fi
}

ensure_staged() {
  if [[ -z "$(git diff --cached --name-only)" ]]; then
    say "No files staged. Staging all tracked/untracked changes..."
    git add -A || die "git add failed."
  fi
}

commit_if_needed() {
  # If there are staged changes but no commit yet, prompt for message.
  if [[ -n "$(git diff --cached --name-only)" ]]; then
    if [[ -n "$1" ]]; then
      msg="$*"
    else
      msg="$(ask "→ Commit message: ")"
      [[ -n "$msg" ]] || msg="update"
    fi
    git commit -m "$msg" || {
      # If commit failed because nothing to commit, continue.
      if git diff --cached --quiet; then
        say "ℹ️ Nothing to commit (working tree clean)."
      else
        die "git commit failed."
      fi
    }
  else
    say "ℹ️ Nothing staged to commit."
  fi
}

push_with_menu() {
  local branch="$1"
  local push_output
  local status

  say "🚀 Pushing to origin/$branch..."
  # capture stderr+stdout and the real exit code via PIPESTATUS
  push_output="$(git push -u origin "$branch" 2>&1 | tee /tmp/git_smart_push.log)"
  status=${PIPESTATUS[0]}

  if [[ $status -eq 0 ]]; then
    say "✅ Push successful."
    return 0
  fi

  say "⚠️  Push failed."
  # Detect common cases
  if grep -qiE "non-fast-forward|fetch first|rejected|Updates were rejected" <<<"$push_output"; then
    say ""
    say "It looks like the remote has commits you don't have."
    say "Choose an action:"
    say "  [1] Pull with rebase, then push"
    say "  [2] Force push (with lease) — overwrite remote history if safe"
    say "  [3] Abort"
    choice="$(ask "→ Enter 1/2/3: ")"
    case "$choice" in
      1)
        say "📥 Rebase on top of origin/$branch..."
        git fetch origin || die "git fetch failed."
        git pull --rebase origin "$branch" || die "git pull --rebase failed (resolve conflicts, then re-run script)."
        say "🔁 Retrying push..."
        git push -u origin "$branch" || die "Push still failing."
        ;;
      2)
        say "⚠️ Forcing push with lease (safer than --force)..."
        git push --force-with-lease -u origin "$branch" || die "Force push failed."
        ;;
      *)
        die "Aborted."
        ;;
    esac
  elif grep -qi "no configured push destination" <<<"$push_output"; then
    say "No upstream is set for this branch; setting it now…"
    git push -u origin "$branch" || die "Failed to set upstream."
  else
    say "Raw git error:"
    echo "$push_output"
    die "Unknown push error. See output above."
  fi
}

# ---------- main ----------
usage() {
  cat <<EOF
Usage:
  $(basename "$0") [commit message]

If no commit message is provided, you'll be prompted.
This script:
  1) Ensures you're in a git repo and 'origin' exists
  2) Stages changes (git add -A) if nothing is staged
  3) Commits with the provided (or prompted) message
  4) Pushes to origin/<current-branch>, and if rejected, offers:
     - Pull --rebase then push
     - Force push (with lease)
     - Abort
EOF
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

need_git
ensure_repo
ensure_origin

BRANCH="$(current_branch)"
[[ -n "$BRANCH" ]] || die "Could not determine current branch."

if have_changes; then
  ensure_staged
  commit_if_needed "$@"
else
  say "ℹ️ Working tree clean — nothing to commit."
fi

push_with_menu "$BRANCH"