#!/bin/bash

# autopullpush.sh
# Usage: ./autopullpush.sh "Your commit message"
# Requires: export GITHUB_TOKEN=your_token

if [ -z "$1" ]; then
  echo "Usage: $0 \"Your commit message\""
  exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
  echo "❌ GITHUB_TOKEN is not set. Run: export GITHUB_TOKEN=your_token"
  exit 1
fi

COMMIT_MSG="$1"

# Step 1: Stash local changes (tracked + untracked)
echo "📦 Stashing local changes..."
git stash push --include-untracked -m "autopullpush stash"

# Step 2: Pull latest changes with rebase
echo "📥 Pulling latest changes from origin/master..."
git pull origin master --rebase
if [ $? -ne 0 ]; then
  echo "❌ Git pull failed. Resolve conflicts and try again."
  exit 1
fi

# Step 3: Re-apply local changes
echo "📤 Re-applying stashed changes..."
git stash pop

# Step 4: Stage and commit changes (if any)
if ! git diff --cached --quiet || ! git diff --quiet; then
  git add .
  git commit -m "$COMMIT_MSG"
else
  echo "🟢 No changes to commit."
fi

# Step 5: Push using token
REPO_URL=$(git config --get remote.origin.url)
TOKEN_URL=$(echo "$REPO_URL" | sed "s|https://|https://walidnegm:${GITHUB_TOKEN}@|")

echo "🚀 Pushing to $REPO_URL..."
git push "$TOKEN_URL"

echo "✅ Git pull → rebase → commit → push completed."

