#!/bin/bash

# git_autopush.sh
# Usage: ./git_autopush.sh "Your commit message"
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

# Stage all changes
git add .

# Commit with message
git commit -m "$COMMIT_MSG"

# Push using HTTPS with token
REPO_URL=$(git config --get remote.origin.url)
TOKEN_URL=$(echo "$REPO_URL" | sed "s|https://|https://walidnegm:${GITHUB_TOKEN}@|")

git push "$TOKEN_URL"

echo "✅ Git push completed."

