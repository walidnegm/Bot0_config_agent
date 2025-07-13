#!/bin/bash

# git_autopush.sh
# Usage: ./git_autopush.sh "Your commit message"

if [ -z "$1" ]; then
  echo "Usage: $0 \"Your commit message\""
  exit 1
fi

COMMIT_MSG="$1"

# Stage all changes
git add .

# Commit with message
git commit -m "$COMMIT_MSG"

# Push to current branch
git push

echo "✅ Git push completed."

