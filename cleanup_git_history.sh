#!/bin/bash
# Git Repository Cleanup Script
# This script removes large files from Git history to reduce repository size

set -e

echo "=========================================="
echo "Git Repository Cleanup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo "Error: Not in a Git repository root"
    exit 1
fi

echo "Step 1: Creating backup branch..."
git branch backup-before-cleanup 2>/dev/null || echo "Backup branch already exists"

echo ""
echo "Step 2: Removing large files from current index..."
# Remove any currently tracked large files
git rm -r --cached --ignore-unmatch "*.pth" "*.mat" "*.jpg" "*.png" "*.dir" "models/" "plots/" "runs/" "results/" "Pytorch_scripts/runs/" "Pytorch_scripts/plots/" "Pytorch_scripts/models/" "Pytorch_scripts/results/" "Pytorch_scripts/Autoencoder_*/" "Pytorch_scripts/Misc_outputs*/" 2>/dev/null || true

# Re-add the exception file
git add -f GSI_header_table.mat 2>/dev/null || true

echo ""
echo "Step 3: Checking for BFG Repo-Cleaner..."
if ! command -v bfg &> /dev/null; then
    echo "BFG Repo-Cleaner not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install bfg
    else
        echo "ERROR: Homebrew not found. Please install BFG manually:"
        echo "  brew install bfg"
        echo "  OR download from: https://rtyley.github.io/bfg-repo-cleaner/"
        exit 1
    fi
fi

echo ""
echo "Step 4: Removing large files from history using BFG..."
echo "This will remove files larger than 1MB from all history..."

# BFG will remove files > 1M from history (except HEAD)
bfg --strip-blobs-bigger-than 1M --no-blob-protection .

echo ""
echo "Step 5: Cleaning up Git repository..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Step 6: Verifying cleanup..."
echo "Largest files remaining in history:"
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print $3, $4}' | \
  sort -rn | \
  head -10

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
echo ""
echo "IMPORTANT NEXT STEPS:"
echo "1. Review changes: git status"
echo "2. Commit the .gitignore update: git add .gitignore && git commit -m 'Update .gitignore to exclude large files'"
echo "3. Force push to remote: git push origin OB-branch --force"
echo ""
echo "WARNING: This rewrites history! Coordinate with team members."
echo "If something goes wrong, restore from: git checkout backup-before-cleanup"
echo ""
