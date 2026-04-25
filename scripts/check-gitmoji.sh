#!/usr/bin/env bash
# pre-commit `commit-msg` hook: enforce gitmoji shortcode in commit subject.
# Merge / Revert / fixup! / squash! / amend! commits are exempt.
set -e
subject=$(head -n1 "$1")
if [[ "$subject" =~ ^(Merge\ |Revert\ |fixup!\ |squash!\ |amend!\ |:[a-z0-9_]+:\ ) ]]; then
  exit 0
fi
echo "ERROR: commit subject must start with a gitmoji shortcode."
echo "       e.g.  :sparkles: Add foo"
echo "       See CLAUDE.md for the shortcode list."
exit 1
