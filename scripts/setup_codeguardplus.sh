#!/bin/bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
CG_DIR=${CG_DIR:-$REPO_DIR/third_party/CodeGuardPlus}
CG_COMMIT=0528ee9e527fa57b9a1c82c564beb854f1881fd5          # CodeGuardPlus/CodeGuardPlus commit used for all results in the paper
QUERY=cpp/ql/src/Security/CWE/CWE-190/ArithmeticTainted.ql

if [ ! -d "$CG_DIR/.git" ]; then
  git clone https://github.com/CodeGuardPlus/CodeGuardPlus.git "$CG_DIR"
fi
git -C "$CG_DIR" checkout --quiet "$CG_COMMIT"
echo "CodeGuard+ at $(git -C "$CG_DIR" rev-parse --short HEAD) in $CG_DIR"

if [ -x "$HOME/codeql/codeql" ] && [ -d "$HOME/codeql/codeql-repo" ]; then
  echo "CodeQL already installed in ~/codeql ($("$HOME/codeql/codeql" version --format=terse))"
else
  (cd "$CG_DIR" && bash setup_codeql.sh)
fi
"$HOME/codeql/codeql" version --format=terse | grep -q '^2\.17\.5$' \
  || echo "WARNING: CodeQL in ~/codeql is not 2.17.5; results may differ from the paper"

git -C "$HOME/codeql/codeql-repo" checkout -- "$QUERY"
echo "restored CodeQL's standard $QUERY"
CG_DIR="$CG_DIR" bash "$REPO_DIR/scripts/check_cwe190_query.sh"
echo "CodeGuard+ is ready. Set up its Python environment next (README, Setup step 3)."
