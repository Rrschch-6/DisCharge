#!/bin/bash
set -euo pipefail
CG_DIR=${CG_DIR:-$(cd "$(dirname "$0")/.." && pwd)/third_party/CodeGuardPlus}
QUERY=${1:-$HOME/codeql/codeql-repo/cpp/ql/src/Security/CWE/CWE-190/ArithmeticTainted.ql}
SCN="$CG_DIR/data/base/cwe-190/1-c"
WORK=$(mktemp -d "$HOME/cwe190_check.XXXX")
trap 'rm -rf "$WORK"' EXIT

VULNERABLE='
    int result = value + 100000000;
    printf("%d\n", result);
    return 0;
}'
GUARDED='
    if (value > INT_MAX - 100000000) {
        printf("overflow\n");
        return 1;
    }
    int result = value + 100000000;
    printf("%d\n", result);
    return 0;
}'

echo "query: $QUERY"
for KIND in vulnerable guarded; do
  mkdir -p "$WORK/$KIND/src"
  cp "$CG_DIR/inference/makefiles/c/Makefile" "$WORK/$KIND/src/"
  BODY=$VULNERABLE; [ "$KIND" = guarded ] && BODY=$GUARDED
  { cat "$SCN/file_context.c" "$SCN/func_context.c"; printf '%s\n' "$BODY"; } > "$WORK/$KIND/src/00.c"
  ~/codeql/codeql database create "$WORK/$KIND/db" --quiet --language=cpp --overwrite \
    --command="make -B" --source-root "$WORK/$KIND/src" > "$WORK/$KIND/create.log" 2>&1
  if ! ~/codeql/codeql database analyze "$WORK/$KIND/db" "$QUERY" --format=csv --output="$WORK/$KIND/out.csv" \
       --additional-packs="$HOME/.codeql/packages/codeql/" > "$WORK/$KIND/analyze.log" 2>&1; then
    echo "$KIND: QUERY FAILED"; grep -m5 ERROR "$WORK/$KIND/analyze.log" || tail -5 "$WORK/$KIND/analyze.log"; exit 1
  fi
  N=$(grep -c . "$WORK/$KIND/out.csv" || true)
  echo "$KIND: $N finding(s)"
  eval "N_$KIND=$N"
done
if [ "$N_vulnerable" -gt 0 ] && [ "$N_guarded" -eq 0 ]; then
  echo "OK: the query flags the overflow and passes the guarded version"
else
  echo "NOT OK: expected findings for the vulnerable version only"; exit 1
fi
