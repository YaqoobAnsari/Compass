#!/bin/bash
# Adopt a newer official CVPR author kit without disturbing the paper.
#
# Only the two vendored formatting files are replaced, cvpr.sty and
# ieeenat_fullname.bst, because those are what the submission is checked
# against. Everything we write (main.tex, preamble.tex, sec/*, main.bib) is
# left alone, and the script reports how the upstream scaffolding moved so the
# changes worth keeping can be applied by hand.
#
#   bash paper/update_cvpr_kit.sh            # show what the newest kit would change
#   bash paper/update_cvpr_kit.sh --apply    # replace the formatting files
set -euo pipefail

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Tags come back in no useful order, so pick the CVPR LaTeX tag with the highest
# embedded year rather than trusting position.
read -r TAG TAGURL < <(curl -sSL "https://api.github.com/repos/cvpr-org/author-kit/tags" \
      | python3 -c "
import json, re, sys, urllib.parse
tags = [t['name'] for t in json.load(sys.stdin)]
print('== author-kit tags ==', file=sys.stderr)
for t in tags[:8]:
    print('   ' + t, file=sys.stderr)
cand = []
for t in tags:
    u = t.upper()
    if not u.startswith('CVPR') or 'MSWORD' in u:
        continue
    m = re.search(r'(20\d\d)', t)
    if m:
        cand.append((int(m.group(1)), t))
if cand:
    year, tag = max(cand)
    print(tag, urllib.parse.quote(tag, safe=''))
")
[ -n "${TAG:-}" ] || { echo "could not determine a CVPR tag"; exit 1; }
echo "== newest CVPR latex kit: $TAG =="

CUR=$(grep -o 'ProvidesPackage{cvpr}\[[0-9]\{4\}' "$PAPER/cvpr.sty" | grep -o '[0-9]\{4\}$' || echo "?")
WANT=$(echo "$TAG" | grep -o '[0-9]\{4\}' | head -1)
echo "   vendored cvpr.sty declares year: $CUR"
echo "   upstream tag year:               $WANT"
if [ "$CUR" = "$WANT" ]; then
  echo "== already on the newest kit; nothing to fetch =="
  exit 0
fi

curl -sSL -o "$WORK/kit.tar.gz" "https://api.github.com/repos/cvpr-org/author-kit/tarball/$TAGURL"
tar xzf "$WORK/kit.tar.gz" -C "$WORK"
SRC=$(find "$WORK" -maxdepth 1 -type d -name "cvpr-org-author-kit-*" | head -1)
[ -d "$SRC" ] || { echo "unexpected tarball layout"; exit 1; }

echo
echo "== formatting files (replaced by --apply) =="
for f in cvpr.sty ieeenat_fullname.bst; do
  if diff -q "$SRC/$f" "$PAPER/$f" >/dev/null 2>&1; then
    echo "   $f: unchanged"
  else
    echo "   $f: $(diff "$SRC/$f" "$PAPER/$f" 2>/dev/null | grep -c '^[<>]') differing lines"
  fi
done

echo
echo "== upstream scaffolding changes (NOT applied; review by hand) =="
for f in main.tex preamble.tex rebuttal.tex; do
  [ -f "$SRC/$f" ] || continue
  echo "--- $f ---"
  diff "$SRC/$f" "$PAPER/$f" 2>/dev/null | head -25 || true
done

if [ "$APPLY" = "1" ]; then
  for f in cvpr.sty ieeenat_fullname.bst; do
    cp "$SRC/$f" "$PAPER/$f"
    echo "applied: $f"
  done
  echo
  echo "Now set \\def\\confYear{...} in main.tex if the target year changed,"
  echo "then rebuild and confirm the page header reads the right conference."
else
  echo
  echo "(dry run. rerun with --apply to replace the two formatting files)"
fi
