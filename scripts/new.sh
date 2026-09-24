#!/usr/bin/env bash
# Create a new file from a template.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/new.sh update                                   weekly update, dated today
  scripts/new.sh note  <project> <slug> ["Title"]         numbered note, e.g. notes/nano-vllm/01-scheduler.md
  scripts/new.sh issue <project> <number> <slug> ["Title"] issue write-up, e.g. notes/nano-vllm/issues/123-oom.md
  scripts/new.sh exp   <slug> ["Question"]                experiment folder, e.g. experiments/001-cuda-graph/

Examples:
  scripts/new.sh note nano-vllm scheduler "How does the scheduler choose prefill or decode?"
  scripts/new.sh issue nano-vllm 123 oom-long-prompt "OOM with long prompts"
  scripts/new.sh exp cuda-graph "How much does CUDA Graph save per decode step?"
EOF
  exit 1
}

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
today="$(date +%Y-%m-%d)"
year="$(date +%Y)"

# Escape text for use in a sed replacement.
esc() { printf '%s' "$1" | sed -e 's/[\/&|]/\\&/g'; }

# Count paths that exist (an unmatched glob stays literal and is not counted).
count() { local n=0 f; for f in "$@"; do [ -e "$f" ] && n=$((n + 1)); done; echo "$n"; }

# render <template> <output> <title> <project> <number>
render() {
  local tpl="templates/$1" out="$2"
  if [ -e "$out" ]; then echo "Already exists: $out" >&2; exit 1; fi
  mkdir -p "$(dirname "$out")"
  sed -e "s|{{DATE}}|$(esc "$today")|g" \
      -e "s|{{TITLE}}|$(esc "$3")|g" \
      -e "s|{{PROJECT}}|$(esc "$4")|g" \
      -e "s|{{NUMBER}}|$(esc "$5")|g" \
      "$tpl" > "$out"
  echo "Created $out"
}

case "${1:-}" in
  update)
    render update.md "updates/$year/$today.md" "" "" ""
    echo "Next: add a line for it under 'Recent updates' in README.md"
    ;;
  note)
    [ $# -ge 3 ] || usage
    project="$2"; slug="$3"; title="${4:-$3}"
    if [ "$(count notes/"$project"/[0-9][0-9]-"$slug".md)" -gt 0 ]; then
      echo "A note named $slug already exists in notes/$project/" >&2; exit 1
    fi
    n=$(count notes/"$project"/[0-9][0-9]-*.md)
    num=$(printf "%02d" $((n + 1)))
    render note.md "notes/$project/$num-$slug.md" "$title" "$project" "$num"
    echo "Next: add it to the Notes table in notes/$project/README.md"
    ;;
  issue)
    [ $# -ge 4 ] || usage
    project="$2"; number="$3"; slug="$4"; title="${5:-$4}"
    render issue.md "notes/$project/issues/$number-$slug.md" "$title" "$project" "$number"
    echo "Next: add a row to CONTRIBUTIONS.md"
    ;;
  exp)
    [ $# -ge 2 ] || usage
    slug="$2"; title="${3:-$2}"
    if [ "$(count experiments/[0-9][0-9][0-9]-"$slug")" -gt 0 ]; then
      echo "An experiment named $slug already exists in experiments/" >&2; exit 1
    fi
    n=$(count experiments/[0-9][0-9][0-9]-*)
    num=$(printf "%03d" $((n + 1)))
    dir="experiments/$num-$slug"
    render experiment.md "$dir/README.md" "$title" "" "$num"
    cp templates/run.sh "$dir/run.sh"
    chmod +x "$dir/run.sh"
    mkdir -p "$dir/results"
    echo "Next: add a row to experiments/README.md"
    ;;
  *)
    usage
    ;;
esac
