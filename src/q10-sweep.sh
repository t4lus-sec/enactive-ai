#!/usr/bin/env bash
# Q10 sweep: several training recipes over one night's weaknesses, compared on the same weaknesses, nothing served.
#
# Why a sweep needs its own script. The cost is not the GPU - an arm is ~17 min (Q9c), and the layer-40 streams and
# the rendering are computed once and shared - it is the number of weaknesses with a usable held-out instance. Ten
# arms judged on eight weaknesses is noise mining, so this script:
#   * compares arms **paired** per weakness (the spread between weaknesses dwarfs the spread between arms);
#   * applies Holm across the arms, because ten arms at 95% gives about a 40% chance of a false winner;
#   * withholds a share of weaknesses (CONFIRM) from the selection entirely, which `weakness-compare` refuses to
#     print without --final, so the winner can be judged once on data the sweep never saw;
#   * renders once per *render-affecting* knob (PREFIX_LOSS, LESSONS_TEXT) and shares everything else, and those two
#     knobs deliberately leave every scored set untouched, so arms across renderings compare document for document;
#   * never switches the served model. The winner is a recipe; a night (q10.sh) is what serves it.
#
# The arms file: one arm per line, "LABEL key=value ...", # for a comment, the first arm is the baseline.
#   base
#   ratio8      CURATED_RATIO=8.0
#   weight3     CURATED_WEIGHT=3.0
#   declarative LESSONS_TEXT=expected
# Knobs: CURATED_RATIO CURATED_WEIGHT REPLAY_RATIO EPOCHS LR LAYERS PREFIX_LOSS LESSONS_TEXT WITH_CONTEXT
# A default set of arms is written for you if ARMS names nothing.
#
#   nohup bash ~/Downloads/q10-sweep.sh >/dev/null 2>&1 & tail -F ~/s0ma-engine/q10-sweep.log
#
# Each arm's own gate report is $SWEEP/gate-<arm>.txt, the comparison is ref/q10-sweep.txt (+ .json), and every
# arm's run is in this script's log (it passes LOG through to q10.sh).
#
# ARMS=FILE, CONFIRM (0.3), GOLDEN (0.1), ALPHA (0.05), FINAL=1 (open the withheld share for the winner),
# DRY=1 (print the plan and stop), KEEP=1 (keep every arm's weights), plus everything q10.sh takes.

set -euo pipefail
NAME=q10-sweep
ENGINE="${ENGINE:-$HOME/s0ma-engine}"
QWORK="${QWORK:-$HOME/qref-work}"
VENV="${VENV:-$HOME/qref-venv}"
REF="${REF:-$ENGINE/ref}"
Q10="${Q10_DIR:-$QWORK/q10}"
PY="${PY:-$VENV/bin/python}"
SWEEP="${SWEEP:-$Q10/sweep-$(date +%Y%m%d)}"
Q10SH="${Q10SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/q10.sh}"
LOG="${LOG:-$ENGINE/$NAME.log}"
CONFIRM="${CONFIRM:-0.3}"
GOLDEN="${GOLDEN:-0.1}"
ALPHA="${ALPHA:-0.05}"
export CONFIRM GOLDEN
t_start=$SECONDS
mkdir -p "$SWEEP" "$REF"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== [$(date +%H:%M:%S), $(( (SECONDS - t_start) / 60 ))m] $*"; }

[[ -x "$PY" ]] || { echo "no python at $PY (PY=PATH)"; exit 1; }
[[ -f "$Q10SH" ]] || { echo "no q10.sh at $Q10SH (Q10SH=FILE): the sweep runs its parts 1-4 per arm"; exit 1; }
grep -q 'ST="\${ST:-' "$Q10SH" || { echo "$Q10SH is older than this sweep script (it needs RENDER/ST/BASE_DIR/GATE overridable)"; exit 1; }
(cd "$QWORK/qref" && "$PY" -m qref weakness-compare --help >/dev/null 2>&1) \
  || { echo "this qref has no weakness-compare: it needs 2026-10-21f or newer ($QWORK/qref/VERSION)"; exit 1; }

# ---- the arms
ARMS="${ARMS:-$SWEEP/arms.txt}"
if [[ ! -f "$ARMS" ]]; then
  cat > "$ARMS" <<'ARMEOF'
# Q10 sweep, one arm per line: LABEL key=value ...   (the first is the baseline)
# Written by q10-sweep.sh as a starting point. One knob at a time from the baseline, because a full grid at these
# weakness counts buys nothing: the paired comparison can separate arms, but it cannot separate interactions.
base
# how much of the corpus the lessons are, by repetition
ratio05        CURATED_RATIO=0.5
ratio8         CURATED_RATIO=8.0
# the same budget spent as loss weight instead of repetition
weight3        CURATED_WEIGHT=3.0
# Q9c found 2 epochs beat 1 on s0ma's own tokens; a tiny curated set may prefer fewer
ep1            EPOCHS=1
# a hotter rate, which interacts with how often a small set is cycled
lr5            LR=5e-5
# does counting s0ma's turns in the lesson's prefix help the correction stick?
prefixown      PREFIX_LOSS=own
# the declarative arm: the lesson is `expected` as prose, the measurement unchanged
declarative    LESSONS_TEXT=expected
# half the slice: cheaper, and behaviour may not need eight layers
layers4        LAYERS=4
ARMEOF
  echo "wrote a starting set of arms to $ARMS - edit it and run again to use your own"
fi
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$ARMS")
(( ${#LINES[@]} >= 2 )) || { echo "$ARMS needs at least two arms (the first is the baseline)"; exit 1; }

LABELS=(); KNOBS=(); VARIANT=()
for ln in "${LINES[@]}"; do
  read -r label rest <<<"$ln"
  pl=none; lt=draft; wc=1
  for kv in $rest; do
    case "$kv" in
      PREFIX_LOSS=*) pl="${kv#*=}" ;;
      LESSONS_TEXT=*) lt="${kv#*=}" ;;
      WITH_CONTEXT=*) wc="${kv#*=}" ;;
      CURATED_RATIO=*|CURATED_WEIGHT=*|REPLAY_RATIO=*|EPOCHS=*|LR=*|LAYERS=*) ;;
      *) echo "arm $label: unknown knob $kv"; exit 1 ;;
    esac
  done
  LABELS+=("$label"); KNOBS+=("$rest"); VARIANT+=("pl$pl-lt$lt-ctx$wc")
done
BASELABEL="${LABELS[0]}"
mapfile -t VARIANTS < <(printf '%s\n' "${VARIANT[@]}" | sort -u)
CANON="${VARIANT[0]}"

echo "sweep: ${#LABELS[@]} arms, baseline $BASELABEL, ${#VARIANTS[@]} rendering(s), confirm share $CONFIRM"
for i in "${!LABELS[@]}"; do printf '  %-14s %-44s [%s]\n' "${LABELS[$i]}" "${KNOBS[$i]:-(baseline)}" "${VARIANT[$i]}"; done
echo "  arms under $SWEEP; the served model is not touched"
[[ "${DRY:-0}" == 1 ]] && { echo "(DRY=1: stopping here)"; exit 0; }

rd() { echo "$SWEEP/rendered-$1"; }           # one rendering per render-affecting variant
stp() { echo "$SWEEP/st-$1"; }                # ... and its own streams prefix

# ---- the canonical rendering first, so the scored sets are fixed before anything trains
ORDERED=("$CANON")
for v in "${VARIANTS[@]}"; do [[ "$v" == "$CANON" ]] || ORDERED+=("$v"); done
VARIANTS=("${ORDERED[@]}")
for v in "${VARIANTS[@]}"; do
  pl="${v%%-lt*}"; pl="${pl#pl}"
  lt="${v#*-lt}"; lt="${lt%%-ctx*}"
  wc="${v##*-ctx}"
  if [[ -f "$(rd "$v")/render.json" ]]; then echo "rendering $v exists"; continue; fi
  step "rendering $v (prefix-loss $pl, lessons $lt, context $wc)"
  RENDER="$(rd "$v")" ST="$(stp "$v")" ST_EVAL="$(stp "$CANON")" PREFIX_LOSS="$pl" LESSONS_TEXT="$lt" \
    WITH_CONTEXT="$wc" PARTS=1 SWITCH=0 LOG="$LOG" GATE="$SWEEP/gate-ignored.txt" bash "$Q10SH" \
    || { echo "rendering $v failed"; exit 1; }
done

# every rendering must present the SAME scored sets, or the arms are not comparable: the knobs above are supposed
# to change the lessons only, and this is where that claim is checked rather than trusted
step "checking every rendering scores the same documents"
"$PY" - "$SWEEP" "${VARIANTS[@]}" <<'EOF' || exit 1
import hashlib, json, os, sys
sweep, variants = sys.argv[1], [v for v in sys.argv[2:] if v]
canon, bad = None, []
for v in variants:
    d = os.path.join(sweep, f"rendered-{v}")
    digests = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".pt") and f != "lessons.pt":
            digests[f] = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()[:16]
    r = json.load(open(os.path.join(d, "render.json")))
    print(f"  {v:22s} {len(digests)} scored sets, lessons "
          f"{r['sets'].get('lessons', {}).get('documents', 0)}d/"
          f"{r['sets'].get('lessons', {}).get('continuation_tokens', 0)}t")
    if canon is None:
        canon = (v, digests)
    elif digests != canon[1]:
        diff = sorted(set(digests) ^ set(canon[1])) or [k for k in digests if digests[k] != canon[1].get(k)]
        bad.append(f"{v} differs from {canon[0]} on {', '.join(diff)}")
if bad:
    sys.exit("the renderings do not score the same documents, so the arms cannot be compared:\n  "
             + "\n  ".join(bad))
print("  identical scored sets: the arms are comparable document for document")
EOF

# ---- the arms, each from the base weights
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"; v="${VARIANT[$i]}"
  armd="$SWEEP/arm-$label"
  if [[ -f "$armd/report.json" ]]; then echo "arm $label is already trained"; continue; fi
  step "arm $((i + 1))/${#LABELS[@]}: $label  ${KNOBS[$i]:-(baseline)}"
  # the scored sets' streams are shared (ST_EVAL); only this variant's lessons are streamed again
  # shellcheck disable=SC2086
  env ${KNOBS[$i]} RENDER="$(rd "$v")" ST="$(stp "$v")" ST_EVAL="$(stp "$CANON")" BASE_DIR="$SWEEP/base" \
    GATE="$SWEEP/gate-$label.txt" ARMD="$armd" ARM="sweep-$label" PARTS=2,3,4 SWITCH=0 \
    KEEP_STREAMS=1 LOG="$LOG" bash "$Q10SH" \
    || { echo "arm $label failed (above); the sweep carries on with the arms that trained"; }
done

# ---- the comparison
step "comparing the arms on the same weaknesses"
A=()
for label in "${LABELS[@]}"; do
  d="$SWEEP/arm-$label"
  [[ -f "$d/report.json" ]] && A+=(--arm "$label=$d") || echo "  $label: no report.json, left out"
done
(( ${#A[@]} >= 4 )) || { echo "fewer than two arms trained; nothing to compare"; exit 1; }   # 2 entries per arm
F=()
[[ "${FINAL:-0}" == 1 ]] && F=(--final)
W=()
[[ "${ALLOW_WRONG_ONLY:-0}" == 1 ]] && W=(--allow-wrong-only)
(cd "$QWORK/qref" && "$PY" -m qref weakness-compare "$(rd "$CANON")" "${A[@]}" --baseline "$BASELABEL" \
    --alpha "$ALPHA" "${F[@]}" "${W[@]}" --out "$REF/q10-sweep.txt") || { echo "weakness-compare failed"; exit 1; }

winner="$(sed -n 's/^Q10 SWEEP WINNER: \([^ ]*\).*/\1/p' "$REF/q10-sweep.txt" | tail -1)"
echo
echo "the report is at $REF/q10-sweep.txt (and q10-sweep.json), each arm's gate at $SWEEP/gate-<arm>.txt"
if [[ -n "$winner" && "$winner" != "$BASELABEL" ]]; then
  k=""
  for i in "${!LABELS[@]}"; do [[ "${LABELS[$i]}" == "$winner" ]] && k="${KNOBS[$i]}"; done
  echo "to run a night with the winning recipe:"
  echo "  ${k:+$k }bash $Q10SH"
  [[ "${FINAL:-0}" == 1 ]] || echo "to judge it on the withheld share, once: FINAL=1 ARMS=$ARMS bash $0"
else
  echo "the baseline stands: no recipe beat it once the correction across arms was applied."
  echo "At these counts that is the common outcome, and the answer is more weaknesses with drafted corrections,"
  echo "not more arms."
fi
if [[ "${KEEP:-0}" != 1 ]]; then
  n=0
  for label in "${LABELS[@]}"; do
    [[ "$label" == "$winner" ]] && continue          # the winner's weights stay, in case you want to serve them
    [[ -f "$SWEEP/arm-$label/slice.safetensors" ]] && { rm -f "$SWEEP/arm-$label/slice.safetensors"; n=$((n + 1)); }
  done
  echo "($n arms' weights deleted, their reports and scores kept; ${winner:-no} winner's weights are at"
  echo " $SWEEP/arm-$winner. KEEP=1 keeps them all.)"
fi
rm -rf "$SWEEP"/st-*-lessons-* 2>/dev/null || true
echo "$NAME.sh took $(( (SECONDS - t_start) / 60 )) min."
