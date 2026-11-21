#!/bin/bash
# Follow the designA_solid_body runs: for each vessel, its most recent log, with the number of
# iterations and, from the latest full table, the smallest distances (X-point, Boozer surface and
# coils to the vessel; coils to the keep-out cylinder) and the mean non-QS, each against its target.
#   bash /mnt/ceph/users/agiuliani/STAR_LITE/star_lite_design/designA_solid_body/suivi.sh [column ...]
# The rows are sorted by the given columns (vessel, iter, xpt, surf, coil, cyl, nonqs; the first
# one decides, the next ones break ties), best first: distances and iter descending, non-QS
# ascending, vessel alphabetical; a missing value comes last. Default: nonqs.
# Next to each value, its trend over the last TREND iterations (default 10, e.g. TREND=50 bash
# suivi.sh): ↑ increasing, ↓ decreasing, → unchanged at the precision printed in the log.
cd "$(dirname "$0")/logs" || exit 1
TREND=${TREND:-10}
# targets (m), as given to boozer_solid_body.py: the margin (1st argument in run_many.sh) for the
# X-points and the Boozer surface, COIL_CLEARANCE_THRESHOLD for the coils, CYLINDER_CLEARANCE for
# the cylinder; non-QS has no target in the optimization: design A's own value is the reference
T_MARGIN=0.06; T_COIL=0.06; T_CYL=0.0; NONQS_A=2.263e-3
# name of each vessel id of boozer_solid_body.py (--vessel-id)
declare -A NAME=([0]=pill [1]=renaissance [2]=torus [3]=helical [4]=helical-R\(t\) [5]=ellipse-pipe
                 [6]=rect-pipe [5004]=welded-4 [5006]=welded-6 [5008]=welded-8)
# sort key of each column in the records built below (tab-separated: vessel, iter, then the value
# and the trend arrow of X-pt, surf, coil, cyl, non-QS); the column names are read without case or
# punctuation (X-pt = xpt)
declare -A KEY=([vessel]=-k1,1 [iter]=-k2,2gr [xpt]=-k3,3gr [surf]=-k5,5gr [coil]=-k7,7gr
                [cyl]=-k9,9gr [nonqs]=-k11,11g)
[ $# -eq 0 ] && set -- nonqs
cols=(); keys=()
for c in "$@"; do
  k=$(echo "$c" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')
  [ -z "${KEY[$k]}" ] && { echo "unknown column '$c' (vessel, iter, xpt, surf, coil, cyl, nonqs)"; exit 1; }
  cols+=("$k"); keys+=("${KEY[$k]}")
done
# colours only on a terminal: green = target met, red = not met
if [ -t 1 ]; then G=$'\e[32m' R=$'\e[31m' N=$'\e[0m'; else G= R= N=; fi
# one value cell, 12 characters wide: $1 = value (may be empty), $2 = its trend arrow, $3 = target,
# $4 = "ge" (value must be >= target, shown in cm, with a check mark and a colour telling whether
# the target is met) or "ref" (non-QS: a reference only, value shown as is)
cell() {
  [ -z "$1" ] && { printf " %12s │" "- "; return; }
  awk -v x="$1" -v a="$2" -v t="$3" -v m="$4" -v g="$G" -v r="$R" -v n="$N" 'BEGIN {
    if (m == "ref") { printf " %10.4e %s │", x, a; exit }
    ok = (x >= t)
    printf " %s%5.1f cm %s %s%s │", (ok ? g : r), 100 * x, a, (ok ? "✓" : "✗"), n }'
}
# trend arrow: $1 = latest value, $2 = value TREND iterations earlier
trend() { awk -v a="$1" -v b="$2" 'BEGIN { if (a == "" || b == "") print " "
          else if (a + 0 > b + 0) print "↑"; else if (a + 0 < b + 0) print "↓"; else print "→" }'; }
ge() { awk -v x="$1" 'BEGIN { printf ">= %.1f cm", 100 * x }'; }
rule() { printf "%s" "$1"; printf '─%.0s' $(seq 15)
         for w in 7 14 14 14 14 14; do printf "%s" "$2"; printf '─%.0s' $(seq $w); done; printf "%s\n" "$3"; }
row() { printf "│ %-13s │ %5s │ %12s │ %12s │ %12s │ %12s │ %12s │\n" "$@"; }

echo "designA_solid_body runs, $(date '+%F %T'), sorted by ${cols[*]} (best first)"
rule ┌ ┬ ┐
row vessel iter "X-pt-vessel" "surf-vessel" "coil-vessel" "coil-cyl" "non-QS"
row target "" "$(ge $T_MARGIN)" "$(ge $T_MARGIN)" "$(ge $T_COIL)" "$(ge $T_CYL)" "A: $(printf %.3e $NONQS_A)"
rule ├ ┼ ┤
jobs=(); recs=""
for v in 0 1 2 3 4 5 6 5004 5006 5008; do
  f=$(ls -t solid_body_vessel${v}_*.out 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  jobs+=("$(echo "$f" | sed -E 's/.*_([^_]+)\.out$/\1/')")
  # one quantity, from its log lines on stdin (one per iteration, the value is the first number of
  # the line): "<latest value><tab><trend arrow over the last TREND iterations>"; a missing value
  # is $1 (-inf for the distances, sorted descending, inf for non-QS, ascending: it comes last)
  q() { local h a b
        h=$(awk '{ if (match($0, /[-+]?[0-9]+\.[0-9]+e[-+][0-9]+/)) print substr($0, RSTART, RLENGTH) }')
        a=$(tail -1 <<< "$h"); b=$(tail -n $((TREND + 1)) <<< "$h" | head -1)
        printf "%s\t%s" "${a:-$1}" "$(trend "$a" "$b")"; }
  it=$(grep -a -c "│ iter" "$f")
  recs+=$(printf "%s\t" "${NAME[$v]:-$v}" "$it" \
            "$(grep -a 'minimum X-point-to-vessel distance' "$f" | q -inf)" \
            "$(grep -a 'minimum Boozer surface-to-vessel distance' "$f" | q -inf)" \
            "$(grep -a 'minimum coil-clearance distance' "$f" | q -inf)" \
            "$(grep -a 'coil to keep-out cylinder' "$f" | q -inf)")"$(grep -a -A1 '│ nonQS' "$f" | grep -av '│ nonQS' | q inf)"$'\n'
done
val() { case "$1" in inf|-inf) ;; *) echo "$1" ;; esac; }   # a missing value is shown as "-"
printf "%s" "$recs" | LC_ALL=C sort -s -t$'\t' "${keys[@]}" |
while IFS=$'\t' read -r name it xp xpa sf sfa cl cla cy cya nq nqa; do
  printf "│ %-13s │ %5s │%s%s%s%s%s\n" "$name" "$it" \
    "$(cell "$(val "$xp")" "$xpa" $T_MARGIN ge)" "$(cell "$(val "$sf")" "$sfa" $T_MARGIN ge)" \
    "$(cell "$(val "$cl")" "$cla" $T_COIL ge)" "$(cell "$(val "$cy")" "$cya" $T_CYL ge)" \
    "$(cell "$(val "$nq")" "$nqa" "" ref)"
done
rule └ ┴ ┘
echo "↑ ↓ →: change over the last $TREND iterations; ✓ target met, ✗ not met; non-QS: design A's value as reference"
for j in $(printf "%s\n" "${jobs[@]}" | sort -u); do
  st=$(squeue -h -j "$j" -o %T 2>/dev/null)
  echo "job $j (${st:-finished}): $PWD/solid_body_vessel<id>_$j.out"
done
