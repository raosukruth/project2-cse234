#!/usr/bin/env bash
# eval_all_runs.sh  —  evaluate all runs from a single rapidfireai experiment.
#
# Usage:
#   ./eval_all_runs.sh <experiment_dir>
#
# Example:
#   ./eval_all_runs.sh ~/rapidfireai/rapidfire_experiments/experkrj_39
#
# run→format mapping (must match configs_spec order in rf_experiments.ipynb):
#   run  1 = A1_basic_r8        basic    120 steps
#   run  2 = A2_pkfk_r8         pkfk     120 steps
#   run  3 = A3_sorted_r8       sorted   120 steps
#   run  4 = B1_basic_r16       basic    120 steps
#   run  5 = B2_basic_r32       basic    120 steps
#   run  6 = C1_basic_lr2e4     basic    120 steps
#   run  7 = C2_basic_lr5e5     basic    120 steps
#   run  8 = D1_basic_attn      basic    120 steps
#   run  9 = E1_pkfk_r16_150    pkfk     150 steps
#   run 10 = F1_pkfk_r8_150     pkfk     150 steps
#   run 11 = F2_basic_r16_150   basic    150 steps
#   run 12 = G1_attn_r4_lr2e5   basic    120 steps  (original exp37 config)
#   run 13 = H1_basic_r8_200    basic    200 steps  (overfitting probe)
#   run 14 = H2_pkfk_r16_200    pkfk     200 steps  (overfitting probe)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_dir>"
    echo "Example: $0 ~/rapidfireai/rapidfire_experiments/experkrj_39"
    exit 1
fi

EXPERIMENT_DIR="$1"
INPUT_JSON="validation_input.json"
GOLD_JSON="validation_gold_schema_links.json"
SCHEMAS_DIR="schemas"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

declare -A LABEL
declare -A FORMAT_MODE

LABEL[1]="A1_basic_r8";       FORMAT_MODE[1]="basic"
LABEL[2]="A2_pkfk_r8";        FORMAT_MODE[2]="pkfk"
LABEL[3]="A3_sorted_r8";      FORMAT_MODE[3]="sorted"
LABEL[4]="B1_basic_r16";      FORMAT_MODE[4]="basic"
LABEL[5]="B2_basic_r32";      FORMAT_MODE[5]="basic"
LABEL[6]="C1_basic_lr2e4";    FORMAT_MODE[6]="basic"
LABEL[7]="C2_basic_lr5e5";    FORMAT_MODE[7]="basic"
LABEL[8]="D1_basic_attn";     FORMAT_MODE[8]="basic"
LABEL[9]="E1_pkfk_r16_150";   FORMAT_MODE[9]="pkfk"
LABEL[10]="F1_pkfk_r8_150";   FORMAT_MODE[10]="pkfk"
LABEL[11]="F2_basic_r16_150"; FORMAT_MODE[11]="basic"
LABEL[12]="G1_attn_r4_lr2e5"; FORMAT_MODE[12]="basic"
LABEL[13]="H1_basic_r8_200";  FORMAT_MODE[13]="basic"
LABEL[14]="H2_pkfk_r16_200";  FORMAT_MODE[14]="pkfk"

declare -A SCORES

for run in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    adapter_dir="$EXPERIMENT_DIR/runs/$run/checkpoints/final_checkpoint"
    label="${LABEL[$run]}"
    fmt="${FORMAT_MODE[$run]}"

    if [ ! -f "$adapter_dir/adapter_config.json" ]; then
        echo "Skipping run $run ($label) — no checkpoint found"
        continue
    fi

    pred_json="$RESULTS_DIR/${label}_predictions.json"
    per_q_csv="$RESULTS_DIR/${label}_per_question.csv"
    summary_txt="$RESULTS_DIR/${label}_eval.txt"

    echo "========================================"
    echo "Run $run: $label  (format=$fmt)"
    echo "========================================"

    python main.py \
        --input "$INPUT_JSON" \
        --output "$pred_json" \
        --schemas_dir "$SCHEMAS_DIR" \
        --adapter_dir "$adapter_dir" \
        --format "$fmt"

    python eval.py \
        --predictions "$pred_json" \
        --gold "$GOLD_JSON" \
        --schemas_dir "$SCHEMAS_DIR" \
        --questions_input "$INPUT_JSON" \
        --per_question_out "$per_q_csv" \
        | tee "$summary_txt"

    lb=$(grep "Leaderboard Score" "$summary_txt" | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    ts=$(grep "Table Score" "$summary_txt" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    cs=$(grep "Column Score" "$summary_txt" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    SCORES[$run]="$lb (table=$ts col=$cs)"
    echo ""
done

echo "========================================"
echo "FINAL SUMMARY (run / label / score)"
echo "========================================"
for run in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    if [ -n "${SCORES[$run]+_}" ]; then
        echo "  run $run  ${LABEL[$run]}  →  ${SCORES[$run]}"
    fi
done

echo ""
echo "  baseline exp37 (G1_attn_r4_lr2e5, 74 steps)  →  0.3983 (table=0.5015 col=0.2950)"
