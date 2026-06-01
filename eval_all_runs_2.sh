#!/usr/bin/env bash
# eval_all_runs_2.sh  —  evaluate all runs from rf_experiments_2.ipynb
#
# Usage:
#   ./eval_all_runs_2.sh <experiment_dir>
#
# Example:
#   ./eval_all_runs_2.sh ~/rapidfireai/rapidfire_experiments/experkrj_41
#
# run→format mapping (must match configs_spec order in rf_experiments_2.ipynb):
#   run  1 = P1_r4_attn_basic_e2   basic   r=4  attn  lr=1e-5  epochs=2
#   run  2 = P2_r4_attn_basic_e3   basic   r=4  attn  lr=1e-5  epochs=3
#   run  3 = P3_r4_all_basic_e2    basic   r=4  all   lr=1e-5  epochs=2
#   run  4 = P4_r8_attn_basic_e2   basic   r=8  attn  lr=1e-5  epochs=2
#   run  5 = P5_r8_all_basic_e2    basic   r=8  all   lr=1e-5  epochs=2
#   run  6 = P6_r4_attn_lr5e6_e2   basic   r=4  attn  lr=5e-6  epochs=2
#   run  7 = P7_r4_attn_pkfk_e2    pkfk    r=4  attn  lr=1e-5  epochs=2
#   run  8 = P8_r8_all_pkfk_e3     pkfk    r=8  all   lr=1e-5  epochs=3

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_dir>"
    echo "Example: $0 ~/rapidfireai/rapidfire_experiments/experkrj_41"
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

LABEL[1]="P1_r4_attn_basic_e2";  FORMAT_MODE[1]="basic"
LABEL[2]="P2_r4_attn_basic_e3";  FORMAT_MODE[2]="basic"
LABEL[3]="P3_r4_all_basic_e2";   FORMAT_MODE[3]="basic"
LABEL[4]="P4_r8_attn_basic_e2";  FORMAT_MODE[4]="basic"
LABEL[5]="P5_r8_all_basic_e2";   FORMAT_MODE[5]="basic"
LABEL[6]="P6_r4_attn_lr5e6_e2";  FORMAT_MODE[6]="basic"
LABEL[7]="P7_r4_attn_pkfk_e2";   FORMAT_MODE[7]="pkfk"
LABEL[8]="P8_r8_all_pkfk_e3";    FORMAT_MODE[8]="pkfk"

declare -A SCORES

for run in 1 2 3 4 5 6 7 8; do
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
echo "FINAL SUMMARY"
echo "========================================"
for run in 1 2 3 4 5 6 7 8; do
    if [ -n "${SCORES[$run]+_}" ]; then
        echo "  run $run  ${LABEL[$run]}  ->  ${SCORES[$run]}"
    fi
done

echo ""
echo "  baseline exp37 (r=4, attn, lr=1e-5, 74 steps)  ->  0.3983 (table=0.5015 col=0.2950)"
