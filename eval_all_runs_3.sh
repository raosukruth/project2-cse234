#!/usr/bin/env bash
# eval_all_runs_3.sh  —  evaluate all runs from rf_experiments_3.ipynb
#
# Usage:
#   ./eval_all_runs_3.sh <experiment_dir>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_dir>"
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

LABEL[1]="Q1_r4_attn_lr2e5_e2";  FORMAT_MODE[1]="basic"
LABEL[2]="Q2_r4_attn_lr2e5_e3";  FORMAT_MODE[2]="basic"
LABEL[3]="Q3_r8_attn_lr2e5_e3";  FORMAT_MODE[3]="basic"
LABEL[4]="Q4_r4_attn_lr1e5_e4";  FORMAT_MODE[4]="basic"
LABEL[5]="Q5_r4_attn_cosine_e3"; FORMAT_MODE[5]="basic"

declare -A SCORES

for run in 1 2 3 4 5; do
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

    lb=$(grep "Leaderboard Score" "$summary_txt" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    ts=$(grep "Table Score" "$summary_txt" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    cs=$(grep "Column Score" "$summary_txt" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    SCORES[$run]="$lb (table=$ts col=$cs)"
    echo ""
done

echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"
for run in 1 2 3 4 5; do
    if [ -n "${SCORES[$run]+_}" ]; then
        echo "  run $run  ${LABEL[$run]}  ->  ${SCORES[$run]}"
    fi
done

echo ""
echo "  best so far: P2_r4_attn_basic_e3  ->  0.4114 (table=0.5043 col=0.3185)"
