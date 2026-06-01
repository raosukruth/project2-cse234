"""Standalone SFT training script bypassing rapidfireai."""
import json, sys, site, os, argparse
sys.path.insert(0, site.getusersitepackages())
os.chdir("/home/sjrao/project2-cse234")

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ── formatting functions ───────────────────────────────────────────────────────

def load_schema(db_id):
    clean_id = db_id.replace(" ", "_").replace("/", "_")
    with open(f"./schemas/{clean_id}.json") as f:
        s = json.load(f)
    schema = {t: [] for t in s["table_names_original"]}
    for i, name in s["column_names_original"]:
        if i == -1: continue
        schema[s["table_names_original"][i]].append(name)
    return s, schema

def basic_fmt(row):
    _, schema = load_schema(row["db_id"])
    sys_p = "You are a schema-linking assistant. Given a question and a database schema, return ONLY a valid JSON object that maps table names to relevant column-name lists."
    prompt = f"Database schema: {schema}\n\nQuestion: {row['question']}\n\nReturn JSON only: {{\"TableName\": [\"col1\", \"col2\"], ...}}"
    answer = json.dumps(row["schema_links"], ensure_ascii=False)
    return {"text": f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"}

def pkfk_fmt(row):
    s, _ = load_schema(row["db_id"])
    col_info = s["column_names_original"]
    pks = s.get("primary_keys", [])
    fks = s.get("foreign_keys", [])
    ann = {}
    for pk in pks:
        for idx in (pk if isinstance(pk, list) else [pk]):
            ann[idx] = "(PK)"
    for fi, ti in fks:
        tname = s["table_names_original"][col_info[ti][0]]
        ann[fi] = f"(PK,FK→{tname})" if ann.get(fi) == "(PK)" else f"(FK→{tname})"
    schema = {}
    for idx, (tidx, cname) in enumerate(col_info):
        if tidx == -1: continue
        tname = s["table_names_original"][tidx]
        schema.setdefault(tname, []).append(f"{cname} {ann[idx]}" if idx in ann else cname)
    sys_p = "You are a schema-linking assistant. Given a question and a database schema with PK/FK annotations, return ONLY a valid JSON object that maps table names to relevant column-name lists (without annotations in the output)."
    prompt = f"Database schema (PK/FK annotated): {schema}\n\nQuestion: {row['question']}\n\nReturn JSON only — column names without annotations: {{\"TableName\": [\"col1\", \"col2\"], ...}}"
    answer = json.dumps(row["schema_links"], ensure_ascii=False)
    return {"text": f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"}

def sorted_fmt(row):
    _, schema = load_schema(row["db_id"])
    sorted_schema = {t: sorted(cols) for t, cols in sorted(schema.items())}
    sys_p = "You are a schema-linking assistant. Given a question and a database schema, return ONLY a valid JSON object that maps table names to relevant column-name lists."
    prompt = f"Database schema (sorted): {sorted_schema}\n\nQuestion: {row['question']}\n\nReturn JSON only: {{\"TableName\": [\"col1\", \"col2\"], ...}}"
    answer = json.dumps(row["schema_links"], ensure_ascii=False)
    return {"text": f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"}

# ── configs ────────────────────────────────────────────────────────────────────
# (label, fmt_func, lora_r, lora_alpha, lr, max_steps, target_modules)
CONFIGS = [
    ("A1_basic_r8",     basic_fmt,  8,  16, 1e-4, 240, ALL_MODULES),
    ("A2_pkfk_r8",      pkfk_fmt,   8,  16, 1e-4, 240, ALL_MODULES),
    ("A3_sorted_r8",    sorted_fmt, 8,  16, 1e-4, 240, ALL_MODULES),
    ("B1_basic_r16",    basic_fmt,  16, 32, 1e-4, 240, ALL_MODULES),
    ("B2_basic_r32",    basic_fmt,  32, 64, 1e-4, 240, ALL_MODULES),
    ("C1_basic_lr2e4",  basic_fmt,  8,  16, 2e-4, 240, ALL_MODULES),
    ("C2_basic_lr5e5",  basic_fmt,  8,  16, 5e-5, 240, ALL_MODULES),
    ("D1_basic_attn",   basic_fmt,  16, 32, 1e-4, 240, ATTN_MODULES),
    ("E1_pkfk_r16_360", pkfk_fmt,   16, 32, 1e-4, 360, ALL_MODULES),
]

# ── main ───────────────────────────────────────────────────────────────────────

def train_config(label, fmt_func, lora_r, lora_alpha, lr, max_steps, target_modules,
                 train_dataset, val_dataset, save_best_to=None):
    print(f"\n{'='*60}")
    print(f"Training: {label}  r={lora_r} alpha={lora_alpha} lr={lr} steps={max_steps}")
    print(f"{'='*60}")

    ds_train = train_dataset.map(fmt_func)
    ds_val   = val_dataset.map(fmt_func)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1, bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    sft_cfg = SFTConfig(
        output_dir=f"./runs/{label}",
        learning_rate=lr,
        lr_scheduler_type="linear",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_steps=max_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=40,
        fp16=True,
        report_to="none",
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
    )

    result = trainer.train()
    print(f"[{label}] train_loss={result.training_loss:.4f}")

    adapter_dir = f"./adapters/{label}"
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved adapter → {adapter_dir}")

    if save_best_to is not None:
        import shutil
        if os.path.exists(save_best_to):
            shutil.rmtree(save_best_to)
        shutil.copytree(adapter_dir, save_best_to)
        print(f"Copied to {save_best_to}")

    del model
    torch.cuda.empty_cache()
    return result.training_loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Config labels to run (default: all). E.g. --configs A1_basic_r8 B1_basic_r16")
    ap.add_argument("--best_adapter_dir", default="./adapter",
                    help="Where to copy the best adapter (lowest train loss)")
    args = ap.parse_args()

    with open("train.json") as f:
        train_dataset = Dataset.from_list(json.load(f))
    with open("validation.json") as f:
        val_dataset = Dataset.from_list(json.load(f))

    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c[0] in args.configs]
        print(f"Running {len(configs_to_run)} configs: {[c[0] for c in configs_to_run]}")

    results = {}
    for cfg in configs_to_run:
        label = cfg[0]
        try:
            loss = train_config(*cfg, train_dataset=train_dataset, val_dataset=val_dataset)
            results[label] = loss
        except Exception as e:
            print(f"[{label}] FAILED: {e}")
            results[label] = float("inf")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for label, loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {label}: train_loss={loss:.4f}")

    best = min(results, key=results.get)
    print(f"\nBest config: {best} (loss={results[best]:.4f})")
    print(f"Copying best adapter → {args.best_adapter_dir}")
    import shutil
    best_adapter_src = f"./adapters/{best}"
    if os.path.exists(args.best_adapter_dir):
        shutil.rmtree(args.best_adapter_dir)
    shutil.copytree(best_adapter_src, args.best_adapter_dir)
    print("Done.")
