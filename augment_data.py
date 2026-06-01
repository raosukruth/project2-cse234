"""Dataset augmenter for schema-linking SFT.

This script supports:
1) prompt-format expansion (basic/pkfk/sorted)
2) lightweight question paraphrases
3) identifier naturalness variants in questions
4) optional synthetic rows from an external json/jsonl file

It outputs training records in the TRL/SFT "text" format.
"""

import argparse
import json
import re
from functools import lru_cache


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=None)
def load_schema(db_id):
    clean_id = db_id.replace(" ", "_").replace("/", "_")
    with open(f"./schemas/{clean_id}.json") as f:
        return json.load(f)


def build_plain_schema(schema_obj):
    schema = {t: [] for t in schema_obj["table_names_original"]}
    for i, name in schema_obj["column_names_original"]:
        if i == -1:
            continue
        schema[schema_obj["table_names_original"][i]].append(name)
    return schema


def build_pkfk_schema(schema_obj):
    col_info = schema_obj["column_names_original"]
    pks = schema_obj.get("primary_keys", [])
    fks = schema_obj.get("foreign_keys", [])
    ann = {}

    for pk in pks:
        for idx in (pk if isinstance(pk, list) else [pk]):
            ann[idx] = "(PK)"

    for fi, ti in fks:
        tname = schema_obj["table_names_original"][col_info[ti][0]]
        ann[fi] = f"(PK,FK->{tname})" if ann.get(fi) == "(PK)" else f"(FK->{tname})"

    schema = {}
    for idx, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        tname = schema_obj["table_names_original"][tidx]
        schema.setdefault(tname, []).append(f"{cname} {ann[idx]}" if idx in ann else cname)
    return schema


def build_sorted_schema(schema_obj):
    plain = build_plain_schema(schema_obj)
    return {t: sorted(cols) for t, cols in sorted(plain.items())}


def format_prompt(schema_repr, question, mode):
    if mode == "pkfk":
        system_prompt = (
            "You are a schema-linking assistant. Given a question and a database schema with "
            "PK/FK annotations, return ONLY a valid JSON object that maps table names to "
            "relevant column-name lists (without annotations in the output)."
        )
        user_prompt = (
            f"Database schema (PK/FK annotated): {schema_repr}\n\n"
            f"Question: {question}\n\n"
            "Return a JSON object with only the relevant tables as keys and lists of relevant "
            "column names as values. You MUST include specific column names. "
            "Column names in output should NOT include annotations. "
            "Example: {\"Orders\": [\"order_id\", \"total\"], \"Customers\": [\"name\"]}"
        )
    elif mode == "sorted":
        system_prompt = (
            "You are a schema-linking assistant. Given a question and a database schema, "
            "return ONLY a valid JSON object that maps table names to relevant column-name lists."
        )
        user_prompt = (
            f"Database schema (sorted): {schema_repr}\n\n"
            f"Question: {question}\n\n"
            "Return a JSON object with only the relevant tables as keys and lists of relevant "
            "column names as values. "
            "Example: {\"Orders\": [\"order_id\", \"total\"], \"Customers\": [\"name\"]}"
        )
    else:
        system_prompt = (
            "You are a schema-linking assistant. Given a question and a database schema, "
            "return ONLY a valid JSON object that maps table names to relevant column-name lists."
        )
        user_prompt = (
            f"Database schema: {schema_repr}\n\n"
            f"Question: {question}\n\n"
            "Return a JSON object with only the relevant tables as keys and lists of relevant "
            "column names as values. You MUST include specific column names. "
            "Example: {\"Orders\": [\"order_id\", \"total\"], \"Customers\": [\"name\"]}"
        )

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_text(row, question, mode):
    schema_obj = load_schema(row["db_id"])
    if mode == "pkfk":
        schema_repr = build_pkfk_schema(schema_obj)
    elif mode == "sorted":
        schema_repr = build_sorted_schema(schema_obj)
    else:
        schema_repr = build_plain_schema(schema_obj)

    answer = json.dumps(row["schema_links"], ensure_ascii=False)
    return format_prompt(schema_repr, question, mode) + answer + "<|im_end|>"


def maybe_replace_prefix(text, src, dst):
    if text.lower().startswith(src.lower()):
        return dst + text[len(src):]
    return None


def paraphrase_question(question, max_paraphrases):
    variants = []
    q = " ".join(question.split())
    variants.append(q)

    candidates = []
    for src, dst in [
        ("What is", "Show"),
        ("What are", "List"),
        ("List", "Show"),
        ("Find", "Identify"),
        ("How many", "Count"),
    ]:
        out = maybe_replace_prefix(q, src, dst)
        if out:
            candidates.append(out)

    if q.endswith("?"):
        candidates.append(q[:-1])
    else:
        candidates.append(q + "?")

    for c in candidates:
        c = " ".join(c.split())
        if c and c not in variants:
            variants.append(c)
        if len(variants) >= max_paraphrases + 1:
            break

    return variants


def naturalize_identifier(name):
    spaced = name.replace("_", " ")
    spaced = re.sub(r"\s+", " ", spaced).strip()
    if not spaced:
        return name
    return spaced.lower()


def naturalness_question_variants(question, row, max_naturalness):
    if max_naturalness <= 0:
        return [question]

    schema_obj = load_schema(row["db_id"])
    table_names = schema_obj.get("table_names_original", [])
    column_names = [c for tidx, c in schema_obj.get("column_names_original", []) if tidx != -1]
    tokens = sorted(set(table_names + column_names), key=len, reverse=True)

    variants = [question]
    for token in tokens:
        if len(variants) >= max_naturalness + 1:
            break
        if token in question and ("_" in token or any(ch.isupper() for ch in token[1:])):
            replacement = naturalize_identifier(token)
            candidate = question.replace(token, replacement)
            if candidate != question and candidate not in variants:
                variants.append(candidate)

    return variants


def load_optional_synthetic(path):
    if not path:
        return []
    if path.endswith(".jsonl"):
        rows = load_jsonl(path)
    else:
        rows = load_json(path)

    filtered = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            print(f"Skipping synthetic row {idx}: expected object")
            continue
        needed = ("db_id", "question", "schema_links")
        if not all(k in row for k in needed):
            print(f"Skipping synthetic row {idx}: missing one of {needed}")
            continue
        filtered.append(row)
    return filtered


def build_augmented_rows(base_rows, modes, paraphrases, naturalness, keep_original):
    augmented = []
    seen = set()

    for row in base_rows:
        question_variants = []
        for q in paraphrase_question(row["question"], paraphrases):
            for nq in naturalness_question_variants(q, row, naturalness):
                if nq not in question_variants:
                    question_variants.append(nq)

        if not keep_original and question_variants:
            question_variants = question_variants[1:]

        for q in question_variants:
            for mode in modes:
                key = (row["db_id"], q, json.dumps(row["schema_links"], sort_keys=True), mode)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    augmented.append({"text": render_text(row, q, mode)})
                except Exception as exc:
                    qid = row.get("question_id", "unknown")
                    print(f"{mode} render failed for question_id={qid}: {exc}")
    return augmented


def parse_args():
    ap = argparse.ArgumentParser(description="Augment schema-linking training data")
    ap.add_argument("--input", default="train.json", help="Base training file")
    ap.add_argument("--output", default="augmented_train.json", help="Output json file")
    ap.add_argument(
        "--modes",
        default="basic,pkfk,sorted",
        help="Comma-separated prompt modes: basic,pkfk,sorted",
    )
    ap.add_argument(
        "--paraphrases",
        type=int,
        default=0,
        help="How many paraphrase variants per question (0 disables paraphrasing)",
    )
    ap.add_argument(
        "--naturalness",
        type=int,
        default=0,
        help="How many identifier-naturalness variants per question (0 disables)",
    )
    ap.add_argument(
        "--synthetic_file",
        default=None,
        help="Optional json/jsonl with synthetic rows containing db_id,question,schema_links",
    )
    ap.add_argument(
        "--drop_original_question",
        action="store_true",
        help="If set, exclude the original question wording and keep only variants",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"basic", "pkfk", "sorted"}
    bad_modes = [m for m in modes if m not in valid_modes]
    if bad_modes:
        raise ValueError(f"Unsupported mode(s): {bad_modes}. Allowed: {sorted(valid_modes)}")

    train = load_json(args.input)
    synthetic = load_optional_synthetic(args.synthetic_file)
    all_rows = train + synthetic

    augmented = build_augmented_rows(
        all_rows,
        modes=modes,
        paraphrases=max(0, args.paraphrases),
        naturalness=max(0, args.naturalness),
        keep_original=not args.drop_original_question,
    )

    with open(args.output, "w") as f:
        json.dump(augmented, f)

    print(f"Base rows: {len(train)}")
    print(f"Synthetic rows accepted: {len(synthetic)}")
    print(f"Total source rows: {len(all_rows)}")
    print(f"Modes: {modes}")
    print(f"Paraphrases per question: {max(0, args.paraphrases)}")
    print(f"Naturalness variants per question: {max(0, args.naturalness)}")
    print(f"Augmented rows: {len(augmented)}")
    print(f"Saved to {args.output}")
