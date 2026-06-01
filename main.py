import argparse
import json
import os
import random
import torch
from json_repair import repair_json
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256
SEED = 42


def set_deterministic_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _list_to_schema_dict(lst):
    """Convert a JSON list of {TableName/Table, Columns} objects to a schema dict."""
    converted = {}
    for item in lst:
        if not isinstance(item, dict):
            continue
        table = None
        for k in ('TableName', 'Table', 'table', 'tablename'):
            v = item.get(k)
            if isinstance(v, str):
                table = v
                break
        cols = None
        for k in ('Columns', 'columns', 'cols'):
            v = item.get(k)
            if isinstance(v, list):
                cols = v
                break
        if table and cols is not None:
            converted[table] = [c for c in cols if isinstance(c, str)]
    return converted


def parse_model_json(response):
    try:
        result = json.loads(response.strip())
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            converted = _list_to_schema_dict(result)
            if converted:
                return converted
    except Exception:
        pass

    # (handles text preamble before JSON, or model outputting list-wrapped content)
    try:
        start = response.find('{')
        if start != -1:
            # Only treat as standalone object if not immediately inside a list
            prefix = response[:start].strip()
            if not prefix.endswith('['):
                depth = 0
                for i in range(start, len(response)):
                    ch = response[i]
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            result = json.loads(response[start:i + 1])
                            if isinstance(result, dict):
                                return result
                            break
    except Exception:
        pass

    # handles truncation, single quotes, etc
    try:
        result = json.loads(repair_json(response))
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            converted = _list_to_schema_dict(result)
            if converted:
                return converted
    except Exception:
        pass

    return {}
    

# def format_schema(data, schema):
#     result = {}

#     if type(data) == dict:
#         columns_key = None

#         for k in data:
#             if k.lower() == "columns":
#                 columns_key = k
#                 break

#         if columns_key and type(data[columns_key]) == list:

#             for k in data:
#                 if k.lower() == "tablename":
#                     for v in data[k]:
#                         result[v] = []
#                     break

#             for item in data[columns_key]:
#                 if type(item) == dict:
#                     table = None
#                     cols = []

#                     for k in item:
#                         if k.lower() == "table":
#                             table = item[k]
#                         elif k.lower() == "columns":
#                             cols = item[k]

#                     if table:
#                         result[table] = cols

#             return result

#         flat = True
#         for v in data.values():
#             if type(v) != list:
#                 flat = False

#         if flat:
#             return data

#     else:
#         return {}

#     return result

def get_table_and_column_data(data):
    table_data = {}

    table_names = data.get("TableName")
    if type(table_names) == list:
        for table_name in table_names:
            if type(table_name) == str:
                table_data[table_name] = []

    column_data = data.get("Columns")
    if type(column_data) != list:
        column_data = data.get("columns")

    if type(column_data) == list:
        for table_info in column_data:
            if type(table_info) != dict:
                continue

            table_name = table_info.get("Table")
            if type(table_name) != str:
                continue

            columns = table_info.get("Columns")
            if type(columns) != list:
                columns = table_info.get("columns")
            if type(columns) != list:
                continue

            valid_columns = []
            for name in columns:
                if type(name) == str:
                    valid_columns.append(name)

            # table_data[table_name] = valid_columns
            if table_name not in table_data:
                table_data[table_name] = []
            table_data[table_name].extend(valid_columns)

    return table_data


def get_simple_table_data(data):
    table_data = {}

    for table_name, columns in data.items():
        if type(table_name) != str:
            continue
        if type(columns) == list:
            valid_columns = [name for name in columns if type(name) == str]
        elif type(columns) == str:
            # Model sometimes outputs a string instead of a list — treat as single-element list
            valid_columns = [columns]
        else:
            # null or other non-list: include table with no columns so it gets table credit
            valid_columns = []
        table_data[table_name] = valid_columns

    return table_data


def find_matching_table(table_name, schema):
    for name in schema:
        if name.lower() == table_name.lower():
            return name
    return None


def get_valid_columns(columns, schema_columns):
    valid_columns = []

    for name in columns:
        for col in schema_columns:
            if col.lower() == name.lower():
                if col not in valid_columns:
                    valid_columns.append(col)
                break

    return valid_columns


def format_schema(data, schema):
    if type(data) != dict:
        return {}

    table_data = get_table_and_column_data(data)
    if len(table_data) == 0:
        table_data = get_simple_table_data(data)

    result = {}
    for table_name, columns in table_data.items():
        matching_table = find_matching_table(table_name, schema)

        if matching_table is None:
            continue

        result[matching_table] = get_valid_columns(
            columns,
            schema[matching_table]
        )

    return result

def load_model(model_name, adapter_dir='./adapter'):
    if PeftModel is None:
        raise ImportError(
            "peft is required for LoRA inference. Install peft and its dependencies before running."
        )
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(
            f"LoRA adapter directory not found: {adapter_dir}. Cannot run without LoRA."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print(f"Loaded adapter from {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_response(model, tokenizer, system_prompt, prompt, max_new_tokens):
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def load_schema_as_dict(db_id, schemas_dir='./schemas'):
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(f'{schemas_dir}/{fname}') as f:
        s = json.load(f)
    schema = {t: [] for t in s['table_names_original']}
    for tidx, cname in s['column_names_original']:
        if tidx == -1:
            continue
        schema[s['table_names_original'][tidx]].append(cname)
    return schema


def load_schema_pkfk(db_id, schemas_dir='./schemas'):
    """Load schema with PK/FK annotations — matches pkfk_formatting_function used in training."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(f'{schemas_dir}/{fname}') as f:
        s = json.load(f)
    col_info = s['column_names_original']
    ann = {}
    for pk in s.get('primary_keys', []):
        for idx in (pk if isinstance(pk, list) else [pk]):
            ann[idx] = '(PK)'
    for fi, ti in s.get('foreign_keys', []):
        tname = s['table_names_original'][col_info[ti][0]]
        ann[fi] = f'(PK,FK→{tname})' if ann.get(fi) == '(PK)' else f'(FK→{tname})'
    schema = {}
    for idx, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        tname = s['table_names_original'][tidx]
        schema.setdefault(tname, []).append(f'{cname} {ann[idx]}' if idx in ann else cname)
    return schema


def build_prompt_and_system(schema, question, format_mode):
    """Return (system_prompt, user_prompt) matching the training format for each config type."""
    if format_mode == 'pkfk':
        system = (
            "You are a schema-linking assistant. "
            "Given a question and a database schema with PK/FK annotations, return ONLY a valid JSON object "
            "that maps table names to relevant column-name lists (without annotations in the output)."
        )
        prompt = (
            f"Database schema (PK/FK annotated): {schema}\n\n"
            f"Question: {question}\n\n"
            'Return JSON only — column names without annotations: {"TableName": ["col1", "col2"], ...}'
        )
    elif format_mode == 'sorted':
        system = (
            "You are a schema-linking assistant. "
            "Given a question and a database schema, return ONLY a valid JSON object "
            "that maps table names to relevant column-name lists."
        )
        prompt = (
            f"Database schema (sorted): {schema}\n\n"
            f"Question: {question}\n\n"
            'Return JSON only in this format: {"TableName": ["col1", "col2"], ...}'
        )
    else:  # basic (default)
        system = (
            "You are a schema-linking assistant. "
            "Given a question and a database schema, return ONLY a valid JSON object "
            "that maps table names to relevant column-name lists."
        )
        prompt = (
            f"Database schema: {schema}\n\n"
            f"Question: {question}\n\n"
            "Return a JSON object with only the relevant tables as keys and lists of relevant column names as values. "
            "You MUST include specific column names — do not return empty lists unless a table has no relevant columns. "
            'Example: {"Orders": ["order_id", "total"], "Customers": ["name"]}'
        )
    return system, prompt


def predict_schema_links(question, db_id, schemas_dir, model, tokenizer,
                         format_mode='basic', verbose=False):
    # Always load the plain schema for output normalization
    schema = load_schema_as_dict(db_id, schemas_dir)

    # pkfk uses an annotated schema in the prompt
    if format_mode == 'pkfk':
        schema_for_prompt = load_schema_pkfk(db_id, schemas_dir)
    elif format_mode == 'sorted':
        schema_for_prompt = {t: sorted(cols) for t, cols in sorted(schema.items())}
    else:
        schema_for_prompt = schema

    system_prompt, prompt = build_prompt_and_system(schema_for_prompt, question, format_mode)
    response = generate_response(model, tokenizer, system_prompt, prompt, MAX_NEW_TOKENS)
    links = parse_model_json(response)
    result = format_schema(links, schema)
    if verbose and not result:
        print(f"  [EMPTY] raw response: {repr(response[:300])}")
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--schemas_dir', default='./schemas')
    ap.add_argument('--adapter_dir', default='./adapter')
    ap.add_argument('--format', default='basic', choices=['basic', 'pkfk', 'sorted'],
                    help='Prompt format — must match what the adapter was trained with')
    ap.add_argument('--verbose', action='store_true', help='Print raw responses for empty predictions')
    args = ap.parse_args()

    set_deterministic_seed(SEED)

    print("Loading model...")
    model, tokenizer = load_model(MODEL_NAME, adapter_dir=args.adapter_dir)
    print("Model loaded successfully")

    with open(args.input) as f:
        items = json.load(f)
    preds = []
    for i, it in enumerate(items):
        links = predict_schema_links(
            it['question'], it['db_id'], args.schemas_dir, model, tokenizer,
            format_mode=args.format, verbose=args.verbose
        )
        preds.append({'question_id': it['question_id'], 'schema_links': links})
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(items)} questions")
    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")
