import argparse
import json
import os
from json_repair import repair_json
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512


def parse_model_json(response):
    try:
        return json.loads(repair_json(response))
    except Exception:
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
        if type(columns) != list:
            continue

        valid_columns = []
        for name in columns:
            if type(name) == str:
                valid_columns.append(name)

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
    print("Loaded adapter from ./adapter")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
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


def predict_schema_links(question, db_id, schemas_dir, model, tokenizer):
    schema = load_schema_as_dict(db_id, schemas_dir)
    prompt = (
        "You are a schema-linking assistant.\n"
        "Return ONLY a valid JSON object mapping table names to relevant column-name lists.\n"
        "Output format example: {\"TableA\":[\"col1\",\"col2\"],\"TableB\":[]}\n"
        "Do not return wrapper keys like TableName or Columns.\n\n"
        f"Database schema:\n{schema}\n\n"
        f"Question: {question}"
    )

    response = generate_response(model, tokenizer, prompt, MAX_NEW_TOKENS)
    print("\n--- RAW MODEL RESPONSE START ---")
    print(response)
    print("--- RAW MODEL RESPONSE END ---")

    links = parse_model_json(response)
    print("PARSED TYPE:", type(links))
    print("PARSED VALUE:", links)

    return format_schema(links, schema )
    # return links


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--schemas_dir', default='./schemas')
    args = ap.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(MODEL_NAME)
    print("Model loaded successfully")

    with open(args.input) as f:
        items = json.load(f)
    preds = []
    for it in items:
        links = predict_schema_links(it['question'], it['db_id'], args.schemas_dir, model, tokenizer)
        preds.append({'question_id': it['question_id'], 'schema_links': links})
    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")
