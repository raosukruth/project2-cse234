import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from json_repair import repair_json


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512


def parse_model_json(response):
    try:
        return json.loads(repair_json(response))
    except Exception:
        return {}

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
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
        if tidx == -1:       # skip the synthetic '*' entry
            continue
        schema[s['table_names_original'][tidx]].append(cname)
    return schema  


def predict_schema_links(question, db_id, schemas_dir, model, tokenizer):
    schema = load_schema_as_dict(db_id, schemas_dir)
    prompt = (
        f"You are given: \n"
        f"(1) The database schema:\n{schema}\n\n"
        f"(2) Question: {question}\n\n"
        f"Identify the tables and columns which are relevant. Please return a JSON object in this format: {{\"TableName\": [\"col1\", \"col2\"]}}"
    )

    response = generate_response(model, tokenizer, prompt, MAX_NEW_TOKENS)
    print("\n--- RAW MODEL RESPONSE START ---")
    print(response)
    print("--- RAW MODEL RESPONSE END ---")

    links = parse_model_json(response)
    print("PARSED TYPE:", type(links))
    print("PARSED VALUE:", links)
    
    return links
    

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
