import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
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


def predict_schema_links(question, db_id, schemas_dir):
    return {}


if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument('--input',  required=True)
    # ap.add_argument('--output', required=True)
    # ap.add_argument('--schemas_dir', default='./schemas')
    # args = ap.parse_args()

    # with open(args.input) as f:
    #     items = json.load(f)
    # preds = []
    # for it in items:
    #     links = predict_schema_links(it['question'], it['db_id'], args.schemas_dir)
    #     preds.append({'question_id': it['question_id'], 'schema_links': links})
    # with open(args.output, 'w') as f:
    #     json.dump(preds, f, indent=2)
    # print(f"Wrote {len(preds)} predictions to {args.output}")

    print("Loading model...")
    model, tokenizer = load_model(MODEL_NAME)
    print("Model loaded successfully")

    test_response = generate_response(model, tokenizer, "Say hello in one sentence")
    print("Test response:", test_response)
