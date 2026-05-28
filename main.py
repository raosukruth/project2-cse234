import argparse
import json


def predict_schema_links(question, db_id, schemas_dir):
    return {}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--schemas_dir', default='./schemas')
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)
    preds = []
    for it in items:
        links = predict_schema_links(it['question'], it['db_id'], args.schemas_dir)
        preds.append({'question_id': it['question_id'], 'schema_links': links})
    with open(args.output, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")
