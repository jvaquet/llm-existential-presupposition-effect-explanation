import os
import json

DATASET_PATH = '../../dataset/'
BATCH_REQUESTS_PATH = 'request_batches'

MODEL = 'gpt-4o-2024-11-20'
MAX_TOKENS = 64


def generate_request(id, message):
    return json.dumps({
        'custom_id': id,
        'method': 'POST',
        'url': '/v1/responses',
        'body': {
            'model': MODEL,
            'input': message,
            'max_output_tokens': MAX_TOKENS,
            'tool_choice': 'none'
        }
    })


def prepare_category(category):
    with open(os.path.join(DATASET_PATH, category), 'r') as f:
        dataset = json.load(f)
    
    requests = []
    slot_id = dataset['slot']
    for existence in ['real', 'fabricated']:
        for entity in dataset[existence]:
            question_existence = dataset['question_existence'].replace(slot_id, entity)
            questions = [question.replace(slot_id, entity) for question in dataset['questions']]

            requests.append(generate_request(f'{existence[0]}|{entity}|e', question_existence))
            requests += [generate_request(f'{existence[0]}|{entity}|q{question_idx}', question) for question_idx, question in enumerate(questions)]

    with open(os.path.join(BATCH_REQUESTS_PATH, f'{category}l'), 'w') as batch_file:
        batch_file.writelines([f'{request}\n' for request in requests])


def prepare_dataset():
    os.makedirs(BATCH_REQUESTS_PATH, exist_ok=True)
    for category in os.listdir(DATASET_PATH):
        prepare_category(category)


if __name__ == '__main__':
    prepare_dataset()