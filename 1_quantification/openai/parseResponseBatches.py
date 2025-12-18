import json
import os

DATASET_PATH = '../../dataset/'
RESPONSES_PATH = 'response_batches'
RESULTS_PATH = '../results/gpt-4o'


def parse_category(category):
    
    with open(os.path.join(DATASET_PATH, category), 'r') as f:
        dataset = json.load(f)

    with open(os.path.join(RESPONSES_PATH, f'{category}l'), 'r') as f:
        responses_str = f.readlines()
        responses = {response['custom_id']: response['response']['body']['output'][0]['content'][0]['text'] for response in map(json.loads, responses_str)}

    results = {
        existence: {
            entity: {
                'answer_existence': {
                    'text': responses[f'{existence[0]}|{entity}|e'],
                    'annotations': {}
                },
                'answers': [
                    { 
                        'text': responses[f'{existence[0]}|{entity}|q{question_idx}'],
                        'annotations': {}
                    }
                    for question_idx in range(len(dataset['questions']))
                ]
            }
            for entity in dataset[existence]
        }
        for existence in ['real', 'fabricated']
    }

    with open(os.path.join(RESULTS_PATH, category), 'w') as f:
        json.dump(results, f)


def parse_responses():
    os.makedirs(RESULTS_PATH, exist_ok=True)
    for category in os.listdir(RESPONSES_PATH):
        parse_category(category[:-1])


if __name__ == '__main__':
    parse_responses()