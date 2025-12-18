import json
import os
import numpy as np
import ollama
import pickle
import sys

# To run, specify ollama host
OLLAMA_HOST = ''
OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST)

DATASET_PATH = '../dataset'
ANSWER_TYPE_TBD = 'TBD'


def apply_slots(slot_id, question_existence, questions, slots):
    return {
        slot: {
            'question_existence': question_existence.replace(slot_id, slot),
            'questions': [
                question.replace(slot_id, slot) for question in questions
            ]
        } for slot in slots
    }

def parse_file(fname):
    with open(fname) as f:
        data = json.load(f)
    slot_id = data['slot']
    question_existence = data['question_existence']
    questions = data['questions']
    entities_real = data['real']
    entities_fabricated = data['fabricated']
    return {
        'real': apply_slots(slot_id, question_existence, questions, entities_real),
        'fabricated': apply_slots(slot_id, question_existence, questions, entities_fabricated)
    }

def answer_yes_no(question, model):
    response = OLLAMA_CLIENT.chat(model=model, messages=[
            {
                'role': 'user',
                'content': question,
            },
        ], 
        options={
            'num_predict': 64
        })
    return response.message.content

def answer_free(question, model):
    response = OLLAMA_CLIENT.chat(model=model, messages=[
        {
            'role': 'user',
            'content': question,
        },
        ], 
        options={
            'num_predict': 64
        })
    return response.message.content

def run_eval(experiment_name, llm):
    # Prepare results folder
    results_path = os.path.join('results', experiment_name)
    os.makedirs(results_path, exist_ok=True)

    # Read Dataset
    files = filter(lambda x: x != "TEMPLATE.json", os.listdir(DATASET_PATH))
    files = filter(lambda x: x not in os.listdir(results_path), files)
    dataset = {f: parse_file(os.path.join(DATASET_PATH, f)) for f in files}


    # Write Metadata
    metadata = OLLAMA_CLIENT.show(llm)
    with open(os.path.join(results_path, 'meta.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # Compute results
    for category_file, category in dataset.items():
        print(f'Processing {category_file}...')
        answers = {
            'real': {
                entity: {
                    'answer_existence': {
                        'text': answer_yes_no(questions['question_existence'], llm),
                        'annotations': {}
                    },
                    'answers': [
                        { 
                            'text': answer_free(q, llm),
                            'annotations': {}
                        }
                        for q in questions['questions']
                    ]
                }
                for entity, questions in category['real'].items()
            },
            'fabricated': {
                entity: {
                    'answer_existence': {
                        'text': answer_yes_no(questions['question_existence'], llm),
                        'annotations': {}
                    },
                    'answers': [
                        { 
                            'text': answer_free(q, llm),
                            'annotations': {}
                        }
                        for q in questions['questions']
                    ]
                }
                for entity, questions in category['fabricated'].items()
            }
        }

        # Save answers to json
        with open(os.path.join(results_path, category_file), 'w') as f:
            json.dump(answers, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python runExperiment.py <name> <LLM>')
        sys.exit(-1)
    experiment_name = sys.argv[1]
    experiment_llm = sys.argv[2]
    run_eval(experiment_name, experiment_llm)


