from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import os
import json

DATASET_PATH = '../dataset/'
DATA_PATH = 'data_deepseek'
N_LATENT_DIM = 5120


def get_chat_tokens(question):
    tokens = olmo_tokenizer.apply_chat_template([
            {'role': 'user', 'content': question}
        ],
        add_generation_prompt=True, return_tensors='pt')
    return tokens



def get_latent_representations(questions):
    n_questions = len(questions)

    activations_lay = []

    for idx_question, question in enumerate(questions):
        input_tokens = get_chat_tokens(question)
        res = olmo_model.forward(input_tokens, output_hidden_states=True)

        activations_lay.append(res.hidden_states[-1][0, -1].cpu().numpy())

    return np.array(activations_lay).astype(np.float16)
            


def process_category(category):
    print(f'Processing {category}', end='', flush=True)
    with open(os.path.join(DATASET_PATH, category), 'rb') as f:
        data = json.load(f)

    questions_real_existence = [data['question_existence'].replace(data['slot'], entity) for entity in data['real']]
    questions_fabr_existence = [data['question_existence'].replace(data['slot'], entity) for entity in data['fabricated']]
    questions_real_property = [template.replace(data['slot'], entity) for entity in data['real'] for template in data['questions']]
    questions_fabr_property = [template.replace(data['slot'], entity) for entity in data['fabricated'] for template in data['questions']]

    activations_REAL_EX_LAY = get_latent_representations(questions_real_existence)
    print('.', end='', flush=True)
    activations_FABR_EX_LAY = get_latent_representations(questions_fabr_existence)
    print('.', end='', flush=True)
    activations_REAL_PROP_LAY = get_latent_representations(questions_real_property)
    print('.', end='', flush=True)
    activations_FABR_PROP_LAY = get_latent_representations(questions_fabr_property)
    print('.', flush=True)

    np.savez(os.path.join(DATA_PATH, category.split('.')[0]),
             activations_REAL_EX_LAY = activations_REAL_EX_LAY,
             activations_FABR_EX_LAY = activations_FABR_EX_LAY,
             activations_REAL_PROP_LAY = activations_REAL_PROP_LAY, 
             activations_FABR_PROP_LAY = activations_FABR_PROP_LAY)


def process_dataset():
    os.makedirs(DATA_PATH, exist_ok=True)
    categories = os.listdir(DATASET_PATH)
    for category in categories:
        process_category(category)


if __name__ == '__main__':
    with torch.no_grad():
        olmo_model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 
                                                    torch_dtype=torch.float16,
                                                    device_map='auto')
        print('Loaded Model', flush=True)
        olmo_tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Llama-70B')
        print('Loaded Tokenizer', flush=True)

        process_dataset()

