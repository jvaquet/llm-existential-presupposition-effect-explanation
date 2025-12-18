from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import os
import json

DATASET_PATH = '../dataset/'
DATA_PATH = 'data_olmo'
N_LAYERS = 40
N_LATENT_DIM = 5120


def get_chat_tokens(question):
    tokens = olmo_tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI.'},
            {'role': 'user', 'content': question}
        ],
        add_generation_prompt=True, return_tensors='pt')
    return tokens


cur_activations = {'MLP': {}, 'ATT': {}}
def save_activation(activation_type, activation_layer):
    def hook(module, input, output):
        cur_activations[activation_type][activation_layer] = output.cpu().numpy()
    return hook


def get_latent_representations(questions):
    n_questions = len(questions)
    
    activations_mlp = np.zeros((N_LAYERS, n_questions, N_LATENT_DIM), dtype=np.float16)
    activations_att = np.zeros((N_LAYERS, n_questions, N_LATENT_DIM), dtype=np.float16)
    activations_lay = np.zeros((N_LAYERS, n_questions, N_LATENT_DIM), dtype=np.float16)

    for idx_question, question in enumerate(questions):
        input_tokens = get_chat_tokens(question)
        res = olmo_model.forward(input_tokens, output_hidden_states=True)

        for idx_layer in range(N_LAYERS):
            activations_mlp[idx_layer, idx_question] = cur_activations['MLP'][idx_layer][0, -1]
            activations_att[idx_layer, idx_question] = cur_activations['ATT'][idx_layer][0, -1]
            activations_lay[idx_layer, idx_question] = res.hidden_states[idx_layer+1][0, -1].cpu().numpy()

    return activations_mlp, activations_att, activations_lay


def process_category(category):
    print(f'Processing {category}', end='', flush=True)
    with open(os.path.join(DATASET_PATH, category), 'rb') as f:
        data = json.load(f)

    questions_real_existence = [data['question_existence'].replace(data['slot'], entity) for entity in data['real']]
    questions_fabr_existence = [data['question_existence'].replace(data['slot'], entity) for entity in data['fabricated']]
    questions_real_property = [template.replace(data['slot'], entity) for entity in data['real'] for template in data['questions']]
    questions_fabr_property = [template.replace(data['slot'], entity) for entity in data['fabricated'] for template in data['questions']]

    activations_REAL_EX_MLP, activations_REAL_EX_ATT, activations_REAL_EX_LAY = get_latent_representations(questions_real_existence)
    print('.', end='', flush=True)
    activations_FABR_EX_MLP, activations_FABR_EX_ATT, activations_FABR_EX_LAY = get_latent_representations(questions_fabr_existence)
    print('.', end='', flush=True)
    activations_REAL_PROP_MLP, activations_REAL_PROP_ATT, activations_REAL_PROP_LAY = get_latent_representations(questions_real_property)
    print('.', end='', flush=True)
    activations_FABR_PROP_MLP, activations_FABR_PROP_ATT, activations_FABR_PROP_LAY = get_latent_representations(questions_fabr_property)
    print('.', flush=True)

    np.savez(os.path.join(DATA_PATH, category.split('.')[0]),
             activations_REAL_EX_MLP = activations_REAL_EX_MLP, 
             activations_REAL_EX_ATT = activations_REAL_EX_ATT, 
             activations_REAL_EX_LAY = activations_REAL_EX_LAY,
             activations_FABR_EX_MLP = activations_FABR_EX_MLP, 
             activations_FABR_EX_ATT = activations_FABR_EX_ATT, 
             activations_FABR_EX_LAY = activations_FABR_EX_LAY,
             activations_REAL_PROP_MLP = activations_REAL_PROP_MLP, 
             activations_REAL_PROP_ATT = activations_REAL_PROP_ATT, 
             activations_REAL_PROP_LAY = activations_REAL_PROP_LAY,
             activations_FABR_PROP_MLP = activations_FABR_PROP_MLP, 
             activations_FABR_PROP_ATT = activations_FABR_PROP_ATT, 
             activations_FABR_PROP_LAY = activations_FABR_PROP_LAY)


def process_dataset():
    os.makedirs(DATA_PATH, exist_ok=True)
    categories = os.listdir(DATASET_PATH)
    for category in categories:
        process_category(category)


if __name__ == '__main__':
    with torch.no_grad():
        olmo_model = AutoModelForCausalLM.from_pretrained('allenai/OLMo-2-1124-13B-Instruct', 
                                                    torch_dtype=torch.float16,
                                                    device_map='auto')
        print('Loaded Model', flush=True)
        olmo_tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-2-1124-13B-Instruct')
        print('Loaded Tokenizer', flush=True)

        print('Registering Hooks', end='', flush=True)
        for idx_layer in range(N_LAYERS):
            olmo_model.model.layers[idx_layer].mlp.down_proj.register_forward_hook(save_activation('MLP', idx_layer))
            olmo_model.model.layers[idx_layer].self_attn.o_proj.register_forward_hook(save_activation('ATT', idx_layer))
            print('.', end='')
        print('Done', flush=True)

        process_dataset()

