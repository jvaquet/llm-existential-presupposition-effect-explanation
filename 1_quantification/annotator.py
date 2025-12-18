import os
import json
import sys
import simple_term_menu
import numpy as np

STYLE_HIGHLIGHT = '\033[1;4;36m'
STYLE_BOLD = '\033[0;1;37m'
STYLE_SUBTLE = '\033[2m'
STYLE_RESET = '\033[0m'

CURSOR_UP = '\033[F'

DATASET_PATH = '../dataset'
RESULTS_PATH = './results/'


ANNOTATION_TYPES_UNANNOTATED = ['TBD']
ANNOTATOR_HUMAN = 'HUMAN'

def print_subtle(text, end='\n'):
    print(f'{STYLE_SUBTLE}{text}{STYLE_RESET}', end=end)

ANNOTATION_MENU_CHOICES = [
    '[u]Unknown', 
    '[r]Real', 
    '[p]Property',
    '[a]Avoid',
    '[f]Fabricated',
    '[n]No Info',
    '[i]Instruct',
    '[g]Generic',
    '[b]Back', 
    '[q]Save and Quit']
ANNOTATION_MENU_CHOICE_LABELS = {
    '[u]Unknown': 'UNKN',
    '[r]Real': 'REAL',
    '[p]Property': 'PROP',
    '[a]Avoid': 'AVD',
    '[f]Fabricated': 'FABR',
    '[n]No Info': 'NOINFO',
    '[g]Generic': 'GEN',
    '[i]Instruct': 'INST'
}

annotation_menu = simple_term_menu.TerminalMenu(ANNOTATION_MENU_CHOICES)

def clear_screen():
    print("\033[H\033[J", end="")

def load_dataset(file):
    path = os.path.join(DATASET_PATH, file)
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset

def make_annotation_screen(cur_entity, cur_slot_id, cur_answer, cur_question,
                           last_entity, last_slot_id, last_answer, last_question,
                           progress_cur, progress_total,
                           category_progress_cur, category_progress_total):
    
    # Generate strings to display
    cur_answer_styled = f'{STYLE_BOLD}{cur_answer['text']}{STYLE_RESET}'\
        .replace(cur_entity, f'{STYLE_HIGHLIGHT}{cur_entity}{STYLE_BOLD}')
    cur_question_styled = cur_question.replace(cur_slot_id, cur_entity)

    last_answer_styled = f'{STYLE_BOLD}{last_answer['text']}{STYLE_RESET}'\
        .replace(last_entity, f'{STYLE_HIGHLIGHT}{last_entity}{STYLE_BOLD}')
    last_question_styled = last_question.replace(last_slot_id, last_entity)
    last_annotation = last_answer['annotations'][ANNOTATOR_HUMAN] \
            if ANNOTATOR_HUMAN in last_answer['annotations'].keys() \
            else '-'

    # Retrieve terminal dimensions
    terminal_size = os.get_terminal_size()
    terminal_width = terminal_size.columns
    terminal_height = terminal_size.lines

    # Calculate lines required for sections
    n_lines_cur = 5 + np.ceil(len(cur_answer_styled) / terminal_width) + np.ceil(len(cur_question_styled) / terminal_width) + cur_answer_styled.count('\n')
    n_lines_last = 8 + np.ceil(len(last_answer_styled) / terminal_width) + np.ceil(len(last_question_styled) / terminal_width) + last_answer_styled.count('\n')

    n_newlines = int(max(0, terminal_height - (n_lines_cur + n_lines_last)))

    # Print actual screen
    # Current Question
    clear_screen()
    print_subtle(f'##### {cur_entity} ({progress_cur}/{progress_total}) [{category_progress_cur}/{category_progress_total}] #####'.center(terminal_width))
    print()
    print_subtle('##### Answer #####'.center(terminal_width))
    print(cur_answer_styled)
    print()
    print_subtle('##### Question #####'.center(terminal_width))
    print_subtle(cur_question_styled)

    # Newlines for menu
    print('\n'*n_newlines, end='')

    # Last Question
    print_subtle('#'*terminal_width)
    print_subtle(f'##### {last_entity} #####'.center(terminal_width))
    print()
    print_subtle('##### Answer #####'.center(terminal_width))
    print(last_answer_styled)
    print()
    print_subtle(f'Annotation: {last_annotation}')
    print()
    print_subtle('##### Question #####'.center(terminal_width))
    print_subtle(last_question_styled, end='')

    # Display menu
    print(CURSOR_UP * int(n_newlines + n_lines_last - 2))
    menu_idx = annotation_menu.show()

    return ANNOTATION_MENU_CHOICES[menu_idx]


def index_file(experiment, file):
    path = os.path.join(RESULTS_PATH, experiment, file)
    with open(path, 'r') as f:
        results = json.load(f)

    n_answers = 0
    to_annotate = []

    for existence, all_entity_answers in results.items():
        for entity, entity_answers in all_entity_answers.items():
            for answer_idx, answer in enumerate(entity_answers['answers']):
                n_answers += 1
                if len(answer['annotations']) == 0:
                    to_annotate.append((existence, entity, answer_idx))

    return {
        'total': n_answers,
        'to_annotate': to_annotate
    }

def make_index_screen(experiment, experiment_index):
    # Generate choices for single categories
    choices_category, choices_category_display = map(list, zip(*[(category, f'{category.split(".")[0]} ({category_index["total"]-len(category_index["to_annotate"])}/{category_index["total"]})') for category, category_index in experiment_index.items()]))

    # Generate Choice for ALL
    n_done, n_total = map(list, zip(*[(category_index["total"]-len(category_index["to_annotate"]), category_index["total"]) for category_index in experiment_index.values()]))
    choice_all_display = f'ALL ({sum(n_done)}/{sum(n_total)})'

    # Retrieve terminal dimensions
    terminal_size = os.get_terminal_size()
    terminal_width = terminal_size.columns

    # Display menu
    clear_screen()
    print('#'*terminal_width)
    print(f' Annotate Experiment: {experiment} '.center(terminal_width, '#'))
    print('#'*terminal_width)

    menu_categories = simple_term_menu.TerminalMenu([choice_all_display] + choices_category_display + ['[q]Quit'])
    choice = menu_categories.show()
    
    # Return selected category
    if choice == 0:
        return 'ALL' 
    elif choice == len(choices_category)+1:
        return 'QUIT'
    else:
        return choices_category[choice-1]
    

def annotate_category(experiment, category_index, category, category_progress_cur=0, category_progress_total=1):
    path = os.path.join(RESULTS_PATH, experiment, category)
    dataset = load_dataset(category)
    category_slot_id = dataset['slot']

    with open(path, 'r') as f:
        results = json.load(f)

    to_annotate = category_index['to_annotate']
    idx = 0
    while idx < len(to_annotate):
        last_existence, last_entity, last_answer_idx = to_annotate[idx-1] if idx > 0 else to_annotate[idx]
        existence, entity, answer_idx = to_annotate[idx]

        choice = make_annotation_screen(entity, \
                                        category_slot_id, \
                                        results[existence][entity]['answers'][answer_idx], \
                                        dataset['questions'][answer_idx],
                                        last_entity, \
                                        category_slot_id, \
                                        results[last_existence][last_entity]['answers'][last_answer_idx], \
                                        dataset['questions'][last_answer_idx],
                                        idx,
                                        len(to_annotate),
                                        category_progress_cur, 
                                        category_progress_total)

        if choice == '[q]Save and Quit':
            with open(path, 'w') as f:
                json.dump(results, f)
            return True
        elif choice == '[b]Back':
            idx -= 2
        else:
            results[existence][entity]['answers'][answer_idx]['annotations'][ANNOTATOR_HUMAN] = ANNOTATION_MENU_CHOICE_LABELS[choice]

        idx += 1
    
    with open(path, 'w') as f:
        json.dump(results, f)
    return False


def annotate_all(experiment, experiment_index):
    for category_idx, category in enumerate(experiment_index.keys()):
        quit = annotate_category(experiment, experiment_index[category], category, 
                                 category_progress_cur=category_idx, 
                                 category_progress_total=len(experiment_index.keys()))
        if quit:
            return


def main(experiment):
    while True:
        # Index all files
        experiment_index = {file: index_file(experiment, file) for file in os.listdir(os.path.join(RESULTS_PATH, experiment)) if file.endswith('.json')}

        # Choose category
        cur_category = make_index_screen(experiment, experiment_index)
        
        # Annotate Category, Categories or Quit
        if cur_category == 'ALL':
            annotate_all(experiment, experiment_index) 
        elif cur_category == 'QUIT':
            break
        else:
            annotate_category(experiment, experiment_index[cur_category], cur_category)

    


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python annotator.py <experiment>')
    experiment = sys.argv[1]
    main(experiment)