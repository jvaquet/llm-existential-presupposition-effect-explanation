
import numpy as np
import os
import sys
import sklearn
import sklearn.decomposition
import sklearn.model_selection
import sklearn.svm
import sklearn.pipeline
import sklearn.ensemble
import sklearn.neural_network

def load_data(question_type, DATA_PATH):
    Xs = []
    ys = []
    data_categories = []
    #categories = os.listdir(DATA_PATH)
    categories = ['BIRD.npz', 'PROGRAMMING_LANGUAGE.npz', 'DRUG.npz', 'ACTOR.npz', 'LAKE.npz', 'STAR.npz', 'BOOK.npz', 'COUNTRY.npz', 'MUSHROOM.npz', 'CAR.npz', 'PHONE.npz', 'REPTILE.npz', 'THEORY.npz', 'SPORT.npz', 'MINERAL.npz', 'SCIENTIST.npz', 'MOUNTAIN.npz', 'TUNNEL.npz', 'TREE.npz', 'BRIDGE.npz', 'SONG.npz', 'POLITICIAN.npz', 'MAMMAL.npz', 'AIRPORT.npz', 'DISEASE.npz', 'CITY.npz', 'ATHLETE.npz', 'MOVIE.npz', 'COMPANY.npz', 'AWARD.npz', 'GALAXY.npz', 'ARTIST.npz', 'WRITER.npz', 'PLANT.npz', 'LIBRARY_PYTHON.npz', 'SPORTS_TEAM.npz', 'MUSEUM.npz', 'COMPOUND.npz', 'ISLAND.npz', 'PAINTING.npz', 'RIVER.npz', 'AMPHIBIAN.npz', 'LIBRARY_JAVASCRIPT.npz']
    
    for idx_category, category in enumerate(categories):
        cur_data = np.load(os.path.join(DATA_PATH, category))

        # Load olmo data differently, as it contains all layers
        if DATA_PATH == 'data_olmo':
            cur_data_real = cur_data[f'activations_REAL_{question_type}_LAY'][-1]
            cur_data_fabr = cur_data[f'activations_FABR_{question_type}_LAY'][-1]
        else:
            cur_data_real = cur_data[f'activations_REAL_{question_type}_LAY']
            cur_data_fabr = cur_data[f'activations_FABR_{question_type}_LAY']

        Xs.append(cur_data_real)
        ys.append(np.zeros(cur_data_real.shape[0]))
        Xs.append(cur_data_fabr)
        ys.append(np.ones(cur_data_fabr.shape[0]))

        data_categories.append(np.full(cur_data_real.shape[0]+cur_data_fabr.shape[0], idx_category))

    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(data_categories)


def score(y, probas):
    pred = probas > 0.5
    n_samples = X.shape[0]
    n_samples_true = np.sum(y==1)
    n_samples_false = np.sum(y==0)
    
    acc = np.sum(y == pred) / n_samples

    TPR = np.sum(pred[y==1] == 1) / n_samples_true
    TNR = np.sum(pred[y==0] == 0) / n_samples_false

    return {
        'ACC': acc,
        'TPR': TPR,
        'TNR': TNR,
        'ROC_AUC': sklearn.metrics.roc_auc_score(y, probas)
    }


def perform_experiment(X, y, groups, model_type):

    if model_type == 'LogReg':
        model = sklearn.linear_model.LogisticRegression()
        method = 'predict_proba'
    elif model_type == 'SVM':
        model = sklearn.svm.SVC()
        method = 'decision_function'
    elif model_type == 'RF':
        model = sklearn.ensemble.RandomForestClassifier(max_depth=6)
        method = 'predict_proba'
    elif model_type == 'MLP':
        model = sklearn.neural_network.MLPClassifier()
        method = 'predict_proba'

    res = sklearn.model_selection.cross_val_predict(model,
        X,
        y,
        groups=groups,
        cv=sklearn.model_selection.LeaveOneGroupOut(),
        n_jobs=64,
        method=method)
    
    if model_type == 'SVM':
        probas = res+0.5
        decision_scores = res
    else:
        probas = res[:,1]
        decision_scores = res[:,1]
    
    scores = score(y, probas)
    
    return scores, decision_scores


if __name__ == '__main__':
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    DATA_PATH = sys.argv[1]
    RESULTS_PATH = sys.argv[2]
    models = sys.argv[3:]

    print('Starting experiments using:')
    print('Data:', DATA_PATH)
    print('Results:', RESULTS_PATH)
    print('Models:', models)

    os.makedirs(RESULTS_PATH, exist_ok=True)

    X, y, groups = load_data('PROP', DATA_PATH)
    for model_type in models:
        for idx in range(5):
            fname = f'PROP_LAY_39_id_{model_type}_{idx}.npz'
            print(f'Generating {fname}...')
            fpath = os.path.join(RESULTS_PATH, fname)
            scores, probas = perform_experiment(X, y, groups, model_type)
            np.savez(fpath, probas=probas, **scores)
