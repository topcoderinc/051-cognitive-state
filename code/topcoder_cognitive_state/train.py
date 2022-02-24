from pyexpat import features
from typing import List, Tuple
import sys
import warnings
warnings.filterwarnings('ignore')

import pickle
from functools import reduce

import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier

from topcoder_cognitive_state.load_data import read_data
from topcoder_cognitive_state.processing import FeaturesGenerator
from sklearn.model_selection import StratifiedGroupKFold
from topcoder_cognitive_state.model import Model


PARAMS = [
    {
        "num_leaves": 151,
        "learning_rate": 0.12,
        "n_estimators": 500,
        "min_child_samples": 4,
        "subsample": 0.96,
        "subsample_freq": 5,
        "colsample_bytree": 0.115,
        "reg_alpha": 0.105,
        "reg_lambda": 9.90,
        "min_child_weight": 0.005519,
        "min_split_gain": 1.94e-14,
        'random_state': 42
    },

    {
        'num_leaves': 79,
        'learning_rate': 0.12,
        'n_estimators': 600,
        'min_child_samples': 14,
        'subsample': 0.75,
        'subsample_freq': 5,
        'colsample_bytree': 0.75,
        'reg_alpha': 2.2,
        'reg_lambda': 1.5,
        'random_state': 424242,
        'min_child_weight': 6.681437316563333e-12,
        'min_split_gain': 0.00039529173804292325,
    },

    {
        'num_leaves': 23,
        'learning_rate': 0.12,
        'n_estimators': 500,
        'min_child_samples': 30,
        'subsample': 0.6,
        'subsample_freq': 5,
        'colsample_bytree': 0.4,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'random_state': 4242,
        'min_child_weight': 0.28,
        'min_split_gain': 9.793058539831146e-08,
    }

]


def get_auc(y_true, probas_pred):
    aucs = []
    preds = np.array(probas_pred)
    preds = preds.reshape(-1, 6)
    for i in range(6):
        auc = roc_auc_score(y_true == i, preds[:, i])
        aucs.append(auc)
    score = np.mean(auc)
    return "mean_auc", score, True


def drop_null_targets(X_raw, y_raw):
    X = X_raw.reset_index(drop=True)
    y = y_raw.reset_index(drop=True)
    ind = y.notnull()
    return X.loc[ind], y[ind]


def train_model(X, Y1, Y3):
    models_1 = []
    models_3 = []
    for params in PARAMS:
        print(params)
        model_1 = LGBMClassifier(**params)
        model_1.fit(X, Y1, eval_set=(X, Y1), verbose=100)

        models_1 += [model_1]
        models_3 += [model_1]

        print("\n")
    return models_1, models_3


def train_models(X, Y1, Y3, META, params_to_train=None):
    if params_to_train is None:
        params_to_train = PARAMS

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, Y1, META))
    folds += list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100).split(X, Y1, META))

    models_1, models_3 = [], []
    for params in params_to_train:
        print("Training model with new params")
        print(params)

        scores_1, scores_3 = [], []
        fin_scores = []
        for fold_ind, (train_index, test_index) in tqdm(enumerate(folds)):
            # get split
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_1_train, y_1_test = Y1[train_index], Y1[test_index]
            y_3_train, y_3_test = Y3[train_index], Y3[test_index]
            # meta_train, meta_val = META[train_index], META[test_index]
            
            # drop null targets
            X_train_1, y_train_1 = drop_null_targets(X_train, y_1_train)
            X_train_3, y_train_3 = drop_null_targets(X_train, y_3_train)
            X_val_1, y_val_1 = drop_null_targets(X_test, y_1_test)
            X_val_3, y_val_3 = drop_null_targets(X_test, y_3_test)
            
            try:
                # train models
                model_1 = LGBMClassifier(**params)
                model_1.fit(
                    X_train_1, y_train_1, 
                    eval_set=(X_val_1, y_val_1), 
                    # eval_metric=get_auc,
                    verbose=100,
                    early_stopping_rounds=50
                )
                auc_1 = get_auc(y_true=y_val_1, probas_pred=model_1.predict_proba(X_val_1))[1]
                scores_1.append(auc_1)
                models_1.append(model_1)

                models_3 = models_1
                auc_3 = auc_1
                scores_3 = scores_1

                fin_score = 0.7 * auc_1 + 0.3 * auc_3
                fin_scores.append(fin_score)
            except Exception as e:
                print(f"Error: {e}. Ignoring")
            print(f"Fold {fold_ind}. Score: {fin_score}")
        print("Params")
        print(params)
        mean_score = np.mean(fin_scores)
        print(f"Mean score = {mean_score}")
    return models_1, models_3, mean_score


def main():
    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Training input file is missing.")
        return 1
    
    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Training output file is missing.")
        return 1

    print('Training started.')
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data, _ = read_data(input_file)
    processor = FeaturesGenerator()
    X, Y1, Y3, META = processor.generate_featres_train(data)
    models_1, models_3 = train_model(X, Y1, Y3)

    main_model = Model(
        features=list(X.columns),
        preprocessor=processor, 
        models_1=models_1, 
        models_3=models_3
    )
    with open(output_file, 'wb') as f:
        pickle.dump(main_model, f, pickle.HIGHEST_PROTOCOL)

    print('Training finished.')
    return 0

if __name__ == "__main__":
    main()
