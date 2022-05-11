"""
This module:  train model.
"""
# pylint: disable-msg=too-many-locals

from typing import List, Optional, Dict, Tuple
import sys
import warnings
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier
from load_data import read_data
from processing import FeaturesGenerator
from model import Model

warnings.filterwarnings("ignore")



PARAMS = [
    {
        "num_leaves": 203,
        'max_depth': 3,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "min_child_samples": 30,
        "subsample": 0.622584668281795,
        "subsample_freq": 5,
        "colsample_bytree": 0.17974037482084979,
        "reg_alpha": 3.6558239011297524,
        "reg_lambda": 5.865351102213303,
        "min_child_weight": 0.4388354272235035,
        "min_split_gain": 0.006193549861032794,
        "random_state": 42,
    },
    {
        "num_leaves": 227,
        'max_depth': 3,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "min_child_samples": 84,
        "subsample": 0.7048761531428217,
        "subsample_freq": 5,
        "colsample_bytree": 0.2806650533512902,
        "reg_alpha": 6.062594325136659,
        "reg_lambda": 4.427396504735556,
        "random_state": 42,
        "min_child_weight": 0.12139028719958726,
        "min_split_gain": 0.0015430980179050107,
    },
    {
        "num_leaves": 247,
        'max_depth': 3,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "min_child_samples": 87,
        "subsample": 0.8640475862646084,
        "subsample_freq": 5,
        "colsample_bytree": 0.22858610039131327,
        "reg_alpha": 3.750962348348377,
        "reg_lambda": 4.394908953876854,
        'random_state': 42,
        "min_child_weight": 0.021180547250587045,
        "min_split_gain": 0.0018611688327432564,
    },
]


def get_auc(y_true: np.array, probas_pred: np.array) -> Tuple[str, float, bool]:
    """
    Calculate avg auc for multiclass classification

    Args:
        y_true (np.array): array of labels
        probas_pred (np.array): array of predicted probs

    Returns:
        Tuple[str, float, bool]: name of metrics, the value of metric, higher = better?
    """
    aucs = []
    preds = np.array(probas_pred)
    preds = preds.reshape(-1, 6)
    for index in range(6):
        auc = roc_auc_score(y_true == index, preds[:, index])
        aucs.append(auc)
    score = np.mean(auc)
    return "mean_auc", score, True


def drop_null_targets(
    x_raw: pd.DataFrame, y_raw: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop all rows where target is missing

    Args:
        x_raw (pd.DataFrame): features
        y_raw (pd.Series): targets

    Returns:
        Tuple[pd.DataFrame, pd.Series]: features and targets without rows with missing data
    """
    x_raw = x_raw.reset_index(drop=True)
    y_raw = y_raw.reset_index(drop=True)
    ind = y_raw.notnull()
    return x_raw.loc[ind], y_raw[ind]


def train_model(
    feature: pd.DataFrame, target1: pd.Series
) -> Tuple[List[LGBMClassifier], List[LGBMClassifier]]:
    """
    Train models for t and t+3 predictions

    Args:
        feature (pd.DataFrame): input features
        target1 (pd.Series): t targets

    Returns:
        List[LGBMClassifier]: list of trained models
    """
    models_1 = []
    models_3 = []
    for params in PARAMS:
        model_1 = LGBMClassifier(**params)
        model_1.fit(feature, target1, eval_set=(feature, target1), verbose=100)

        models_1 += [model_1]
        models_3 += [model_1]

    return models_1, models_3


def train_models(
    feature: pd.DataFrame,
    target1: pd.Series,
    target3: pd.Series,
    meta: pd.Series,
    params_to_train: Optional[List[Dict]] = None,
) -> Tuple[List[LGBMClassifier], List[LGBMClassifier], float]:
    """
    Train models

    Args:
        feature (pd.DataFrame): input features
        target1 (pd.Series): t targets
        target3 (pd.Series): t+3 targets
        meta (pd.Series): metadata
        params_to_train (Optional[List[Dict]], optional): list of dicts of hyperparams to use for
        training. Defaults to None.

    Returns:
        Tuple[List[LGBMClassifier], List[LGBMClassifier], float]: t and t+3 trained models,
        mean validation score
    """
    if params_to_train is None:
        params_to_train = PARAMS

    # use repeated statified group kfold for training
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            feature, target1, meta
        )
    )
    folds += list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100).split(
            feature, target1, meta
        )
    )

    models_1, models_3 = [], []
    for params in params_to_train:
        print("Training model with new params")
        print(params)

        scores_1 = []
        fin_scores = []
        for fold_ind, (train_index, test_index) in tqdm(enumerate(folds)):
            # get split
            x_train, x_test = feature.loc[train_index], feature.loc[test_index]
            y_1_train, y_1_test = target1[train_index], target1[test_index]
            y_3_train, y_3_test = target3[train_index], target3[test_index]
            # meta_train, meta_val = META[train_index], META[test_index]

            # drop null targets
            x_train_1, y_train_1 = drop_null_targets(x_train, y_1_train)
            _, _ = drop_null_targets(x_train, y_3_train)
            x_val_1, y_val_1 = drop_null_targets(x_test, y_1_test)
            _, _ = drop_null_targets(x_test, y_3_test)

            try:
                # train models
                model_1 = LGBMClassifier(**params)
                model_1.fit(
                    x_train_1,
                    y_train_1,
                    eval_set=(x_val_1, y_val_1),
                    verbose=100,
                    early_stopping_rounds=50,
                )
                auc_1 = get_auc(
                    y_true=y_val_1, probas_pred=model_1.predict_proba(x_val_1)
                )[1]
                scores_1.append(auc_1)
                models_1.append(model_1)

                models_3 = models_1
                auc_3 = auc_1

                fin_score = 0.7 * auc_1 + 0.3 * auc_3
                fin_scores.append(fin_score)
            except ValueError as exception:
                print(f"Error: {exception}. Ignoring")
            print(f"Fold {fold_ind}. Score: {fin_scores}")
        mean_score = np.mean(fin_scores)
    return models_1, models_3, mean_score


def main():
    """
    Run training
    """
    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Training input file is missing.")
        return 1

    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Training output file is missing.")
        return 1

    print("Training started.")

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data, _ = read_data(input_file)
    processor = FeaturesGenerator()
    feature, target1, _, _ = processor.generate_features_train(data)
    models_1, models_3 = train_model(feature, target1)

    main_model = Model(
        features=list(feature.columns),
        preprocessor=processor,
        models_1=models_1,
        models_3=models_3,
    )
    with open(output_file, "wb") as file:
        pickle.dump(main_model, file, pickle.HIGHEST_PROTOCOL)

    print("Training finished.")
    return 0


if __name__ == "__main__":
    main()
