from typing import List, Optional, Dict, Tuple
import sys
import warnings
import pickle

from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier
import pandas as pd

from topcoder_cognitive_state.load_data import read_data
from topcoder_cognitive_state.processing import FeaturesGenerator
from sklearn.model_selection import StratifiedGroupKFold
from topcoder_cognitive_state.model import Model

warnings.filterwarnings("ignore")


# lightgbm model hyperparams
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
        "random_state": 42,
    },
    {
        "num_leaves": 79,
        "learning_rate": 0.12,
        "n_estimators": 600,
        "min_child_samples": 14,
        "subsample": 0.75,
        "subsample_freq": 5,
        "colsample_bytree": 0.75,
        "reg_alpha": 2.2,
        "reg_lambda": 1.5,
        "random_state": 424242,
        "min_child_weight": 6.681437316563333e-12,
        "min_split_gain": 0.00039529173804292325,
    },
    {
        "num_leaves": 23,
        "learning_rate": 0.12,
        "n_estimators": 500,
        "min_child_samples": 30,
        "subsample": 0.6,
        "subsample_freq": 5,
        "colsample_bytree": 0.4,
        "reg_alpha": 0,
        "reg_lambda": 0,
        "random_state": 4242,
        "min_child_weight": 0.28,
        "min_split_gain": 9.793058539831146e-08,
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
    for i in range(6):
        auc = roc_auc_score(y_true == i, preds[:, i])
        aucs.append(auc)
    score = np.mean(auc)
    return "mean_auc", score, True


def drop_null_targets(
    X_raw: pd.DataFrame, y_raw: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop all rows where target is missing

    Args:
        X_raw (pd.DataFrame): features
        y_raw (pd.Series): targets

    Returns:
        Tuple[pd.DataFrame, pd.Series]: features and targets without rows with missing data
    """
    X = X_raw.reset_index(drop=True)
    y = y_raw.reset_index(drop=True)
    ind = y.notnull()
    return X.loc[ind], y[ind]


def train_model(
    X: pd.DataFrame, Y1: pd.Series, Y3: pd.Series
) -> Tuple[List[LGBMClassifier], List[LGBMClassifier]]:
    """
    Train models for t and t+3 predictions

    Args:
        X (pd.DataFrame): input features
        Y1 (pd.Series): t targets
        Y3 (pd.Series): t+3 targets

    Returns:
        List[LGBMClassifier]: list of trained models
    """
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


def train_models(
    X: pd.Dataframe,
    Y1: pd.Series,
    Y3: pd.Series,
    META: pd.Series,
    params_to_train: Optional[List[Dict]] = None,
) -> Tuple[List[LGBMClassifier], List[LGBMClassifier], float]:
    """
    Train models

    Args:
        X (pd.DataFrame): input features
        Y1 (pd.Series): t targets
        Y3 (pd.Series): t+3 targets
        META (pd.Series): metadata
        params_to_train (Optional[List[Dict]], optional): list of dicts of hyperparams to use for training. Defaults to None.

    Returns:
        Tuple[List[LGBMClassifier], List[LGBMClassifier], float]: t and t+3 trained models, mean validation score
    """
    if params_to_train is None:
        params_to_train = PARAMS

    # use repeated statified group kfold for training
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            X, Y1, META
        )
    )
    folds += list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100).split(
            X, Y1, META
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
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_1_train, y_1_test = Y1[train_index], Y1[test_index]
            y_3_train, y_3_test = Y3[train_index], Y3[test_index]
            # meta_train, meta_val = META[train_index], META[test_index]

            # drop null targets
            X_train_1, y_train_1 = drop_null_targets(X_train, y_1_train)
            _, _ = drop_null_targets(X_train, y_3_train)
            X_val_1, y_val_1 = drop_null_targets(X_test, y_1_test)
            _, _ = drop_null_targets(X_test, y_3_test)

            try:
                # train models
                model_1 = LGBMClassifier(**params)
                model_1.fit(
                    X_train_1,
                    y_train_1,
                    eval_set=(X_val_1, y_val_1),
                    verbose=100,
                    early_stopping_rounds=50,
                )
                auc_1 = get_auc(
                    y_true=y_val_1, probas_pred=model_1.predict_proba(X_val_1)
                )[1]
                scores_1.append(auc_1)
                models_1.append(model_1)

                models_3 = models_1
                auc_3 = auc_1

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
    X, Y1, Y3, _ = processor.generate_features_train(data)
    models_1, models_3 = train_model(X, Y1, Y3)

    main_model = Model(
        features=list(X.columns),
        preprocessor=processor,
        models_1=models_1,
        models_3=models_3,
    )
    with open(output_file, "wb") as f:
        pickle.dump(main_model, f, pickle.HIGHEST_PROTOCOL)

    print("Training finished.")
    return 0


if __name__ == "__main__":
    main()
