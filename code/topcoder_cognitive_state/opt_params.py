"""
This module:  fine tune hyper params.
"""

import os
import sys
import warnings
import copy
import logging
import optuna
from load_data import read_data
from processing import FeaturesGenerator
from train import train_models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")


def main():
    """
        hyperparams fine tune using optuna
    """
    if len(sys.argv[1]) == 0 or len(sys.argv) < 2:
        print("Input file of opt params is missing.")
        return 1

    if len(sys.argv[2]) == 0 or len(sys.argv) < 3:
        print("log path is missing")
        return 1

    path_to_log = sys.argv[2]
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(path_to_log, mode="w"), logging.StreamHandler()],
    )

    input_file = sys.argv[1]
    _ = sys.argv[2]

    data, _ = read_data(input_file)
    processor = FeaturesGenerator()
    train_x, train_y1, train_y3, meta = processor.generate_features_train(data)

    default_params = {
        "num_leaves": 127,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "min_child_samples": 7,
        "subsample": 0.67,
        "subsample_freq": 5,
        "colsample_bytree": 0.67,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "device": "gpu",
        "gpu_platform_id": 1,
        "gpu_device_id": 0
    }

    def objective(trial):
        params = copy.deepcopy(default_params)
        params.update(
            {
                "num_leaves": trial.suggest_int("num_leaves", 7, 255, 4),
                "max_depth": trial.suggest_int("max_depth", 3, 14, 1),
                "min_child_weight": trial.suggest_loguniform(
                    "min_child_weight", 1e-3, 1
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 1, 100, 1, log=True
                ),
                "min_split_gain": trial.suggest_loguniform("min_split_gain", 1e-3, 1),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
            }
        )
        _, _, test_score = train_models(train_x, train_y1, train_y3, meta, params_to_train=[params])

        return test_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    logging.info("Number of finished trials: %s", len(study.trials))
    logging.info("Best trial:")
    best_trial = study.best_trial
    logging.info("  Value: %s", best_trial.value)
    logging.info("  Params: ")
    for key, value in best_trial.params.items():
        logging.info("    %s: %s", key, value)


if __name__ == "__main__":
    main()
