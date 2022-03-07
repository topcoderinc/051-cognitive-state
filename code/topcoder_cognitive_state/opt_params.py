import sys
import warnings
import copy
import logging
import optuna

from topcoder_cognitive_state.load_data import read_data
from topcoder_cognitive_state.processing import FeaturesGenerator
from topcoder_cognitive_state.model import Model  # noqa: F401
from topcoder_cognitive_state.train import train_models


warnings.filterwarnings("ignore")


def main():
    """
    Optimize model's hyperparams using optuna
    """
    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Training input file is missing.")
        return 1

    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Path to log is missing")
        return 1

    path_to_log = sys.argv[2]
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(path_to_log, mode="w"), logging.StreamHandler()],
    )

    print("Training started.")

    input_file = sys.argv[1]
    _ = sys.argv[2]

    data, _ = read_data(input_file)
    processor = FeaturesGenerator()
    X, Y1, Y3, META = processor.generate_features_train(data)

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
    }

    def objective(trial):
        params = copy.deepcopy(default_params)
        params.update(
            {
                "num_leaves": trial.suggest_int("num_leaves", 7, 255, 8),
                "max_depth": trial.suggest_int("max_depth", 3, 14, 1),
                "min_child_weight": trial.suggest_loguniform(
                    "min_child_weight", 1e-18, 1
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 1, 100, 1, log=True
                ),
                "min_split_gain": trial.suggest_loguniform("min_split_gain", 1e-18, 1),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10),
            }
        )
        _, _, test_score = train_models(X, Y1, Y3, META, params_to_train=[params])

        logging.info(f"Next itter score - {test_score}")
        for k, v in params.items():
            logging.info(f"'{k}': {v},")
        logging.info("")
        return test_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    logging.info("Number of finished trials: {}".format(len(study.trials)))
    logging.info("Best trial:")
    best_trial = study.best_trial
    logging.info("  Value: {}".format(best_trial.value))
    logging.info("  Params: ")
    for key, value in best_trial.params.items():
        logging.info("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
