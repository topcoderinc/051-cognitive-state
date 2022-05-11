"""
This module: model functions
"""
# pylint: disable-msg=too-many-locals

from typing import List, Tuple
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from constants import DATA_COLS
from processing import FeaturesGenerator


ALLOWED_IMPORTANT_FEATURES = set(DATA_COLS)


class Model:
    """
        This is main model class
    """
    def __init__(
        self,
        features: List[str],
        preprocessor: FeaturesGenerator,
        models_1: List[LGBMClassifier],
        models_3: List[LGBMClassifier],
    ):
        """
        Model

        Args:
            features (List[str]): list of features which were used during training
            preprocessor (FeaturesGenerator): preprocessor
            models_1 (List[LGBMClassifier]): list model models for t predictions
            models_3 (List[LGBMClassifier]): list model models for t+3 predictions
        """
        self.features = features
        self.preprocessor = preprocessor
        self.models_1 = models_1
        self.models_3 = models_3
        self.num_classes = 6

    def ensure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe into the same format which was used during training.
        I.e., ensure that all features are present. If a feature is missing - replace it with 0.
        Ensure order of features in the dataframe.

        Args:
            data (pd.DataFrame): input data

        Returns:
            pd.DataFrame: processed data
        """
        for col in self.features:
            if col not in data.columns:
                data[col] = 0
        data = data[self.features]
        return data

    def predict(
        self, x_raw: pd.DataFrame
    ) -> Tuple[np.array, np.array, np.array, np.array, List[List[str]]]:
        """
        Make predictions for t and t+3, find the most important features

        Args:
            x_raw (pd.DataFrame): raw input data

        Returns:
            Tuple[np.array, np.array, np.array, np.array, List[List[str]]]:
            predictions and most important features
        """
        # generate features
        data, _, _, group = self.preprocessor.generate_features(x_raw, get_targers=False)

        # make predictions
        y_hat_1, most_important_features = make_predictions(data, group, self.models_1)
        y_hat_3 = y_hat_1

        # transform predictions into labels
        y_hat_1_label = self.preprocessor.apply_label2target(np.argmax(y_hat_1, axis=1))
        y_hat_3_label = y_hat_1_label
        return y_hat_1, y_hat_3, y_hat_1_label, y_hat_3_label, most_important_features


def sum_arrays(arrays: List[np.array]) -> np.array:
    """
    Calculate sum of list of arrays
    """
    data = arrays[0]
    if len(arrays) == 1:
        return data
    for _data in arrays[1:]:
        data += _data
    return data


def mean_arrays(arrays) -> np.array:
    """
    Calculate the mean of the list of arrays
    """
    return sum_arrays(arrays) / len(arrays)


def postprocess_preds(predicts: np.array, group: pd.Series) -> pd.DataFrame:
    """
    Smooth predictions by running rolling mean within the group

    Args:
        predicts (np.array): array of predicted probs
        group (pd.Series): group id

    Returns:
        pd.DataFrame: smoothed predicted probs
    """
    data = pd.DataFrame(predicts)
    data["group"] = group
    result = data.groupby("group").rolling(window=999_999, min_periods=1).mean().values
    return result


def make_predictions(
    data: pd.DataFrame, group: pd.Series, models: List[LGBMClassifier]
) -> Tuple[np.array, List[List[str]]]:
    """
    Make predictions and calculate the most important features

    Args:
        data (pd.DataFrame): input raw data
        group (pd.Series): group id for each row in input data
        models (List[Model]): list of models. Final predictions is
            avg prediction of all models predictions.

    Returns:
        Tuple[np.array, List[List[str]]]: predictions and most important features
    """
    y_hat = []
    shap_values = []
    for model in models:
        # make predictions and calculate SHAP importance
        y_hat_pred = model.predict_proba(data)
        contribs = model.predict_proba(data, pred_contrib=True)

        # contribs - list of n_samples * (n_features+1)*n_classes
        # drop shap sum column: (n_samples, (n_features + 1) * n_classes) ->
        # n_samples, n_features * n_classes
        y_shap_pred = np.vstack(contribs)
        n_features = data.shape[1]
        indexes = [
            i
            for i in range(y_shap_pred.shape[1])
            if (i // n_features > 0) and (i % n_features == 1)
        ]
        y_shap_pred = y_shap_pred[:, indexes]

        # smooth predictions and shap importance using group id
        y_hat_pred = postprocess_preds(y_hat_pred, group)
        y_shap_pred = postprocess_preds(y_shap_pred, group)

        y_hat.append(y_hat_pred)
        shap_values.append(y_shap_pred)

    # calculate mean for predictions and SHAP importance
    y_hat = mean_arrays(y_hat)
    shap_values = mean_arrays(shap_values)

    # select most important featuers from list of allowed features
    most_important_features = []
    topn_featuers = 3
    for i in range(shap_values.shape[0]):
        row = shap_values[i, :]
        ind_argsorted = np.argsort(row)[::-1]  # high -> low
        tmp = []
        for j in ind_argsorted:
            feature = list(data.columns)[j % n_features]
            for raw_feature in ALLOWED_IMPORTANT_FEATURES:
                if (raw_feature in feature) and (raw_feature not in tmp):
                    tmp.append(raw_feature)
        tmp = tmp[:topn_featuers]
        most_important_features.append(tmp)
    return y_hat, most_important_features
