from typing import List, Tuple

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

from topcoder_cognitive_state.processing import FeaturesGenerator


ALLOWED_IMPORTANT_FEATURES = set(
    [
        "E4_BVP",
        "E4_GSR",
        "LooxidLink_EEG_A3",
        "LooxidLink_EEG_A4",
        "LooxidLink_EEG_FP1",
        "LooxidLink_EEG_FP2",
        "LooxidLink_EEG_A7",
        "LooxidLink_EEG_A8",
        "Muse_EEG_TP9",
        "Muse_EEG_AF7",
        "Muse_EEG_AF8",
        "Muse_EEG_TP10",
        "Muse_PPG_0",
        "Muse_PPG_1",
        "Muse_PPG_2",
        "Myo_GYR_X",
        "Myo_GYR_Y",
        "Myo_GYR_Z",
        "Myo_EMG_0",
        "Myo_EMG_1",
        "Myo_EMG_2",
        "Myo_EMG_3",
        "Myo_EMG_4",
        "Myo_EMG_5",
        "Myo_EMG_6",
        "Myo_EMG_7",
        "PICARD_fnirs_0",
        "PICARD_fnirs_1",
        "Polar_bpm",
        "Polar_hrv",
        "ViveEye_eyeOpenness_L",
        "ViveEye_pupilDiameter_L",
        "ViveEye_pupilPos_L_X",
        "ViveEye_pupilPos_L_Y",
        "ViveEye_gazeOrigin_L_X",
        "ViveEye_gazeOrigin_L_Y",
        "ViveEye_gazeOrigin_L_Z",
        "ViveEye_gazeDirection_L_X",
        "ViveEye_gazeDirection_L_Y",
        "ViveEye_gazeDirection_L_Z",
        "ViveEye_eyeOpenness_R",
        "ViveEye_pupilDiameter_R",
        "ViveEye_pupilPos_R_X",
        "ViveEye_pupilPos_R_Y",
        "ViveEye_gazeOrigin_R_X",
        "ViveEye_gazeOrigin_R_Y",
        "ViveEye_gazeOrigin_R_Z",
        "ViveEye_gazeDirection_R_X",
        "ViveEye_gazeDirection_R_Y",
        "ViveEye_gazeDirection_R_Z",
        "Zephyr_HR",
        "Zephyr_HRV",
    ]
)


class Model:
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

    def ensure_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe into the same format which was used during training.
        I.e., ensure that all features are present. If a feature is missing - replace it with 0.
        Ensure order of features in the dataframe.

        Args:
            x (pd.DataFrame): input data

        Returns:
            pd.DataFrame: processed data
        """
        for col in self.features:
            if col not in x.columns:
                x[col] = 0
        x = x[self.features]
        return x

    def predict(
        self, x_raw: pd.DataFrame
    ) -> Tuple[np.array, np.array, np.array, np.array, List[List[str]]]:
        """
        Make predictions for t and t+3, find the most important features

        Args:
            x_raw (pd.DataFrame): raw input data

        Returns:
            Tuple[np.array, np.array, np.array, np.array, List[List[str]]]: predictions and most important features
        """
        # generate features
        x, _, _, group = self.preprocessor.generate_features(x_raw, get_targers=False)

        # make predictions
        y_hat_1, most_important_features = make_predictions(x, group, self.models_1)
        y_hat_3 = y_hat_1

        # transform predictions into labels
        y_hat_1_label = self.preprocessor.apply_label2target(np.argmax(y_hat_1, axis=1))
        y_hat_3_label = y_hat_1_label
        return y_hat_1, y_hat_3, y_hat_1_label, y_hat_3_label, most_important_features


def sum_arrays(arrs: List[np.array]) -> np.array:
    """
    Calculate sum of list of arrays
    """
    x = arrs[0]
    if len(arrs) == 1:
        return x
    for x2 in arrs[1:]:
        x += x2
    return x


def mean_arrays(arrs) -> np.array:
    """
    Calculate the mean of the list of arrays
    """
    return sum_arrays(arrs) / len(arrs)


def postprocess_preds(preds: np.array, group: pd.Series) -> pd.DataFrame:
    """
    Smooth predictions by running rolling mean within the group

    Args:
        preds (np.array): array of predicted probs
        group (pd.Series): group id

    Returns:
        pd.DataFrame: smoothed predicted probs
    """
    df = pd.DataFrame(preds)
    df["group"] = group
    res = df.groupby("group").rolling(window=999_999, min_periods=1).mean().values
    return res


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
        # drop shap sum column: (n_samples, (n_features + 1) * n_classes) -> n_samples, n_features * n_classes
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
