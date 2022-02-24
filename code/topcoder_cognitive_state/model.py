from typing import List, Tuple

from functools import reduce

import pandas as pd
import numpy as np
from scipy import stats


ALLOWED_IMPORTANT_FEATURES = set(['E4_BVP', 'E4_GSR', 'LooxidLink_EEG_A3', 'LooxidLink_EEG_A4', 'LooxidLink_EEG_FP1', 'LooxidLink_EEG_FP2',
            'LooxidLink_EEG_A7', 'LooxidLink_EEG_A8', 'Muse_EEG_TP9', 'Muse_EEG_AF7', 'Muse_EEG_AF8', 'Muse_EEG_TP10',
            'Muse_PPG_0', 'Muse_PPG_1', 'Muse_PPG_2', 'Myo_GYR_X', 'Myo_GYR_Y', 'Myo_GYR_Z', 'Myo_EMG_0', 'Myo_EMG_1',
            'Myo_EMG_2', 'Myo_EMG_3', 'Myo_EMG_4', 'Myo_EMG_5', 'Myo_EMG_6', 'Myo_EMG_7', 'PICARD_fnirs_0', 'PICARD_fnirs_1',
            'Polar_bpm', 'Polar_hrv', 'ViveEye_eyeOpenness_L', 'ViveEye_pupilDiameter_L', 'ViveEye_pupilPos_L_X',
            'ViveEye_pupilPos_L_Y', 'ViveEye_gazeOrigin_L_X', 'ViveEye_gazeOrigin_L_Y', 'ViveEye_gazeOrigin_L_Z',
            'ViveEye_gazeDirection_L_X', 'ViveEye_gazeDirection_L_Y', 'ViveEye_gazeDirection_L_Z', 'ViveEye_eyeOpenness_R',
            'ViveEye_pupilDiameter_R', 'ViveEye_pupilPos_R_X', 'ViveEye_pupilPos_R_Y', 'ViveEye_gazeOrigin_R_X',
            'ViveEye_gazeOrigin_R_Y', 'ViveEye_gazeOrigin_R_Z', 'ViveEye_gazeDirection_R_X', 'ViveEye_gazeDirection_R_Y',
            'ViveEye_gazeDirection_R_Z', 'Zephyr_HR', 'Zephyr_HRV'])


def sum_arrays(arrs):
    x = arrs[0]
    if len(arrs) == 1:
        return x
    for x2 in arrs[1:]:
        x += x2
    return x


def mean_arrays(arrs):
    return sum_arrays(arrs) / len(arrs)


def postprocess_preds(preds, group):
    df = pd.DataFrame(preds)
    df['group'] = group
    res = df.groupby('group').rolling(window=999_999, min_periods=1).mean().values
    return res


def make_predictions(data: pd.DataFrame, group: pd.Series, models) -> Tuple[np.array, List[List[str]]]:
    y_hat = []
    shap_values = []
    for model in models:
        y_hat_pred = model.predict_proba(data)
        contribs = model.predict_proba(data, pred_contrib=True)

        # contribs - list of n_samples * (n_features+1)*n_classes
        # drop shap sum column: (n_samples, (n_features + 1) * n_classes) -> n_samples, n_features * n_classes
        y_shap_pred = np.vstack(contribs)
        n_features = data.shape[1]
        indexes = [i for i in range(y_shap_pred.shape[1]) if (i // n_features > 0) and (i % n_features == 1)]
        y_shap_pred = y_shap_pred[:, indexes]

        y_hat_pred = postprocess_preds(y_hat_pred, group)
        y_shap_pred = postprocess_preds(y_shap_pred, group)

        y_hat.append(y_hat_pred)
        shap_values.append(y_shap_pred)
    
    y_hat = mean_arrays(y_hat)
    shap_values = mean_arrays(shap_values)

    # select most important featuers using shap values
    most_important_features = []
    topn_featuers = 3
    for i in range(shap_values.shape[0]):
        row = shap_values[i, :]
        ind_argsorted = np.argsort(row)[::-1] # high -> low
        tmp = []
        for j in ind_argsorted:
            feature = list(data.columns)[j % n_features]
            for raw_feature in ALLOWED_IMPORTANT_FEATURES:
                if (raw_feature in feature) and (raw_feature not in tmp):
                    tmp.append(raw_feature)
        tmp = tmp[:topn_featuers]
        most_important_features.append(tmp)
    return y_hat, most_important_features
    

class Model:
    def __init__(self, features: List[str], preprocessor, models_1, models_3):
        self.features = features
        self.preprocessor = preprocessor
        self.models_1 = models_1
        self.models_3 = models_3
        self.num_classes = 6

    def ensure_features(self, x):
        for col in self.features:
            if col not in x.columns:
                x[col] = 0
        x = x[self.features]
        return x
    
    def predict(self, x_raw):
        x, _, _, group = self.preprocessor.generate_featres(x_raw, get_targers=False)

        # get probs
        y_hat_1, most_important_features = make_predictions(x, group, self.models_1)
        y_hat_3 = y_hat_1

        # get labels
        y_hat_1_label = self.preprocessor.apply_label2target(np.argmax(y_hat_1, axis=1))
        y_hat_3_label = y_hat_1_label
        return y_hat_1, y_hat_3, y_hat_1_label, y_hat_3_label, most_important_features