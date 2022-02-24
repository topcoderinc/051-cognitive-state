import re

import pandas as pd 
import numpy as np
from tqdm import tqdm

from topcoder_cognitive_state.CONSTANTS import TARGET2LABEL, LABEL2TARGET


def get_distance(x1, x2):
    delta = 0
    for a1, a2 in zip(x1,x2):
        delta += (a1 - a2) ** 2
    delta = delta ** (1 / len(x1))
    return delta


class FeaturesGenerator:
    def __init__(self, target_column="induced_state"):
        self.target_column = target_column
        
        self.target2label = TARGET2LABEL
        self.label2target = LABEL2TARGET

    def rename_cols(self, x):
        x = x.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        return x

    def get_targets(self, x, future: int = 1):
        if future == 0:
            y = x[self.target_column]
        else:
            y = x.groupby("session_id")[self.target_column].shift(-1 * future).fillna(method='ffill')
            ind = y.isnull()
            y[ind] = x.loc[ind, self.target_column]
        y_label = self.apply_target2label(y)
        return y_label
    
    def apply_target2label(self, y):
        return pd.Series(y).map(self.target2label)
    
    def apply_label2target(self, y):
        return pd.Series(y).map(self.label2target)

    def calc_eyes_distances(self, x):
        new_features = pd.DataFrame({})
        shifts = [1, 3]
        eyes = ["L", "R"]

        # L / R eyes
        new_features["ViveEye_pupilPos_LR_distance"] = get_distance(
                (x["ViveEye_pupilPos_L_X"], x["ViveEye_pupilPos_L_Y"]), 
                (x["ViveEye_pupilPos_R_X"], x["ViveEye_pupilPos_R_Y"])
            )

        # distances
        for s in shifts:
            for pos in eyes:
                new_features[f"ViveEye_pupilPos_distance_{s}_{pos}"] = get_distance(
                    (
                        x[f"ViveEye_pupilPos_{pos}_X"], 
                        x[f"ViveEye_pupilPos_{pos}_Y"]
                    ), 
                    (
                        x.groupby("session_id")[f"ViveEye_pupilPos_{pos}_X"].shift(s).values, 
                        x.groupby("session_id")[f"ViveEye_pupilPos_{pos}_Y"].shift(s).values
                    )
                )
    
        # L / R eyes
        for feature in ["ViveEye_gazeOrigin", "ViveEye_gazeDirection"]:
            new_features["{feature}_LR_distance"] = get_distance(
                (x[f"{feature}_L_X"], x[f"{feature}_L_Y"], x[f"{feature}_L_Z"]), 
                (x[f"{feature}_R_X"], x[f"{feature}_R_Y"], x[f"{feature}_R_Z"])
            )
            for s in shifts:
                for pos in eyes:
                    new_features[f"distance_{feature}_{s}_{pos}"] = get_distance(
                        (
                            x[f"{feature}_{pos}_X"], 
                            x[f"{feature}_{pos}_Y"], 
                            x[f"{feature}_{pos}_Z"]
                        ), 
                        (
                            x.groupby("session_id")[f"{feature}_{pos}_X"].shift(s).values, 
                            x.groupby("session_id")[f"{feature}_{pos}_Y"].shift(s).values, 
                            x.groupby("session_id")[f"{feature}_{pos}_Z"].shift(s).values
                        ), 
                    )
        return new_features

    def get_session_id_and_time_since_break(self, data):
        tmp = data.copy().reset_index()
        diff = tmp["timestamp"].diff().dt.total_seconds()
        tmp["session_id"] = np.cumsum(diff > 1)

        data["session_id"] = tmp["session_id"].values
        return data

    def generate_featres(self, x, get_targers: bool = False):
        x = self.get_session_id_and_time_since_break(x)
        x = x.reset_index(drop=True)

        # 1 and 3 seconds
        if get_targers:
            y1 = self.get_targets(x, future=0)
            y3 = self.get_targets(x, future=3)
        else:
            y1 = None
            y3 = None
        if self.target_column in x.columns:
            x = x.drop(self.target_column, axis=1)

        # additional featurers
        for c1, c2 in [
            ('Zephyr_HR', 'Zephyr_HRV'),
            ('Polar_bpm', 'Polar_hrv'),
            ('Zephyr_HRV', 'Polar_hrv'),
        ]:
            x[f"{c1}_div_{c2}"] = x[c1] / x[c2]

        dfs = [x]

        # rolling stats
        windows = [5, 999_999]
        cols = list(x.columns)
        for w in windows:
            rolling_mean = x.groupby("session_id")[cols].rolling(min_periods=1, window=w).mean().reset_index(drop=True)
            rolling_std = x.groupby("session_id")[cols].rolling(min_periods=1, window=w).std().reset_index(drop=True)
            normed = (x - rolling_mean) / (rolling_std + 1)
            
            normed = normed.add_prefix(f"normed_by_session_{w}_")
            rolling_mean = rolling_mean.add_prefix(f"mean_by_session_{w}_")
            rolling_std = rolling_std.add_prefix(f"std_by_session_{w}_")
            
            dfs += [rolling_mean, rolling_std, normed]

        windows = [999_999]
        global_cols = cols # 
        for w in windows:
            rolling_mean = x[global_cols].rolling(min_periods=2, window=w).mean().reset_index(drop=True)
            rolling_std = x[global_cols].rolling(min_periods=2, window=w).std().reset_index(drop=True)
            normed = (x - rolling_mean) / (rolling_std + 1)
            
            normed = normed.add_prefix(f"normed_global_{w}_")
            rolling_mean = rolling_mean.add_prefix(f"mean_global_{w}_")
            rolling_std = rolling_std.add_prefix(f"std_global_{w}_")
            
            dfs += [rolling_mean, rolling_std, normed]

        # shift features 
        for s in [1, 3]:
            gr_s = x.groupby("session_id")[cols].shift(s).reset_index(drop=True)
            tmp = x - gr_s
            tmp = tmp.add_prefix(f"shift_{s}_")
            dfs += [tmp]

        df = pd.concat(dfs, axis=1)
        df = pd.concat([x, self.calc_eyes_distances(df)], axis=1)

        session_id = df["session_id"].copy()

        df = df.drop("session_id", axis=1)
        df = self.rename_cols(df)
        return df, y1, y3, session_id

    def generate_featres_train(self, data):
        # last index is time
        indexes = [i[:-1] for i in data.index]
        indexes = list(set(indexes))

        # for each comb - generate features
        X, Y1, Y3, META = [], [], [], []
        for index in tqdm(indexes):
            index_data_to_select = index  # use full index
            x = data.loc[index_data_to_select]
            
            r, y1, y3, session_id = self.generate_featres(x, get_targers=True)
            
            X.append(r)
            Y1.append(y1)
            Y3.append(y3)
            
            # group id = person + task 
            task = pd.Series([index] * r.shape[0])
            _meta = task.astype("str") + "__" + session_id.astype("str")
            META.append(_meta)

        X = pd.concat(X, axis=0).reset_index(drop=True)
        Y1 = pd.concat(Y1, axis=0).reset_index(drop=True)
        Y3 = pd.concat(Y3, axis=0).reset_index(drop=True)
        META = pd.concat(META, axis=0).reset_index(drop=True)
        return X, Y1, Y3, META