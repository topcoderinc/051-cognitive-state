"""
This module:  processing data and get features.
"""
# pylint: disable-msg=too-many-locals

from typing import Tuple
import re
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from constants import TARGET2LABEL, LABEL2TARGET

warnings.filterwarnings("ignore")


class FeaturesGenerator:
    """
        FeaturesGenerator is used for feature generating
    """
    def __init__(self, target_column: str = "induced_state"):
        """
        Generate features using raw data

        Args:
            target_column (str, optional): target column. Defaults to "induced_state".
        """
        self.target_column = target_column
        self.target2label = TARGET2LABEL
        self.label2target = LABEL2TARGET

    @classmethod
    def rename_cols(cls, data: pd.DataFrame):
        """
        Lightgbm doesn't work well with all features names, so we need to rename some of them."""
        data = data.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
        return data

    def get_targets(self, data: pd.DataFrame, future: int = 1) -> pd.Series:
        """
        Create training targets

        Args:
            data (pd.DataFrame): input data
            future (int, optional): steps to look into future.
                Future=1 equals the current moment (t).
                Future=2 equals to the next moment (t+1).
                Defaults to 1.

        Returns:
            pd.Series: target
        """
        if future == 0:
            target = data[self.target_column]
        else:
            target = (
                data.groupby("session_id")[self.target_column]
                .shift(-1 * future)
                .fillna(method="ffill")  # replace missing targets with previous value
            )
            ind = target.isnull()
            target[ind] = data.loc[ind, self.target_column]
        target_label = self.apply_target2label(target)
        return target_label

    def apply_target2label(self, target: pd.Series) -> pd.Series:
        """
            target to label
        Args:
            target:
        Returns:    label results
        """
        # 'low' -> 0, 'medium' -> 1, etc
        return pd.Series(target).map(self.target2label)

    def apply_label2target(self, target: pd.Series) -> pd.Series:
        """
            label to target
        Args:
            target:
        Returns:    target results
        """
        #  -> 'low', 1 -> 'medium', etc
        return pd.Series(target).map(self.label2target)

    @classmethod
    def get_session_id_and_time_since_break(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate session id and time since the last break.
        """
        tmp = data.copy().reset_index()
        diff = tmp["timestamp"].diff().dt.total_seconds()
        tmp["session_id"] = np.cumsum(diff > 1)
        data["session_id"] = tmp["session_id"].values
        return data

    def generate_features(
        self, data: pd.DataFrame, get_targers: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Process raw data and create new features for training

        Args:
            data (pd.DataFrame): raw input data
            get_targers (bool, optional): generate targets? Needed for training. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]: output
        """
        data = self.get_session_id_and_time_since_break(data)
        data = data.reset_index(drop=True)

        # create targets for t and t+3 models
        if get_targers:
            target1 = self.get_targets(data, future=0)
            target3 = self.get_targets(data, future=3)
        else:
            target1 = None
            target3 = None
        if self.target_column in data.columns:
            data = data.drop(self.target_column, axis=1)

        # additional featurers
        for feature1, feature2 in [
            ("Zephyr_HR", "Zephyr_HRV"),
            ("Polar_bpm", "Polar_hrv"),
            ("Zephyr_HRV", "Polar_hrv"),
        ]:
            data[f"{feature1}_div_{feature2}"] = data[feature1] / data[feature2]

        datas = [data]

        # create rolling stats features
        windows = [5, 10, 20, 50, 100, 500, 1000]
        cols = list(data.columns)
        for window in windows:
            rolling_mean = (
                data.groupby("session_id")[cols]
                .rolling(min_periods=1, window=window)
                .mean()
                .reset_index(drop=True)
            )
            rolling_max = (
                data.groupby("session_id")[cols]
                .rolling(min_periods=1, window=window)
                .max()
                .reset_index(drop=True)
            )
            rolling_median = (
                data.groupby("session_id")[cols]
                .rolling(min_periods=1, window=window)
                .median()
                .reset_index(drop=True)
            )
            rolling_std = (
                data.groupby("session_id")[cols]
                .rolling(min_periods=1, window=window)
                .std()
                .reset_index(drop=True)
            )
            normed = (data - rolling_mean) / (rolling_std + 1)

            normed = normed.add_prefix(f"normed_by_session_{window}_")
            rolling_mean = rolling_mean.add_prefix(f"mean_by_session_{window}_")
            rolling_median = rolling_median.add_prefix(f"median_by_session_{window}_")
            rolling_max = rolling_max.add_prefix(f"max_by_session_{window}_")
            rolling_std = rolling_std.add_prefix(f"std_by_session_{window}_")

            datas += [rolling_max, rolling_mean, rolling_median, rolling_std, normed]

        # create global stats features
        windows = [999_999]
        global_cols = cols
        for window in windows:
            rolling_mean = (
                data[global_cols]
                .rolling(min_periods=2, window=window)
                .mean()
                .reset_index(drop=True)
            )
            rolling_max = (
                data[global_cols]
                .rolling(min_periods=2, window=window)
                .max()
                .reset_index(drop=True)
            )
            rolling_median = (
                data[global_cols]
                .rolling(min_periods=2, window=window)
                .median()
                .reset_index(drop=True)
            )
            rolling_std = (
                data[global_cols]
                .rolling(min_periods=2, window=window)
                .std()
                .reset_index(drop=True)
            )
            normed = (data - rolling_mean) / (rolling_std + 1)

            normed = normed.add_prefix(f"normed_global_{window}_")
            rolling_mean = rolling_mean.add_prefix(f"mean_global_{window}_")
            rolling_median = rolling_median.add_prefix(f"median_global_{window}_")
            rolling_max = rolling_max.add_prefix(f"max_global_{window}_")
            rolling_std = rolling_std.add_prefix(f"std_global_{window}_")

            datas += [rolling_max, rolling_mean, rolling_median, rolling_std, normed]

        # shift features
        for shift in [1, 3, 5]:
            gr_s = data.groupby("session_id")[cols].shift(shift).reset_index(drop=True)
            tmp = data - gr_s
            tmp = tmp.add_prefix(f"shift_{shift}_")
            datas += [tmp]

        datas = pd.concat(datas, axis=1)
        # df = pd.concat([x, self.calc_eyes_distances(df)], axis=1)

        session_id = datas["session_id"].copy()

        datas = datas.drop("session_id", axis=1)
        datas = self.rename_cols(datas)
        return datas, target1, target3, session_id

    def generate_features_train(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Generate features and targets for training

        Args:
            data (pd.DataFrame): raw input data
        """
        # last index is time
        indexes = [i[:-1] for i in data.index]
        indexes = list(set(indexes))

        # for each comb - generate features
        feature, target1, target3, meta = [], [], [], []
        for index in tqdm(indexes):
            index_data_to_select = index  # use full index
            data_index = data.loc[index_data_to_select]

            _feature, _target1, _target3, session_id = self.generate_features(data_index,
                                                                              get_targers=True)

            feature.append(_feature)
            target1.append(_target1)
            target3.append(_target3)

            # group id = person + task
            task = pd.Series([index] * _feature.shape[0])
            _meta = task.astype("str") + "__" + session_id.astype("str")
            meta.append(_meta)

        feature = pd.concat(feature, axis=0).reset_index(drop=True)
        target1 = pd.concat(target1, axis=0).reset_index(drop=True)
        target3 = pd.concat(target3, axis=0).reset_index(drop=True)
        meta = pd.concat(meta, axis=0).reset_index(drop=True)
        return feature, target1, target3, meta
