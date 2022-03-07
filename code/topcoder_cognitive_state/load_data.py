from typing import Tuple
import time

import pandas as pd
from tqdm import tqdm

from topcoder_cognitive_state.CONSTANTS import METADATA_COLUMNS, NAN_VALUES


def _test_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    This test contains three tests which are run manually:

        1. Check if some columns are missing
        2. Check if some columns have None values
        3. Check if some columns have -9999.9 (missing) values
    """
    # cols = ['ViveEye_gazeOrigin_L_X', 'ViveEye_gazeOrigin_L_Y', 'ViveEye_gazeOrigin_L_Z']
    # cols = ['Myo_EMG_0', 'Myo_EMG_1', 'Myo_EMG_2', 'Myo_EMG_3', 'Myo_EMG_4', 'Myo_EMG_5', 'Myo_EMG_6']
    # cols = ['Polar_bpm', 'Polar_hrv', 'tlx_score']
    cols = [
        # features
        "tlx_score",
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
        "ViveEye_pupilPos_L_X",
        "ViveEye_pupilPos_L_Y",
        "ViveEye_pupilPos_R_X",
        "ViveEye_pupilPos_R_Y",
        "ViveEye_gazeOrigin_L_X",
        "ViveEye_gazeOrigin_L_Y",
        "ViveEye_gazeOrigin_L_Z",
        "ViveEye_gazeOrigin_R_X",
        "ViveEye_gazeOrigin_R_Y",
        "ViveEye_gazeOrigin_R_Z",
        "ViveEye_gazeDirection_L_X",
        "ViveEye_gazeDirection_L_Y",
        "ViveEye_gazeDirection_L_Z",
        "ViveEye_gazeDirection_R_X",
        "ViveEye_gazeDirection_R_Y",
        "ViveEye_gazeDirection_R_Z",
        "ViveEye_eyeOpenness_L",
        "ViveEye_pupilDiameter_L",
        "ViveEye_eyeOpenness_R",
        "ViveEye_pupilDiameter_R",
        "Zephyr_HR",
        "Zephyr_HRV",
    ]

    # case 1 - no column
    # df = df.drop(cols, axis=1)

    # case 2 - None values
    # for col in cols:
    #    df[col] = None

    # case 3 - missing values
    for col in cols:
        df[col] = -9999.9
    return df


def read_and_prepare_data_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Read raw data and prepare it for processing.
    I.e., create columns if they are missing,
    replace missing values with None,
    etc.

    Args:
        df (pd.DataFrame): input raw data

    Returns:
        pd.DataFrame: processed data
    """
    EXPECTED_COLUMNS = [
        # features
        "tlx_score",
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
        "ViveEye_pupilPos_L_X",
        "ViveEye_pupilPos_L_Y",
        "ViveEye_pupilPos_R_X",
        "ViveEye_pupilPos_R_Y",
        "ViveEye_gazeOrigin_L_X",
        "ViveEye_gazeOrigin_L_Y",
        "ViveEye_gazeOrigin_L_Z",
        "ViveEye_gazeOrigin_R_X",
        "ViveEye_gazeOrigin_R_Y",
        "ViveEye_gazeOrigin_R_Z",
        "ViveEye_gazeDirection_L_X",
        "ViveEye_gazeDirection_L_Y",
        "ViveEye_gazeDirection_L_Z",
        "ViveEye_gazeDirection_R_X",
        "ViveEye_gazeDirection_R_Y",
        "ViveEye_gazeDirection_R_Z",
        "ViveEye_eyeOpenness_L",
        "ViveEye_pupilDiameter_L",
        "ViveEye_eyeOpenness_R",
        "ViveEye_pupilDiameter_R",
        "Zephyr_HR",
        "Zephyr_HRV",
        # target
        "induced_state",
    ]

    # uncomment to enable test
    # df = _test_missing_features(df)

    # test_suite
    if "test_suite" not in df.columns:
        df["test_suite"] = "test"

    df["time"] = pd.to_datetime(df["time"], unit="us")
    df["timestamp"] = df["time"].dt.round("1s")
    df = df.drop("time", axis=1)

    ags = df.groupby(METADATA_COLUMNS + ["timestamp"]).first()

    fillna_constant = 0

    # replace missing features with zero
    missing_features = [col for col in EXPECTED_COLUMNS if col not in ags.columns]
    for col in missing_features:
        ags[col] = fillna_constant

    # fill none values with 0
    ags = ags.fillna(fillna_constant)

    # fill -999 values with 0
    for v in NAN_VALUES:
        ags = ags.replace(v, fillna_constant)

    # keep columns in the same order
    ags = ags[EXPECTED_COLUMNS]
    return ags


def get_dummy_template(df: pd.DataFrame) -> pd.DataFrame:
    """
    The template is needed to match the expected sample submission format.
    """
    df["time"] = pd.to_datetime(df["time"], unit="us")
    df["timestamp"] = df["time"].dt.round("1s")
    df = df.drop("time", axis=1)
    dummy_template = df.drop_duplicates(
        subset=METADATA_COLUMNS + ["timestamp"], keep="first"
    ).reset_index(drop=True)
    dummy_template = dummy_template[METADATA_COLUMNS + ["timestamp"]]
    return dummy_template


def get_needed_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read data for training/testing and prepare template format for submission

    Return:
        res1 - pd.DataFrame - read data
        res2 - pd.DataFrame - template for submission
    """
    res1 = read_and_prepare_data_chunk(df)
    res2 = get_dummy_template(df)
    return res1, res2


def read_data(
    path_to_data: str, debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read data. The data is read in chunks to reduce memory consumption.

    Args:
        path_to_data (str): path to data
        debug (bool, optional): run data loading on a sample of data. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Read data and prepared template for submission
    """
    t_start = time.time()
    chunksize = 10**6

    if path_to_data is None:
        path_to_data = "./data/training-data.zip"

    # use small dataset when debugging
    if debug:
        nrows = 1 * chunksize
    else:
        nrows = None

    # create chunks iterator to read data
    chunks = pd.read_csv(
        path_to_data, na_values=NAN_VALUES, chunksize=chunksize, nrows=nrows
    )

    # get data for processing
    full_result = []
    for chunk in tqdm(chunks):
        tmp = get_needed_data(chunk)
        full_result.append(tmp)
    res = [i[0] for i in full_result]
    res = pd.concat(res, axis=0)
    res = res.sort_index()
    res = res[~res.index.duplicated(keep="first")]

    # collect dummies for sub
    res2 = [i[1] for i in full_result]
    res2 = pd.concat(res2, axis=0)
    res2 = res2.drop_duplicates(
        subset=METADATA_COLUMNS + ["timestamp"], keep="first"
    ).reset_index(drop=True)
    t_end = time.time()
    print(f"Data is read. Time per reading: {(t_end-t_start)/60:.2f} minutes")
    return res, res2
