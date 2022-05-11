"""
This module: loading data
"""
from typing import Tuple
import time
from constants import METADATA_COLUMNS, NAN_VALUES, DATA_COLS
import pandas as pd
from tqdm import tqdm


def _test_missing_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    This test contains three tests which are run manually:

        1. Check if some columns are missing
        2. Check if some columns have None values
        3. Check if some columns have -9999.9 (missing) values
    """
    cols = DATA_COLS

    # case 1 - no column
    # df = df.drop(cols, axis=1)

    # case 2 - None values
    # for col in cols:
    #    df[col] = None

    # case 3 - missing values
    for col in cols:
        data[col] = -9999.9
    return data


def read_and_prepare_data_chunk(data: pd.DataFrame) -> pd.DataFrame:
    """
    Read raw data and prepare it for processing.
    I.e., create columns if they are missing,
    replace missing values with None,
    etc.

    Args:
        data (pd.DataFrame): input raw data

    Returns:
        pd.DataFrame: processed data
    """
    expected_columns = DATA_COLS + [
        "induced_state",
    ]

    # uncomment to enable test
    # df = _test_missing_features(df)

    # test_suite
    if "test_suite" not in data.columns:
        data["test_suite"] = "test"

    data["time"] = pd.to_datetime(data["time"], unit="us")
    data["timestamp"] = data["time"].dt.round("1s")
    data = data.drop("time", axis=1)

    ags = data.groupby(METADATA_COLUMNS + ["timestamp"]).first()

    fillna_constant = -999

    # replace missing features with zero
    missing_features = [col for col in expected_columns if col not in ags.columns]
    for col in missing_features:
        ags[col] = fillna_constant

    # fill none values with -999
    ags = ags.fillna(fillna_constant)

    # fill -999 values with -999
    for value in NAN_VALUES:
        ags = ags.replace(value, fillna_constant)

    # keep columns in the same order
    ags = ags[expected_columns]
    return ags


def get_dummy_template(data: pd.DataFrame) -> pd.DataFrame:
    """
    The template is needed to match the expected sample submission format.
    """
    data["time"] = pd.to_datetime(data["time"], unit="us")
    data["timestamp"] = data["time"].dt.round("1s")
    data = data.drop("time", axis=1)
    dummy_template = data.drop_duplicates(
        subset=METADATA_COLUMNS + ["timestamp"], keep="first"
    ).reset_index(drop=True)
    dummy_template = dummy_template[METADATA_COLUMNS + ["timestamp"]]
    return dummy_template


def get_needed_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read data for training/testing and prepare template format for submission

    Return:
        res1 - pd.DataFrame - read data
        res2 - pd.DataFrame - template for submission
    """
    res1 = read_and_prepare_data_chunk(data)
    res2 = get_dummy_template(data)
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
