"""
This module: saving constant values
"""

TARGET2LABEL = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "baseline": 3,
    "channelized": 4,
    "surprise": 5,
}
LABEL2TARGET = dict(zip(TARGET2LABEL.values(), TARGET2LABEL.keys()))
METADATA_COLUMNS = ["test_suite"]
NAN_VALUES = [-9999.9]
DATA_COLS = [
        # features
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
