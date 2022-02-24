TARGET2LABEL = {
    "low": 0, 
    "medium": 1, 
    "high": 2, 
    "baseline": 3, 
    "channelized": 4, 
    "surprise": 5 
}
LABEL2TARGET = dict(zip(TARGET2LABEL.values(), TARGET2LABEL.keys()))
        

METADATA_COLUMNS = ["test_suite"]
NAN_VALUES = [-9999.9]