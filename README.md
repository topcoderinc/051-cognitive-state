# NASA Cognitive State Determination - Fault Tolerance Marathon Match
## STEPS to Build & Run

- Step 1 - cd code & build docker `docker build -t latest .` 
- Step 2 - run model training 
`docker run -v /path_to_data/:/work/data/ -v /path_to_save_model/:/work/model/ latest sh train.sh /work/data/data_training.zip`
- Step 3 - run model prediction
`docker run -v /path_to_data/:/data/ -v /path_to_model/:/work/model/ latest sh test.sh /data/test_data_file.zip /data/file_to_save_predictions.csv`
