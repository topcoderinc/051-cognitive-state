# Whitepaper
## Overview
This codebase is a winner of the Topcoder NASA Cognitive State Determination marathon match. As part of the final submission the competitor was asked to compelte this document. Personal details have been removed. 

1. Introduction
Tell us a bit about yourself, and why you have decided to participate in the contest.
- Handle: tEarth
- Placement you achieved in the MM: 2nd
- About you: Data Scientist at Unity
- Why you participated in the MM: money, fame

2. Solution Development
How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider? Also try to add your cross validation approaches used.
- During EDA, I noticed that the timestamps might have “holes” between neighbors. The hole is defined as: if the delta time between two neighbors' rows is above one second, there is a “hole” between them. These two neighbors belong to different “sessions.” I noticed that the sensor data between other sessions might be different. 
- I also noticed that the actual target is always constant within these sessions. I incorporated my findings into feature generation and postprocessing of my predictions.
- I used stratified group k fold for validation. Stratified = each fold has approximately the same number of samples for each class. Group = provided test_suite column.

3. Solution Architecture
Please describe how your algorithm handles the following steps:
- Feature generation: I had several ideas on features generation, and I combined them into the following groups. 1 - Raw sensor data. Provide data “as is.” 2 - Rolling statistics with different time windows (5, 999999 seconds) for both separate sessions and “global” (i.e. no separate sessions). Rolling statistics include: mean, std, z-score: [x - mean(x)] / std(x) 3 - Shift features, i.e. the value of sensor data a second ago, two secodns ago, etc. 4 - Features based on the interactions between sensor data, e.g., the value of Zephyr_HR divided by the value of Zephyr_HRV. 5 - Features based on the distances between eyes positions and gazing points.
- Correlation: I used gradient boosting trees for classification. The model consists of many decision trees, each tree outputs a probability of a certain class. The final prediction is a weighted sum of predictions of each tree. 
- Postprocessing: The target is the same within a single session. Therefore, I post-processed predictions by calculating the rolling average of the model’s predictions from the beginning of the session till time t (including time t) for which we’re making predictions.

4. Final Approach
Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance; Include the feature importance techniques used and how you selected the data tags used.

- Target: We have to predict cognitive start for time t and t+3. The target for t is equal to the value of the induced_state column. The target for t+3 is the same as the target for t because the cognitive state is the same within the session. The data and target are the same, so I decided to train a single model, and use the same model for making predictions for t and t+3. Note: There is a separation between t and t+3 models in the code. I decided to keep it just in case the data would be different in the future.
-Model: I used the Lightgbm classifier. I optimized hyperparameters using Optuna. The final prediction is the average predictions of several Lightgbm classifiers with different hyperparameters.
Features importance: To determine the most important features for making predictions for time t, I used built-in SHAP values calculation. Then, I selected top-3 sensor features from the output.

5. Open Source Resources, Frameworks and Libraries
Please specify the name of the open-source resource, the URL to where it can be found, and it’s license type: all libraries are open-sourced.

- pandas==1.3.5
- numpy==1.22.1
- lightgbm==3.3.2
- scikit-learn==1.0.2
- tqdm==4.62.3
- scipy==1.7.3
- optuna==2.10.0

6. Potential Algorithm Improvements
Please specify any potential improvements that can be made to the algorithm:
- We can test improving hyperparameters of the model by running additional hyperparameters tuning 
- The model is vulnerable to the absence of input features, so it’s a good idea to use augmented data (i.e. features when one or more input signals are missing) for training
- We can test other than gradient boosting models. I.e. neural nets. 

7. Algorithm Limitations
Please specify any potential limitations with the algorithm:
- The algorithm requires historical data for features engineering. If the historical data isn’t available the performance of the algorithm will drop. 
- The model uses all available input features for making predictions. The absence of features may result in a worse score.

8. Deployment Guide
Please provide the exact steps required to build and deploy the code:
-Same as pt 9 excluding model training (step 2).

9. Final Verification
Please provide instructions that explain how to train the algorithm and have it execute against sample data:
- Step 1 - build docker `docker build -t 'latest' .` 
- Step 2 - run model training 
`docker run -v /path_to_data/:/work/data/ \ -v /path_to_save_model/:/work/model/ \ latest sh train.sh /work/data/data_training.zip`
- Step 3 - run model predictions 
`docker run -v /path_to_data/:/data/ \ -v /path_to_model/:/work/model/ \ latest sh test.sh /data/test_data_file.zip /data/file_to_save_predictions.csv`

