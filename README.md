# NASA Conitive State Determination :thought_balloon:
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

In the near future, Astronauts will need to carry out autonomous operations as they venture beyond Low Earth Orbit. For many of the activities that the astronauts perform, the same training is provided to all crew members months ahead of their mission. A system that provides real-time operations support that is optimized and tailored to each astronaut's psychophysiological state at the time of the activities and which can be used during the training leading up to the mission would significantly improve human spaceflight operations. 

In this challenge, solvers were asked to determine the present cognitive state of a trainee based on biosensor data, as well as predict what their future cognitive state will be, three seconds in the future.

## Data loading 

The provided data has a high frequency (1 sec = thousands of rows), but the labeling was done manually. Also, final predictions are expected to have a frequency of one (one second = one row). I decided to transform the data so second equals one row both for training and testing. I rounded all timestamps to the closes second and tested several approaches: 
1. Take the first value
2. Take the last value 
3. Calculate the mean value for numerical values and mode for the target
4. Calculate mean and std for numerical values and mode for target

The first approach showed the best results. 

## EDA

During EDA, I noticed that the timestamps might have “holes” between neighbors. The hole is defined as: if the delta time between two neighbors' rows is above one second, there is a “hole” between them. These two neighbors belong to different “sessions.” I noticed that the sensor data between other sessions might be different. I also noticed that the actual target is always constant within these sessions. I incorporated my findings into feature generation and postprocessing of my predictions. 


## Creating target 

We have to predict cognitive start for time `t` and `t+3`. The target for `t` is equal to the value of the `induced_state` column. The target for `t+3` is the same as the target for `t` because the cognitive state is the same within the session. The data and target are the same, so I decided to train a single model, and use the same model for making predictions for `t` and `t+3`. Note: There is a separation between `t` and `t+3` models in the code. I decided to keep it just in case the data would be different in the future. 


## Feature generation 

I had several ideas on features generation, and I combined them into the following groups. 

1 - Raw sensor data. Provide data “as is.”
2 - Rolling statistics with different time windows (5, 999999 seconds) for both separate sessions and “global” (i.e. no separate sessions). Rolling statistincs include: mean, std, z-score: [x - mean(x)] / std(x)
3 - Shift features, i.e. the value of sensor data a second ago, two secodns ago, etc.
4 - Features based on the interactions between sensor data, e.g., the value of Zephyr_HR divided by the value of Zephyr_HRV. 
5 - Features based on the distances between eyes positions and gazing points. 


## Validation 

I used stratified group k fold for validation. Stratified = each fold has approximately the same number of samples for each class. Group = provided `test_suite` column. 


## Model

I used the Lightgbm classifier. I optimized hyperparameters using Optuna. The final prediction is the average predictions of several Lightgbm classifiers with different hyperparameters. 

## Postprocessing 

As mentioned in EDA, the target is the same within a single session. Therefore, I post-processed predictions by calculating the rolling average of the model’s predictions from the beginning of the session till time `t` (including time `t`) for which we’re making predictions. 

## Important features 

To determine the most important features for making predictions for time t, I used built-in SHAP values calculation. Then, I selected top-3 sensor features from the output. 

## Libraries 

You can find the list of used libraries in `requirenments.txt`.
