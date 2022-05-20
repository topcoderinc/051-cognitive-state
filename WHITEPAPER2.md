# Fault Tolerance Whitepaper
## Overview
This codebase is a winner of the Topcoder NASA Cognitive State Determination - Fault Tolerance marathon match. As part of the final submission the competitor was asked to compelte this document. Personal details have been removed. 

## 1. Introduction
Tell us a bit about yourself, and why you have decided to participate in the contest.
- **Handle:** fly_fly	
- **Placement you achieved in the MM:** 1st
- **About you:**  freelancer
- **Why you participated in the MM:** TCO points, money, fame

## 2. Solution Development
How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider? Also try to add your cross validation approaches used.
- During EDA, I noticed that eye distance feature has bad influence on performance of the model.  In my opinion, the loss of left or right features which may amplify the error because of square calculation. In the end, I did not use the eye distance feature and this reduction improve score a lot (about +10%).   
- To make the model less sensitive to the loss of some features, I increase the value of L1 and L2 regularization, and add more time windows of features based on original (including: mean, median, std, max).
- Stratified group k fold is also used for validation. Stratified = each fold has approximately the same number of samples for each class. Group = provided test_suite column.

## 3. Open Source Resources, Frameworks and Libraries
Please specify the name of the open source resource, the URL to where it can be found, and it’s license type:
- all libraries are open-sourced
- pandas==1.3.5 
- numpy==1.22.1 
- lightgbm==3.3.2 
- scikit-learn==1.0.2
- tqdm==4.62.3 
- scipy==1.7.3 
- optuna==2.10.0

## 4. Potential Algorithm Improvements
Please specify any potential improvements that can be made to the algorithm:
- You can try other models. Such as TabNet or other Neural Networks.
- You can try more hyperparameter tuning.


## 5. Algorithm Limitations
Please specify any potential limitations with the algorithm:
- This is not a pretrained model, so you need to train it from scratch before predcit.
- Too many absent of data features may result in a worse score.
