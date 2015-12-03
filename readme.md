# Prediction of the number of bicycles at rental stations
The goal in this assignment is to predict the number of available bicycles in
all rental stations 3 hours in advance. There are at least two use cases for
such predictions. First, a user plans to rent (or return) a bike in 3 hours'
time and wants to choose a bike station which is not empty (or full). Second,
the company wants to avoid situations where a station is empty or full and therefore
needs to move bikes between stations. For this purpose, they need to know which stations
are more likely to be empty or full soon.

The assignment will be organised by 3 phases. During each phase you will be given
a particular task. You can earn different proportions of the marks by finishing
these phases step by step.

## Phase 1:
In this phase, you will be given the data of 75 stations (Station 201 to Station 275)
for the period of one month. The task is to predict availability for each station for the next 3 months.

There are two approaches for this task:

1. Train a different model for each station.
2. Train a single model for all the stations.

Implement your models based on both approaches and check which approach is better. Investigate and discuss the results.

(The training data is given by Train.zip, the test data is given by test.csv).

(Build your models and submit the predictions according to the format given by example_leaderboard_submission.csv).

### Assessment - up to 60%
- Train a different model for each station and check the performance. (15%)
- Train a single model for all the stations and check the performance. (15%)
- Try different feature selection for both approaches and compare the results. (10%)
- Both classifiers should give a performance better than the baseline (see below). (10%)
- A comparison between the two approaches and a discussion of the results. (10%)

## Phase 2:
Now you will be given a set of linear models trained on other stations (Station 1 to Station 200) with the training data from a whole year. Although these models are not trained on the stations to be predicted, they can still be used since there should be some similarity among different stations. To successfully use these models can help reuse the knowledge learned from a whole year's data.

The task then is to figure out how to predict the stations in Phase 1 by only using these trained models. Investigate the resulting performances and compare to your own classifiers in Phase 1.

(The pre-trained linear models are given by Models.zip).

### Assessment - up to 85%
- Investigate a method to use the provided linear models. (10%)
- Your method should give a performance better than the baseline (see below). (5%)
- Compare the results to your own models in Phase 1 and discuss the results. (10%)

## Phase 3:
Try to achieve an even better performance by designing a approach to combine your own models with the given linear models.

### Assessment - up to 100%
- Describe your method and discuss the results. (10%)
- Achieve a better performance than the models in Phase 1 and Phase 2. (5%)

## Baseline
Two baseline approaches are given for the participant to evaluate their results.

1. Predict as bikes_3h_ago (i.e. same number of bikes as 3 hours ago).
    * For the public leaderbroad, the MAE for this baseline is about 2.714.
2. Predict as bikes_3h_ago + difference_profile (i.e. predicting the same change as usually between this time of week and 3 hours ago).
    * For the public leaderbroad, the MAE for this baseline is about 2.261 with the full profile.
    * For the public leaderbroad, the MAE for this baseline is about 2.295 with the short profile.

## Evaluation

The predictions are evaluated according to the mean absolute error (MAE) between the predicted and true values. The winner is the participant who submitted the predictions with the lowest mean absolute error.

File descriptions

Train.zip - the training set, unzip this and you will get the csv file for each station.
test.csv - the test set.
sampleSubmission.csv - a sample submission file in the correct format
Models.zip - linear models pre-trained from other stations.
Data fields

The data contain 4 station features, 8 time features, 7 weather features, 1 task-specific feature and 4 profile features plus 1 target variable. The target variable 
is 'bikes' and it is a non-negative integer representing the median number of available bikes during the respective hour in the respective rental station. There are 
4 features regarding the station:

station - integer from 1 to 275, representing the number of the station, also in the file name
latitude - real number representing geographical latitude of the station
longitude - real number representing geographical longitude of the station
numDocks - positive integer representing the maximal number of bikes that can be present in the station

There are 8 features regarding the timepoint:

timestamp - integer representing the Unix timestamp (seconds since Unix Epoch)
year - integer with 4 digits
month - integer from 1 (January) to 12 (December)
day - integer from 1 to 31
hour - integer from 0 to 23
weekday - string (Monday, Tuesday, ..., Sunday)
weekhour - integer from 1 to 168 representing the hour of the week (Monday 0h is weekhour 1, Sunday 23h is weekhour 168)
isHoliday - 1 (a national or local holiday), 0 (not a holiday)

There are 7 features regarding weather, these do not differ across stations:

windMaxSpeed.m.s
windMeanSpeed.m.s
windDirection.grades
temperature.C
relHumidity.HR
airPressure.mb
precipitation.l.m2

There is one feature regarding the number of bikes in the station 3 hours ago:

bikes_3h_ago
The profile variables are calculated from earlier available timepoints on the same station:

The 'full_profile_bikes' feature is the arithmetic average of the target variable 'bikes' during all past timepoints with the same weekhour, in the same station.
The 'full_profile_3h_diff_bikes' feature is the arithmetic average of the calculated feature 'bikes-bikes_3h_ago' during all past timepoints with the same weekhour, 
in the same station.
The 'short_*' profile s the same as the full profiles except that it only uses past 4 timepoints with the same weekhour. If there are less than 4 such timepoints 
then all are used. The missing values are ignored in all profile calculations, i.e. only the timepoints with existing values are averaged.
MODELS

All models are presented in the CSV format. 

For each station there are 6 linear models, all built using R function rlm from the package MASS, with missing value imputation using function na.roughfix from 
package randomForest. The models use the following features (plus an intercept term):

short: bikes_3h_ago, short_profile_3h_diff_bikes, short_profile_bikes
short_temp: bikes_3h_ago, short_profile_3h_diff_bikes, short_profile_bikes, temperature.C
full: bikes_3h_ago, full_profile_3h_diff_bikes, full_profile_bikes
full_temp: bikes_3h_ago, full_profile_3h_diff_bikes, full_profile_bikes, temperature.C
short_full: bikes_3h_ago, short_profile_3h_diff_bikes, short_profile_bikes, full_profile_3h_diff_bikes, full_profile_bikes
short_full_temp: bikes_3h_ago, short_profile_3h_diff_bikes, short_profile_bikes, full_profile_3h_diff_bikes, full_profile_bikes, temperature.C
