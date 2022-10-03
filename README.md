# Credit-Worthiness-Prediction

# Objective and Background
The goal of this project was to develop a predictive model that can determine the credit worthiness of a customer. 0 indicates a high risk, and 1 indicates a low risk. Then loans could potentially be made to just the low risk people.  Since the dataset is imbalanced the scoring function will be the F1-score instead of simple accuracy. The F1 score is the harmonic mean of precision and recall.

# About the Test Set:
The training set has 24720 examples of Class 0 and 7841 examples of Class 1 (32561 total). There is no information about the distribution of labels in the test set, which contains 13305 examples. The training file has 13 columns, with the first 12 being the features and the 13th the label. The testing file has 12 columns corresponding to the features.


# Solution:
- The K nearest neighbor algorithm is used for performing classification.To classify the credit, the similarities from train data to test data are calculated and k nearest neighbors are obtained. From k nearest neighbors, the labels to identify the creditworthiness of a customer can be obtained.

- Refer to the report.pdf for further details.
