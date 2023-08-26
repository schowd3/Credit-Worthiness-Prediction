# Credit-Worthiness-Prediction

# Objective and Background
The aim of this project was to create a predictive model capable of assessing a customer's creditworthiness. In this context, a credit score of 0 corresponds to high risk, while a score of 1 signifies low risk. The intention was to provide loans exclusively to individuals classified as low-risk, considering the dataset's inherent imbalance. As a result, the F1-score was chosen as the scoring function over simple accuracy. The F1-score represents the harmonic mean of precision and recall, offering a more suitable evaluation metric for imbalanced datasets.

# About the Test Set:
In the training set, there are 24,720 instances of Class 0 and 7,841 instances of Class 1, totaling 32,561 examples. The distribution of labels in the test set is unspecified, which comprises 13,305 examples. The training file is composed of 13 columns: the first 12 columns represent features, and the 13th column contains the corresponding labels. In contrast, the testing file consists of 12 columns that correspond solely to the features.

# Solution:
- The K nearest neighbor algorithm is used for performing classification. The similarities from train data to test data are calculated to classify the credit and k nearest neighbors are obtained. From k nearest neighbors, the labels to identify the creditworthiness of a customer can be obtained.

- Refer to the report.pdf for further details.
