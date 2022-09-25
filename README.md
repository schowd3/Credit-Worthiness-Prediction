# Credit-Worthiness-Prediction
The goal of this project was to develop a predictive model that can determine the credit worthiness of a customer. 0 indicates a high risk, and 1 indicates a low risk. Then loans could potentially be made to just the low risk people.  Since the dataset is imbalanced the scoring function will be the F1-score instead of simple accuracy. The F1 score is the harmonic mean of precision and recall.



The K nearest neighbor algorithm is used for performing classification.To classify the credit, the similarities from train data to test data are calculated and k nearest neighbors are obtained. From k nearest neighbors, the labels to identify the creditworthiness of a customer can be obtained.
