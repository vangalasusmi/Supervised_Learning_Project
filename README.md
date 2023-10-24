# Machine_learning_project-supervised-learning

## Project/Goals

The goal of the project is to perform a full supervised learning machine learning project on a "Diabetes" dataset. This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset. [Kaggle Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

## Process

This is a binary classification problem and I am using two ensemble methods to predict if a patient has diabetes or not. The two models are Random Forest Classifier and eXtreme Gradient Boosting. Their performances will be compared using evaluation metrics like precision, recall, f1 score and the AUC ROC score.

### The data was explored and cleaned it from inappropriate values. Missing values were disguised as 0. These values were replaced with the mean for `[Insulin]` while for `[Pregnancies]` and `[SkinThickness]`, the zeros were left alone, as this is completely normal. It is not possible for `[Insulin]` to be 0, the person will not survive. Other exploration and preprocessing tasks where carried out as follows:

1. All the columns were numerical datatypes which enabled us carry out statistical tests and plot comprehensive graphs. Histogram plots were used to observe the distribution of the data. Checking for if they follow the Gaussian distribution.

2. Bar plots were used to observe the distribution of the label for each feature. Violin plots were also used to further view the spread of the distribution and box plots were included for observation of the distribution's density and quartile values.

3. Box plots were further used to visualize all the outliers spread for each feature per target label. Pair plot was used to visualize the linear relationship between all the features.

4. Clipping was used to treat outliers in the dataset. Clipping was the choice for treating outliers because the dataset is quite small and the outliers are not evenly spread across the features. Hence, removing any would result in data quality issues and the model would not be properly trained. The clipping was done by calculating the mean and standard deviation as the upper and lower limit.

5. Observed class imbalance in the target and implemented the `stratify` option in the `train_test_split` preprocessing function to take care of the imbalance. The dataset does not have duplicates.

6. The data was then normalized using the `MinMaxScaler` function. This is because most of the features did not follow a Gaussian distribution. After normalization, the data was standardized using the `StandardScaler` function.

7. The data set were split for training and validation using the `train_test_split` function. The ratio was 80:20 in favor of the training data.

8. During EDA, it was discovered that `[SkinThickness]` was not correlated with `[Outcome]`. This was further validated through the feature scores obtained through the Random Forest algorithm. The column was dropped before training the model further.

## Results
The Random Forest Classifier model was trained and evaluated. After hyperparameter tuning, the model had evaluation metrics:
- Precision -- 70.73%
- Recall -- 53.7%
- F1 Score -- 61.05%

And an AUC ROC score of 70.85%.

The eXtreme Gradient Boost model was trained and evaluated. After hyperparameter tuning, the model had evaluation metrics:
- Precision -- 68.97%
- Recall -- 37.04%
- F1 Score -- 48.19%

And an AUC ROC score of 80.92%.

I ran the two models a significant number of times, in a bid to improve on the results. Before removing the `[SkinThickness]` feature, the XGBoost model performed much better than the Random Forest model in the model evaluation metrics. Except that it lagged behind in recall (false positives). After removing the feature, I got the presented results.

### Key takeaways
- The evaluation means that the Random Forest model does a better job at correctly predicting positive diabetes cases (true positives) compared to the XGBoost model. 
- The precision and AUC score of the XGBoost is higher than that of the Random Forest. This means that overall, the XGBoost will have maximum performance compared to Random Forest.
- The final decision on which model is better will depend on the intention of the client. If the client wants to minimize false negatives (prioritizing precision) then XGBoost is the better model. If the client wants to prioritize recall (minimize false positives), then Random Forest is the better model.

> The above takeaways were obtained before removing the `[SkinThickness]` feature. From the current model evaluation metrics, it can be clearly seen that Random Forest model is far better suited for this problem compared to the XGBoost. Which is a bumper for me, as I am a huge fan of the XGBoost model!!!

From this model process, and based on the current data preprocessing, the conclusion is that the Random Forest is better suited for predicting diabetes.The RF model predicts both true positives and false negatives better than the XGBoost. RF also has a higher F1 score. These are enough model evaluation metrics to crown Random Forest model as superior.

> Although, the performance of the models significantly differ, they both seem to agree on the ranking of feature importance. Both models predicted that `[Glucose]`, `[BMI]`, and `[Age]` are the three most important features that determines the presence of diabetes disease. Although, they both disagree on the next two features `[Insulin]` and `[DiabetesPedigreeFunction]`. All other feature importance are ranked the same.

## Challenges 
 - Not enough time to thoroughly think through the problem formulation and be creative. The estimate given on Compass was completely misleading as the Rubrics requirements are very comprehensive.
 - Everything was done in a rush and there was no time to properly think and intepret the model results and outputs.
 - The dataset is quite small and the preferred model will cannot be deployed because it has not been trained on enough data.

## Future Goals
Try to investigate the reason why the XGBoost has a significantly lower overall performnace, compared to the Random Forest. Probably remove some more features (like `[Pregnancies]`, `[BloodPressure]`) and compare the performance of both models. Check if normalizing and after which scaling was applied had an impact on the model's performance.
