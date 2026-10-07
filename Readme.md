# Titanic Survival Prediction using Logistic Regression

This is a machine learning project where I used Logistic Regression to predict whether a passenger survived the Titanic disaster.

I built this project while learning classification in Machine Learning. I went through the complete process from preprocessing the dataset to training the model and checking its performance.


## About the Dataset

I used the Titanic dataset with 891 rows and 12 columns.

The features I used are:

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare

The target variable is:

- Survived
  - 0 = Did not survive
  - 1 = Survived

I did not use all the columns from the dataset. I selected the features that I wanted to use for this first Logistic Regression model.

## What I Did

The main steps in this project were:

1. Loaded and explored the dataset
2. Selected the features and target
3. Split the data into training and testing sets
4. Handled missing values in the Age column
5. Converted Sex into a numerical feature
6. Applied StandardScaler
7. Trained a Logistic Regression model
8. Made predictions on the test data
9. Evaluated the model using different metrics
10. Visualized the results


## Data Preprocessing

### Missing Values

The Age column had missing values.

Instead of removing those rows, I used the median Age from the training data to fill the missing values.

### Encoding

The Sex column contains text values such as Male and Female.

I converted it into a numerical column:

`Sex_male`

where:

- 0 = Female
- 1 = Male

### Feature Scaling

I used StandardScaler before training the Logistic Regression model.

The scaler was fitted only on the training data and then used to transform the test data.

This helped avoid data leakage from the test set.

## Model

I used Logistic Regression from scikit-learn.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,
    max_iter=1000
)

model.fit(X_train, y_train)