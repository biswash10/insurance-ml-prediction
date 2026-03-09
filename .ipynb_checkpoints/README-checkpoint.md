# Insurance Claim Prediction using Machine Learning

## Project Overview

This project predicts **medical insurance claim costs** using machine
learning models.

The dataset contains information about individuals such as:

-   Age
-   Gender
-   BMI
-   Blood Pressure
-   Diabetes status
-   Number of children
-   Smoking habit
-   Region

Using these features, the model predicts the **insurance claim amount**.

The goal of this project is to: - Perform **data analysis** - Apply
**data preprocessing** - Train **machine learning models** - Compare
model performance

------------------------------------------------------------------------

# Dataset Information

Dataset size:

-   Rows: 1332
-   Columns: 10

Features used for prediction:

  Feature         Description
  --------------- ----------------------
  age             Age of the person
  gender          Male / Female
  bmi             Body Mass Index
  bloodpressure   Blood pressure value
  diabetic        Diabetes status
  children        Number of children
  smoker          Smoking status

Target variable:

`claim`

which represents the **insurance cost**.

------------------------------------------------------------------------

# Data Preprocessing

The following preprocessing steps were performed:

1.  Removed missing values using `dropna()`
2.  Checked duplicates
3.  Encoded categorical variables using **Label Encoding**
4.  Feature scaling using **StandardScaler**
5.  Split data into:

-   80% Training Data
-   20% Testing Data

------------------------------------------------------------------------

# Exploratory Data Analysis

## Distribution of Numerical Features

![Numerical Distribution](images/numeric_feature_distribution.png)

## Distribution of Categorical Features

![Categorical Distribution](images/categorical_distribution.png)

## Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

------------------------------------------------------------------------

# Machine Learning Models Used

## 1. Linear Regression

Linear Regression is a **supervised machine learning algorithm used for
regression tasks**.

It assumes a **linear relationship** between the input features and the
target variable.

Formula:

y = b0 + b1x1 + b2x2 + ... + bnxn

Where: - `y` = predicted claim - `x` = input features - `b` =
coefficients

Advantages:

-   Simple
-   Fast
-   Easy to interpret

Limitations:

-   Cannot capture complex nonlinear relationships.

------------------------------------------------------------------------

## 2. Random Forest Regressor

Random Forest is an **ensemble learning algorithm** that combines
multiple decision trees.

Instead of using one decision tree, Random Forest builds **many trees**
and averages their predictions.

How it works:

1.  Creates many decision trees
2.  Each tree is trained on random subsets of data
3.  Final prediction is the **average of all trees**

Advantages:

-   Handles nonlinear relationships
-   Reduces overfitting
-   Often achieves higher accuracy

------------------------------------------------------------------------

# Model Evaluation Metrics

Three evaluation metrics were used.

## 1. R² Score (Coefficient of Determination)

Measures how well the model explains the variance in the data.

Range:

-   0 → Poor model
-   1 → Perfect model

Higher is better.

## 2. MAE (Mean Absolute Error)

Average absolute difference between predicted and actual values.

MAE = mean(\|y_true - y_pred\|)

Lower MAE means better predictions.

## 3. RMSE (Root Mean Squared Error)

Measures error magnitude but penalizes large errors more.

RMSE = sqrt(mean((y_true - y_pred)\^2))

Lower RMSE means better model performance.

------------------------------------------------------------------------

# Model Performance Comparison

  Model               R2     MAE    RMSE
  ------------------- ------ ------ ------
  Linear Regression   0.68   5096   6947
  Random Forest       0.79   4149   5652

Random Forest performed better than Linear Regression.

------------------------------------------------------------------------

# Model Comparison Graphs

## R² Comparison

![R2 Comparison](images/model_r2_comparison.png)

## MAE Comparison

![MAE Comparison](images/model_mae_comparison.png)

## RMSE Comparison

![RMSE Comparison](images/model_rmse_comparison.png)

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-Learn
-   Jupyter Notebook

------------------------------------------------------------------------

# Conclusion

-   Random Forest performed better than Linear Regression.
-   Smoking and BMI strongly influence insurance claim costs.
-   Ensemble models like Random Forest handle complex relationships
    better.

Future improvements could include:

-   Hyperparameter tuning
-   Using Gradient Boosting models
-   Deploying the model as an API

------------------------------------------------------------------------

# Author

Machine Learning Project
