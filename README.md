Housing Price Prediction
Overview

This project aims to predict housing prices using machine learning. The dataset contains various features related to house attributes, and the target variable is the sale price (SalePrice). Initially, a RandomForestRegressor was considered, but for better performance, we used the XGBoost model. To streamline the process and optimize the workflow, we utilized Pipeline from scikit-learn.
Key Features

    Model: XGBRegressor (XGBoost), chosen for its better performance compared to the initial RandomForestRegressor.
    Pipeline: A Pipeline is used to chain together preprocessing steps (imputation, scaling) and model training to create an efficient and concise workflow.
    Evaluation: The model is evaluated using cross-validation and mean absolute error (MAE).

Steps in the Process

    Data Loading:
        Training and test data are loaded from CSV files (train.csv and test.csv).
        Rows with missing values in the target variable (SalePrice) are removed.

    Feature Selection:
        Only numeric features are selected for training.
        The target variable (SalePrice) is separated from the features.

    Data Splitting:
        The data is split into training (80%) and validation (20%) sets.

    Pipeline Definition:
    A Pipeline is defined with the following steps:
        Imputation: Missing values are handled using SimpleImputer with the median strategy.
        Scaling: Features are scaled using MinMaxScaler to normalize them to a range [0, 1].
        Model: The model used is XGBRegressor, a powerful gradient boosting algorithm.

    Cross-Validation:
        The model's performance is evaluated using 5-fold cross-validation.
        The negative mean absolute error (MAE) is used as the scoring metric.

    Model Training and Prediction:
        The pipeline is trained on the training set.
        Predictions are made on the validation set, and the MAE is calculated to evaluate model performance.

    Final Model:
        The model is retrained on the entire dataset (combining training and validation sets) and used to predict on the test set.

    Visualization:
        A histogram of the predicted sale prices is plotted to visualize the distribution of the predictions.

Code Optimizations

    XGBoost:
    The XGBoost model was selected over RandomForestRegressor due to its better performance in regression tasks, providing better predictive accuracy.

    Pipeline:
    Using a Pipeline ensures that the preprocessing and model training steps are linked together in a structured and maintainable way. This eliminates redundant code and makes it easier to apply transformations to new datasets.

Requirements

    Python 3.x
    pandas
    numpy
    matplotlib
    scikit-learn
    xgboost
    seaborn (for visualization)

You can install the necessary libraries using pip:

pip install pandas numpy matplotlib scikit-learn xgboost seaborn

Usage

    Place the training (train.csv) and test (test.csv) datasets in the same directory as this script.
    Run the script, which will:
        Load the data,
        Train the model,
        Evaluate its performance,
        Display a histogram of predicted sale prices.
