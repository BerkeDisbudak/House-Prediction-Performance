Model Training and Prediction - Housing Price Prediction
Overview

This Python script demonstrates how to train a machine learning model to predict housing prices using the XGBoost regressor. Initially, a RandomForestRegressor was considered, but XGBoost was selected for its better performance in this case. To streamline the process and make the code more efficient, we utilized a Pipeline to handle preprocessing, model training, and evaluation in a single workflow.
Key Steps

    Data Loading: The dataset is loaded from CSV files. We read both the training and test datasets and clean the data by dropping rows with missing target values (SalePrice).

    Feature Selection: Only numerical features are selected for the training process, and the target variable (SalePrice) is separated from the features.

    Data Splitting: The training data is split into training and validation sets (80% for training and 20% for validation).

    Pipeline Creation: A pipeline is created to:
        Impute missing values using the median strategy (SimpleImputer).
        Scale the features to a [0, 1] range using MinMaxScaler.
        Train the model using XGBRegressor (XGBoost).

    Cross-Validation: We perform 5-fold cross-validation to evaluate the model's performance on the training set using the negative mean absolute error (MAE) as the scoring metric.

    Model Training: The pipeline is trained on the full training set, and predictions are made on the validation set to compute the MAE.

    Final Model Training: The model is retrained on the entire dataset (including the validation data) and used to make predictions on the test set.

    Results Visualization: A histogram of the predicted sale prices is plotted to visualize the distribution of predictions.

Code Optimizations

    XGBoost: Although RandomForestRegressor was initially considered, XGBoost was chosen due to its superior performance in this regression task, providing better accuracy and model efficiency.

    Pipeline: The use of a Pipeline makes the code more concise and maintainable by automating preprocessing, scaling, and model fitting in a single object. This minimizes code duplication and improves efficiency when making predictions on new datasets.

Requirements

    Python 3.x
    pandas
    numpy
    matplotlib
    scikit-learn
    xgboost
    seaborn (for visualization, optional)

You can install the required libraries using:

pip install pandas numpy matplotlib scikit-learn xgboost seaborn

Usage

    Place the training and test data CSV files (train.csv and test.csv) in the same directory as the script.
    Run the script to load the data, train the model, and visualize the results.

Let me know if you need further modifications or clarifications!
