# Energy Efficiency Prediction for Buildings 🏢☀️❄️

This project focuses on predicting the heating load (Y1) and cooling load (Y2) of residential buildings using machine learning techniques. It utilizes the "Energy Efficiency" dataset (ENB2012_data), which contains various building parameters. Two regression models, XGBoost Regressor and K-Nearest Neighbors (KNN) Regressor, are implemented, optimized through hyperparameter tuning, and their performances are compared.

## Dataset

*   **Name:** Energy Efficiency Dataset (ENB2012_data.csv)
*   **Source:** The dataset originates from simulations using 12 different building shapes in Ecotect.
*   **Citation:** Tsanas, A., & Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. *Energy and Buildings*, *49*, 560-567.
*   **Dataset Details:**
    *   The dataset comprises 768 samples and 8 features, aiming to predict two real-valued responses (Heating Load and Cooling Load).
    *   **Features (X1-X8):**
        *   `X1`: Relative Compactness
        *   `X2`: Surface Area
        *   `X3`: Wall Area
        *   `X4`: Roof Area
        *   `X5`: Overall Height
        *   `X6`: Orientation
        *   `X7`: Glazing Area
        *   `X8`: Glazing Area Distribution
    *   **Target Variables (Y1, Y2):**
        *   `Y1`: Heating Load
        *   `Y2`: Cooling Load

## Key Features of the Analysis

*   **Multi-Output Regression:** Predicts two target variables (Heating Load and Cooling Load) simultaneously using `MultiOutputRegressor`.
*   **Model Comparison:** Implements, trains, and evaluates XGBoost Regressor and K-Nearest Neighbors (KNN) Regressor.
*   **Hyperparameter Optimization:**
    *   `RandomizedSearchCV` is used for optimizing XGBoost.
    *   `GridSearchCV` is used for optimizing KNN.
*   **Feature Engineering & Preprocessing:**
    *   **XGBoost:** Uses a subset of features (`X1` Relative Compactness, `X3` Wall Area, `X7` Glazing Area).
    *   **KNN:** Uses all 8 input features. Features (from `X2` to `X8`) are standardized using `StandardScaler`, while `X1` (Relative Compactness) is kept in its original scale.
    *   Data is split into training (80%) and testing (20%) sets.
*   **Comprehensive Evaluation:** Models are evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.
*   **Data Visualization:**
    *   Correlation heatmap of all variables.
    *   Scatter plots showing relationships between selected features and target variables.
    *   Box plots for visualizing the distribution of selected variables.
    *   Pairplot for a deeper look into feature interactions and distributions.
    *   Observed vs. Predicted values plot for the best XGBoost model.
    *   Learning curve for the best XGBoost model to assess overfitting/underfitting.

## Methodology

1.  **Data Loading and Exploration:**
    *   The `ENB2012_data.csv` dataset is loaded.
    *   Initial data exploration includes checking data types, summary statistics, and missing values.
    *   Visualizations (heatmap, scatter plots, box plots, pairplot) are generated to understand data distributions and relationships.

2.  **Data Preparation:**
    *   Features (X) and target variables (Y1, Y2) are separated.
    *   Specific features are selected for the XGBoost model (`X1`, `X3`, `X7`). All features are used for the KNN model.
    *   The dataset is split into training and testing sets.
    *   For the KNN model, features (except `X1`) are standardized using `StandardScaler`.

3.  **Model Training and Hyperparameter Tuning:**
    *   **XGBoost:**
        *   `XGBRegressor` is wrapped with `MultiOutputRegressor`.
        *   Hyperparameters are tuned using `RandomizedSearchCV` with 5-fold cross-validation, optimizing for R² score.
    *   **KNN:**
        *   `KNeighborsRegressor` is wrapped with `MultiOutputRegressor`.
        *   Hyperparameters (number of neighbors, weights, metric) are tuned using `GridSearchCV` with 5-fold cross-validation, optimizing for R² score.

4.  **Model Evaluation:**
    *   The best-tuned XGBoost and KNN models are used to make predictions on the test set.
    *   Performance is evaluated using MAE, MSE, RMSE, and R² score.
    *   A comparison table summarizing the performance of both models is printed.

5.  **Visualization of Results:**
    *   A scatter plot of observed vs. predicted values is generated for the XGBoost model's predictions on both target variables.
    *   A learning curve is plotted for the best XGBoost model to analyze its training behavior.

## Requirements

*   Python 3.x
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   xgboost

You can install these dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Use code with caution.
Markdown
How to Run
Ensure all dependencies are installed (see Requirements section).
Place the dataset file ENB2012_data.csv in the same directory as the Python script Energy_Efficient.py.
Run the script from your terminal:
python Energy_Efficient.py
Use code with caution.
Bash
The script will:
Print dataset information and descriptive statistics.
Display various exploratory data visualizations.
Train and tune the XGBoost and KNN models, printing the best parameters and cross-validation scores.
Output the performance metrics (MAE, MSE, RMSE, R²) for both models on the test set.
Display a comparison table of the model performances.
Show the Observed vs. Predicted plot and the Learning Curve for the XGBoost model.
Expected Output
The script will produce console output detailing the data, model training process, best hyperparameters, and evaluation metrics. Several plot windows will appear showcasing the visualizations described above. The final output will include a clear comparison of the XGBoost and KNN models' predictive performance.
Further Dataset Information
(As provided in the original context)
We perform energy analysis using 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters. We simulate various settings as functions of the aforementioned characteristics to obtain 768 building shapes. The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses. It can also be used as a multi-class classification problem if the response is rounded to the nearest integer.
Attribute Information:
The dataset contains eight attributes (or features, denoted by X1…X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.
Specifically:
X1 Relative Compactness
X2 Surface Area
X3 Wall Area
X4 Roof Area
X5 Overall Height
X6 Orientation
X7 Glazing Area
X8 Glazing Area Distribution
y1 Heating Load
y2 Cooling Load
Relevant Papers:
A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools', Energy and Buildings, Vol. 49, pp. 560-567, 2012
Citation Request:
A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools', Energy and Buildings, Vol. 49, pp. 560-567, 2012
For further details on the data analysis methodology:
A. Tsanas, 'Accurate telemonitoring of Parkinson’s disease symptom severity using nonlinear speech signal processing and statistical machine learning', D.Phil. thesis, University of Oxford, 2012
