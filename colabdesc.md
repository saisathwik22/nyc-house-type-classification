## Project Description: Airbnb Room Type Prediction in NYC

This project aims to develop a machine learning model to predict the `room_type` of Airbnb listings in New York City based on various property characteristics. The workflow encompasses data loading, extensive exploratory data analysis (EDA), data cleaning and feature engineering, model selection, hyperparameter tuning, and final model evaluation and saving.

### 1. Data Loading and Initial Exploration

- **Data Source**: The Airbnb dataset for New York City was loaded using the `kagglehub` library, which provides a convenient way to access Kaggle datasets within a Colab environment. The data was then read into a Pandas DataFrame using `pd.read_csv()`.
- **Initial Inspection**: Basic commands such as `df.head()`, `df.info()`, `df.describe()`, `df.shape`, and `df.isnull().sum()` were used to get a first glance at the data's structure, column types, summary statistics, dimensions, and the extent of missing values.
- **Target Variable Analysis**: The `room_type` column, which is our target variable, was examined for unique values (`df['room_type'].unique()`) and value counts (`df['room_type'].value_counts()`). This step revealed that the classes (`Entire home/apt`, `Private room`, `Shared room`) are imbalanced, with 'Shared room' being the minority class.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand the data's distribution, relationships between variables, and to identify potential patterns or anomalies:

- **Numerical Feature Distributions**: Histograms were generated for key numerical columns (`price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`) to visualize their distributions and identify skewness or outliers.
- **Categorical Feature Distributions**: A count plot of `neighbourhood_group` was created to show the distribution of listings across different boroughs in NYC.
- **Room Type vs. Price**: A box plot visualized the relationship between `room_type` and `price`, which helped in understanding price variations across different room types and highlighted the presence of significant outliers in pricing.
- **Correlation Analysis**: A correlation matrix, including `latitude` and `longitude` along with other numerical features, was computed and visualized using a Seaborn heatmap. This helped in identifying linear relationships between variables.
- **Geographical Distribution**: Scatter plots of `longitude` vs `latitude` were used to visualize the geographical distribution of listings across NYC. These plots were further enhanced by coloring points by `room_type` and adjusting transparency (`alpha=0.5`) and size (`s=10`) to better observe spatial patterns and concentrations of different room types.

### 3. Data Cleaning & Feature Engineering

This stage involved preparing the data for model training:

- **Dropping Irrelevant Columns**: Identifier columns such as `id`, `name`, `host_id`, `host_name`, and `last_review` were dropped as they are unique identifiers or free text and do not provide generalized predictive power for the model. The cleaned DataFrame was stored as `df_clean`.
- **Handling Missing Values**: Missing values in the `reviews_per_month` column were imputed with 0, assuming that listings without reviews do not have a review rate. (Note: `name` and `host_name` also had missing values, but these columns were dropped).
- **Outlier Capping**: Extreme outliers in `price` and `minimum_nights` were capped at their 99th percentile using `clip(upper=quantile(0.99))`. This prevents highly unusual values (likely data entry errors) from disproportionately influencing the model while retaining the overall distribution of the data.
- **Feature-Target Separation**: The dataset was split into features (`X`) and the target variable (`y`), where `y` represents `room_type`.

### 4. Train / Test Split

- The data was divided into training and testing sets using `train_test_split` from `sklearn.model_selection`.
- A `test_size` of 33% was used, meaning one-third of the data was reserved for final evaluation.
- `random_state=42` ensured reproducibility of the split.
- Crucially, `stratify=y` was applied to ensure that the proportion of each `room_type` class was maintained in both the training and testing sets. This is vital for handling imbalanced datasets and ensuring that the model is evaluated on a representative sample of all classes.

### 5. Preprocessing with `ColumnTransformer` and `Pipeline`

To ensure robust and consistent data transformations without data leakage, a `ColumnTransformer` was used within a `Pipeline`:

- **Numerical Pipeline**: For numerical features (`latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`):
    - `SimpleImputer(strategy='median')`: Fills any remaining missing numerical values with the median of the respective column (learned only from the training data).
    - `StandardScaler()`: Scales the features to have a mean of 0 and a standard deviation of 1.
- **Categorical Pipeline**: For categorical features (`neighbourhood_group`, `neighbourhood`):
    - `SimpleImputer(strategy='most_frequent')`: Fills any missing categorical values with the most frequent category.
    - `OneHotEncoder(handle_unknown='ignore')`: Converts categorical variables into a one-hot encoded format, creating binary columns for each category. `handle_unknown='ignore'` prevents errors if unseen categories appear in the test set.
- **`ColumnTransformer`**: Combines these pipelines, applying the numerical transformations to `numerical_cols` and categorical transformations to `categorical_cols`.

### 6. Model Comparison

Several classification algorithms were compared to identify the most suitable model for the task. Each model was wrapped in a `Pipeline` that included the `preprocessor` to ensure consistent data transformation:

- **Models Evaluated**:
    - `Logistic Regression`: A linear model, serving as a baseline.
    - `Decision Tree Classifier`: A non-linear model, prone to overfitting.
    - `Random Forest Classifier`: An ensemble method that typically performs well.
    - `Gradient Boosting Classifier`: Another powerful ensemble method.
- **Class Imbalance Handling**: For models that support it (Logistic Regression, Decision Tree, Random Forest), `class_weight="balanced"` was used to give more importance to the minority class during training, addressing the imbalance observed in `room_type`.
- **Evaluation Metric**: Due to class imbalance, both `accuracy` and `f1_macro` were used for evaluation via 3-fold stratified cross-validation on the training data. `f1_macro` calculates the F1-score independently for each class and then takes the average, treating all classes equally.
- **Results**: The performance (mean accuracy and macro F1-score) of each model was printed, indicating that Random Forest and Gradient Boosting performed best in terms of macro F1-score.

### 7. Hyperparameter Tuning

- **Model Selection**: The Random Forest Classifier, which showed strong performance in the initial comparison, was chosen for hyperparameter tuning.
- **Tuning Strategy**: `RandomizedSearchCV` was employed for tuning, as it efficiently explores a wide range of hyperparameter combinations compared to an exhaustive grid search. `n_iter=10` specified the number of parameter settings that are sampled.
- **Hyperparameters Tuned**:
    - `classifier__n_estimators`: Number of trees in the forest (e.g., 100, 200, 150, 300).
    - `classifier__max_depth`: Maximum depth of the individual trees (e.g., 8, 12, 15, 20, or `None` for unlimited depth).
    - `classifier__min_samples_split`: Minimum number of samples required to split an internal node (e.g., 2, 5, 10).
- **Optimization Metric**: The search was optimized for `f1_macro`.
- **Results**: The best parameters and the corresponding cross-validated Macro-F1 score were printed, indicating the optimal configuration found for the Random Forest model.

### 8. Final Model Evaluation

- **Best Estimator**: The `best_pipeline` (which incorporates the preprocessor and the best-tuned Random Forest model from `RandomizedSearchCV`) was used to make predictions on the unseen `X_test` data.
- **Metrics**: `accuracy_score` and `f1_score` (with `average='macro'`) were calculated on the test set to report the final performance of the chosen model.
- **Confusion Matrix**: A confusion matrix was generated and visualized using `seaborn.heatmap`. This provided a detailed breakdown of correct and incorrect classifications for each `room_type`, helping to understand the model's performance on individual classes, especially the minority class. The `xticklabels` and `yticklabels` were set to the actual class names for better interpretability.

### 9. Model Saving

- **Serialization**: The entire `best_pipeline`, including all preprocessing steps and the optimized Random Forest model, was serialized using `joblib.dump()` and saved as `Model_Pipeline.pkl`.
- **Purpose**: Saving the complete pipeline ensures that the model can be easily deployed and used for new, unseen data without needing to manually re-apply preprocessing steps, preventing inconsistencies between training and inference environments.
