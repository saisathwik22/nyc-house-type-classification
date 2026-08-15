# 📘 Project Reference: NYC Airbnb Room Type Predictor

*A complete personal reference document — project rationale, technical decisions, metrics, and talking points for future recall, interviews, or resume/portfolio updates.*

---

## 1. Project Summary

**What it is**: A machine learning classification project that predicts the room type of a New York City Airbnb listing — *Entire home/apt*, *Private room*, or *Shared room* — using listing metadata (location, price, host activity, availability). The trained model is deployed as a live web app with a FastAPI backend and a vanilla JS frontend.

**Why it matters (the pitch)**: This project demonstrates the full ML lifecycle end-to-end — from raw, messy real-world data through EDA, cleaning, feature engineering, handling class imbalance, model comparison, hyperparameter tuning, and finally shipping the model behind a production API with a working UI. It's not just a notebook — it's a deployed product.

**Live demo**: https://nyc-airbnb-room-type-predictor.onrender.com

---

## 2. Dataset

- **Source**: NYC Airbnb Open Data (`dgomonov/new-york-city-airbnb-open-data` on Kaggle), loaded via `kagglehub`.
- **Size**: 48,895 listings × 16 columns.
- **Target variable**: `room_type` — 3 classes.
- **Class distribution** (this is the single most important fact about this dataset — it drove almost every downstream decision):

  | Room Type | Count | Share |
  |---|---|---|
  | Entire home/apt | 25,409 | ~52.0% |
  | Private room | 22,326 | ~45.7% |
  | Shared room | 1,160 | ~2.4% |

- **Missing values found**: `name` (16), `host_name` (21), `last_review` (10,052), `reviews_per_month` (10,052).

---

## 3. Exploratory Data Analysis — Key Findings

- **Skewness**: every core numerical feature was right-skewed:
  - `price`: 2.78
  - `minimum_nights`: 2.36
  - `number_of_reviews`: 3.69
  - `reviews_per_month`: 3.30
  - `calculated_host_listings_count`: 7.93 (most skewed — a small number of hosts manage many listings)
  - `availability_365`: 0.76 (least skewed)

  → This is *why* `StandardScaler` and outlier capping were necessary — skewed, unbounded numerical features can dominate distance/gradient-based models if left unscaled.

- **Room type vs. price**: box plots showed clear price differences across room types, with `Entire home/apt` priced highest on average and both room types showing significant outliers on the high end.

- **Correlation heatmap**: no strong multicollinearity between numerical features. Notable: `reviews_per_month` and `number_of_reviews` correlated at 0.55 (makes sense — both track review activity), but the other pairwise correlations were weak. This meant no features needed to be dropped purely for collinearity.

- **Geographic scatter (lat/lon by room type)**: revealed spatial clustering — certain neighbourhoods/boroughs skew heavily toward one room type over another (e.g., dense Manhattan areas trend toward entire homes / private rooms, with shared rooms sparse and scattered). This is part of why `latitude`/`longitude` and `neighbourhood_group`/`neighbourhood` turned out to be useful predictive features rather than being dropped.

---

## 4. Data Cleaning & Feature Engineering — Decisions & Reasoning

| Decision | Reasoning |
|---|---|
| Dropped `id`, `name`, `host_id`, `host_name`, `last_review` | Unique identifiers or free text — no generalizable signal for predicting room type across *new* listings. Keeping them risks overfitting to noise (e.g. specific host IDs) rather than learning real patterns. |
| Imputed `reviews_per_month` missing values with 0 | Missingness here directly corresponds to "no reviews yet" — 0 is the semantically correct fill, not just a statistical placeholder. |
| Capped `price` and `minimum_nights` at the 99th percentile (`.clip`) | These fields had extreme outliers, most likely data-entry errors (e.g. $10,000/night listings, 1000-night minimums) rather than real signal. Capping (not deleting) preserves the row while preventing outliers from disproportionately skewing the trained model. |

**Final feature set (10 features)**:
`latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`, `neighbourhood_group`, `neighbourhood`

---

## 5. Train/Test Split

```python
train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
```

- **33% held out for testing** (16,136 listings).
- **`stratify=y`** — this is the detail worth remembering and explaining: without stratification, a random split could easily produce a test set with very few or zero `Shared room` examples (since it's only ~2.4% of the data), making evaluation unreliable. Stratifying preserves the same class proportions in both splits.

---

## 6. Preprocessing Pipeline

Built with `ColumnTransformer` inside a scikit-learn `Pipeline` — this was a deliberate architecture choice to prevent **data leakage**: every transformation (imputation, scaling, encoding) is *fit* only on training data, then *applied* to test/inference data, never the reverse.

| Feature type | Columns | Steps |
|---|---|---|
| Numerical | `latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365` | `SimpleImputer(strategy='median')` → `StandardScaler()` |
| Categorical | `neighbourhood_group`, `neighbourhood` | `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')` |

**Why median (not mean) for numerical imputation**: the features are skewed (see EDA section) — median is more robust to outliers/skew than mean.

**Why `handle_unknown='ignore'` on the encoder**: `neighbourhood` has many unique values; this prevents the pipeline from crashing if it encounters a neighbourhood at inference time that wasn't seen during training — it just zeroes out that feature instead of throwing an error. Important for production robustness.

---

## 7. Model Comparison

Four classifiers, each wrapped in the same pipeline with the shared preprocessor, evaluated via **3-fold stratified cross-validation** on the training data.

**Class imbalance handling**: `class_weight="balanced"` applied to Logistic Regression, Decision Tree, and Random Forest. **Gradient Boosting does not support `class_weight`** in scikit-learn, so it was evaluated without it — a real constraint worth remembering if asked about it.

**Why macro F1 as the primary metric, not accuracy**: accuracy is misleading on imbalanced data — a model that never predicts `Shared room` at all could still score ~85%+ accuracy just by defaulting to the majority classes. Macro F1 computes F1 per class and averages them *unweighted*, so poor performance on the rare class pulls the score down just as much as poor performance on a common class. This is the correct metric when all classes matter regardless of frequency.

| Model | Accuracy | Macro F1 | `class_weight` used? |
|---|---|---|---|
| Logistic Regression | 65.9% | 52.2% | Yes |
| Decision Tree | 78.2% | 64.7% | Yes |
| **Random Forest** | **85.1%** | **71.5%** | Yes |
| Gradient Boosting | 85.0% | 70.5% | No (unsupported) |

**Result**: Random Forest and Gradient Boosting were close on accuracy, but Random Forest had a clear edge on macro F1 (71.5% vs 70.5%) — meaning it handled the minority `Shared room` class noticeably better. Random Forest was selected for tuning.

---

## 8. Hyperparameter Tuning

- **Method**: `RandomizedSearchCV` (not `GridSearchCV`) — chosen because it samples a fixed number of parameter combinations rather than exhaustively testing every combination, which is far cheaper computationally while still exploring the space effectively. Good default choice when the grid is even moderately large.
- **`cv=3`**, optimizing for `f1_macro`.

**Search space**:
```python
param_distribution = {
    "classifier__n_estimators": [100, 150, 200, 300],
    "classifier__max_depth": [8, 12, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
}
```

**Best parameters found**:
```python
{'n_estimators': 200, 'max_depth': None, 'min_samples_split': 10}
```

**Best cross-validated macro F1**: **73.06%**

*Interesting note*: `max_depth=None` (unlimited depth) won out — combined with `min_samples_split=10` acting as the regularizer instead of a depth cap. Worth remembering if asked "how did you control overfitting" — the answer here is via `min_samples_split`, not depth limiting.

---

## 9. Final Model Evaluation (Held-Out Test Set)

| Metric | Score |
|---|---|
| Accuracy | **85.43%** |
| Macro F1 | **73.42%** |

- Test macro F1 (73.42%) is very close to the cross-validated macro F1 (73.06%) — a good sign that the model generalizes well and isn't overfit to the training folds.
- A confusion matrix was generated (with `xticklabels`/`yticklabels` set to actual class names) to inspect where errors concentrate. The expected weak point, based on the imbalance, is confusion between `Private room` and `Shared room` (both lower-availability categories that can look similar on price/features), which is the natural place to investigate first if asked "where does the model struggle?"

---

## 10. Model Serialization & Deployment

- **Serialization**: `joblib.dump(best_pipeline, "Model_Pipeline.pkl")` — saves the *entire* pipeline (preprocessing + trained model) as one object, not just the raw classifier.
  - **Why this matters**: at inference time, the API just loads this one file and calls `.predict()` / `.predict_proba()` directly on raw-ish input — no need to manually re-implement imputation/scaling/encoding logic in the API layer, which eliminates a whole class of train/serve skew bugs.

- **Backend**: FastAPI app exposes a `POST /predict` endpoint. Request body validated via Pydantic (type checks, range constraints on lat/lon/price/etc.). pandas builds a single-row DataFrame from the validated input, which is passed straight into the loaded pipeline.

- **Response shape**:
```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.85, 0.12, 0.03]
}
```
Returns the full probability distribution across all 3 classes, not just the top prediction — deliberately chosen so the frontend can show confidence, not just a flat label.

- **Frontend**: plain HTML/CSS/JS (no framework) — form for input, fetch call to the API, animated visualization of the probability distribution.

- **Deployment**: Render.com, backend and frontend both static/served separately.

---

## 11. Tech Stack Reference

| Layer | Tools |
|---|---|
| Data acquisition | kagglehub |
| Data handling | pandas |
| EDA / visualization | matplotlib, seaborn |
| Modeling | scikit-learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `RandomizedSearchCV`) |
| Serialization | joblib |
| Backend | FastAPI, Pydantic, uvicorn |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Deployment | Render.com |

---

## 12. Likely Interview Questions & Prepared Answers

**Q: Why macro F1 instead of accuracy?**
A: The dataset is imbalanced (~2.4% Shared room). Accuracy can look deceptively high while completely ignoring the minority class. Macro F1 averages per-class F1 scores unweighted, so the rare class's performance matters just as much as the common classes'.

**Q: How did you handle class imbalance?**
A: Two layers — `stratify=y` in the train/test split to preserve class ratios in both sets, and `class_weight="balanced"` in the models that support it, which upweights the loss contribution of minority-class errors during training.

**Q: Why Random Forest over Gradient Boosting, when accuracy was almost identical?**
A: Macro F1 was the deciding metric (71.5% vs 70.5%), and Random Forest also natively supports `class_weight`, which Gradient Boosting doesn't in scikit-learn — giving it a more principled way to handle the imbalance.

**Q: Why RandomizedSearchCV instead of GridSearchCV?**
A: The parameter space (4 × 5 × 3 = 60 combinations) was moderate-large; RandomizedSearchCV samples a fixed budget of combinations, giving good coverage of the space at a fraction of the compute cost of exhaustive search.

**Q: How do you prevent data leakage in your pipeline?**
A: All preprocessing (imputation, scaling, encoding) is wrapped inside a scikit-learn `Pipeline`/`ColumnTransformer`, fit only on the training split. This guarantees test-time statistics (medians, categories) never leak into training.

**Q: What would you improve if you had more time?**
Reasonable, honest talking points:
- Try SMOTE or other resampling techniques for the minority class, in addition to `class_weight`.
- Add more engineered features (e.g. distance to city center, review recency).
- Try feature importance analysis (`feature_importances_` from Random Forest) to explain predictions.
- Set up proper model monitoring/versioning for the deployed pipeline.
- Expand hyperparameter search (`n_iter`, wider grid) given more compute budget.

---

## 13. Resume Bullet (Reference Copy)

**NYC Airbnb Room Type Predictor** | Python, scikit-learn, FastAPI, pandas, JavaScript

- Built an end-to-end ML classification pipeline on 48,895 NYC Airbnb listings to predict room type (Entire home/apt, Private room, Shared room), handling significant class imbalance (Shared room ~2.4% of listings) using stratified splits and `class_weight="balanced"`
- Performed EDA (distribution analysis, correlation heatmaps, geographic scatter plots) and feature engineering — dropped non-predictive identifier columns, imputed missing values, and capped outliers at the 99th percentile
- Built a leak-proof preprocessing pipeline with `ColumnTransformer` (median imputation + standardization for numerical features, most-frequent imputation + one-hot encoding for categorical features)
- Benchmarked 4 classifiers via 3-fold stratified cross-validation on macro F1: Logistic Regression (52.2%), Decision Tree (64.7%), Random Forest (71.5%), Gradient Boosting (70.5%) — selected Random Forest as the top performer
- Tuned Random Forest via `RandomizedSearchCV`, improving cross-validated macro F1 to 73.1%; final model achieved 85.4% accuracy and 73.4% macro F1 on the held-out test set
- Serialized the complete pipeline with `joblib` and deployed it behind a FastAPI backend with a vanilla JS/HTML/CSS frontend for real-time predictions with class probabilities

---

## 14. Links

- **GitHub repo**: *(add your link here)*
- **Live demo**: https://nyc-airbnb-room-type-predictor.onrender.com
- **Dataset**: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

---

*Last updated: 2026-08-15*
