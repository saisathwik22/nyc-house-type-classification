# 🏙️ NYC Airbnb Room Type Predictor

A machine learning application that predicts the room type of a New York City Airbnb listing — **Entire home/apt**, **Private room**, or **Shared room** — based on listing characteristics like location, price, and host activity. The trained model is served through a FastAPI backend with a web interface for real-time predictions.

🔗 **Live Demo:** https://nyc-house-type-classification-1.onrender.com/

---

## 📌 What It Does

- **Input**: The user provides key listing details — location (latitude/longitude), price per night, minimum nights required, number of reviews, reviews per month, number of other listings the host manages, yearly availability, and the NYC borough/neighbourhood.
- **Prediction**: The model classifies the listing into one of three room types — **Entire home/apt**, **Private room**, or **Shared room** — based on patterns learned from real NYC Airbnb data.
- **Confidence scores**: Instead of a single flat answer, the app returns probability scores for all three room types, showing how confident the model is rather than just its top guess.
- **Practical use cases**: auto-categorizing listings with missing labels, flagging inconsistencies (e.g. a listing marked "entire home" that behaves statistically like a shared room), or helping analysts understand what features drive how a listing is structured.

---

## 🧠 The Machine Learning Pipeline

### 1. Data Loading & Initial Exploration
- Dataset: **NYC Airbnb Open Data** (48,895 listings, 16 columns), loaded via `kagglehub` into a Pandas DataFrame.
- Initial inspection with `df.head()`, `df.info()`, `df.describe()`, `df.shape`, and `df.isnull().sum()`.
- Missing values found in `name` (16), `host_name` (21), `last_review` (10,052), and `reviews_per_month` (10,052).
- Target variable (`room_type`) distribution:

  | Room Type | Count | Share |
  |---|---|---|
  | Entire home/apt | 25,409 | ~52.0% |
  | Private room | 22,326 | ~45.7% |
  | Shared room | 1,160 | ~2.4% |

  This confirmed a **significant class imbalance**, with `Shared room` as a clear minority class — a key factor shaping every later modeling decision.

### 2. Exploratory Data Analysis (EDA)
- **Numerical distributions**: histograms for `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, and `availability_365`.
- **Skewness check**: all core numerical features were right-skewed — `price` (2.78), `minimum_nights` (2.36), `number_of_reviews` (3.69), `reviews_per_month` (3.30), `calculated_host_listings_count` (7.93), `availability_365` (0.76) — confirming the need for outlier handling and scaling.
- **Categorical distribution**: count plot of `neighbourhood_group` across NYC boroughs.
- **Room type vs. price**: box plot to compare pricing across room types and surface outliers.
- **Correlation analysis**: heatmap of numerical features, e.g. `reviews_per_month` correlated moderately with `number_of_reviews` (0.55), while most other pairwise correlations were weak.
- **Geographical distribution**: scatter plots of longitude vs. latitude, colored by room type (with alpha/size tuning for readability), to visualize spatial concentration patterns across boroughs.

### 3. Data Cleaning & Feature Engineering
- Dropped identifier/free-text columns with no generalizable predictive value: `id`, `name`, `host_id`, `host_name`, `last_review`.
- Imputed missing `reviews_per_month` values with 0 (listings with no reviews assumed to have no review rate).
- Capped extreme outliers in `price` and `minimum_nights` at the **99th percentile** using `.clip(upper=quantile(0.99))`, reducing the influence of likely data-entry errors while preserving the overall distribution shape.
- Split into features (`X`) and target (`y = room_type`).

### 4. Train/Test Split
- `train_test_split` with `test_size=0.33`, `random_state=42`.
- `stratify=y` used to preserve the ~52/46/2 class proportions in both sets — critical given the imbalance.

### 5. Preprocessing Pipeline (`ColumnTransformer`)
Built as a single pipeline so all transformations are learned only from training data (no leakage):

| Feature type | Steps |
|---|---|
| Numerical (`latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`) | `SimpleImputer(strategy='median')` → `StandardScaler()` |
| Categorical (`neighbourhood_group`, `neighbourhood`) | `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')` |

### 6. Model Comparison
Four classifiers were evaluated, each wrapped in a full pipeline with the shared preprocessor. `class_weight="balanced"` was applied where supported (Gradient Boosting doesn't support it, so it was left as-is) to counter the minority-class imbalance. Models were compared via **3-fold stratified cross-validation**, scored on both accuracy and **macro F1** (macro F1 weighs all three classes equally, which matters far more than accuracy here given how rare `Shared room` is):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 65.9% | 52.2% |
| Decision Tree | 78.2% | 64.7% |
| **Random Forest** | **85.1%** | **71.5%** |
| Gradient Boosting | 85.0% | 70.5% |

**Random Forest** was selected as the best-performing model, narrowly ahead of Gradient Boosting on macro F1.

### 7. Hyperparameter Tuning
- Random Forest tuned using `RandomizedSearchCV` (`n_iter` search over the grid below, `cv=3`), optimizing for `f1_macro`.
- Search space:
  - `n_estimators`: 100, 150, 200, 300
  - `max_depth`: 8, 12, 15, 20, or unlimited (`None`)
  - `min_samples_split`: 2, 5, 10
- **Best parameters found**: `n_estimators=200`, `max_depth=None`, `min_samples_split=10`
- **Best cross-validated macro F1**: **73.06%**

### 8. Final Model Evaluation
Evaluated on the held-out test set (16,136 listings):

| Metric | Score |
|---|---|
| Accuracy | **85.43%** |
| Macro F1 | **73.42%** |

A confusion matrix (with class labels) was generated to inspect per-class performance in detail, with particular attention to how well the model distinguishes the minority `Shared room` class from `Private room`.

### 9. Model Saving
- The full pipeline — preprocessing + tuned Random Forest — serialized with `joblib.dump()` as `Model_Pipeline.pkl`.
- Saving the entire pipeline (not just the raw model) ensures preprocessing is applied identically at inference time, avoiding train/serve skew.

---

## 🔌 Model in Production

### Input Features (10)

| Feature | Type | Constraints |
|---|---|---|
| `latitude` | float | -90 to 90 |
| `longitude` | float | -180 to 180 |
| `price` | float | > 0 |
| `minimum_nights` | int | 1–365 |
| `number_of_reviews` | int | ≥ 0 |
| `reviews_per_month` | float | ≥ 0 |
| `calculated_host_listings_count` | int | ≥ 0 |
| `availability_365` | int | 0–365 |
| `neighbourhood_group` | str | NYC borough |
| `neighbourhood` | str | Specific neighbourhood |

### `POST /predict`
**Request:**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "price": 150,
  "minimum_nights": 3,
  "number_of_reviews": 24,
  "reviews_per_month": 1.2,
  "calculated_host_listings_count": 1,
  "availability_365": 180,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}
```

**Response:**
```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.85, 0.12, 0.03]
}
```

Input validation is handled by Pydantic (type checks, range constraints, non-empty strings), returning a `422` with details on invalid input.

---

## 🏗️ Architecture

- **Backend** — FastAPI serves the pipeline via a `/predict` endpoint; pandas builds the input row, and the saved scikit-learn pipeline (loaded with `joblib`) handles preprocessing and inference in one step. CORS enabled for cross-origin frontend requests.
- **Frontend** — Vanilla HTML/CSS/JS interface (no framework) with real-time API health checks, form validation, and a visual probability display.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier, RandomizedSearchCV) |
| Data handling | pandas, kagglehub |
| Visualization (EDA) | matplotlib, seaborn |
| Serialization | joblib |
| Backend | FastAPI, Pydantic, uvicorn |
| Frontend | HTML5, CSS3, vanilla JavaScript |

---

## 🚀 Setup & Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the backend
```bash
uvicorn main:app --reload
```
- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

### Run the frontend
Open `index.html` directly, or serve it locally:
```bash
python -m http.server 8080
```

---

## 📁 Project Structure

```
house_types_classification/
├── main.py             # FastAPI backend serving the ML pipeline
├── index.html           # Frontend structure
├── script.js             # Frontend logic & API calls
├── style.css              # Styling
├── requirements.txt
├── Model_Pipeline.pkl   # Serialized preprocessing + trained Random Forest pipeline
└── README.md
```

---

## 📝 License

Provided as-is for educational purposes.
