# 🏙️ NYC Airbnb Room Type Predictor

A machine learning application that predicts the room type of a New York City Airbnb listing — **Entire home/apt**, **Private room**, or **Shared room** — based on listing characteristics like location, price, and host activity. The trained model is served through a FastAPI backend with a web interface for real-time predictions.

🔗 **Live Demo:** https://nyc-house-type-classification-1.onrender.com/

---

## 📌 What It Does
- Input: The user provides key listing details — location (latitude/longitude), price per night, minimum nights required, number of reviews, reviews per month, number of other listings the host manages, yearly availability, and the NYC borough/neighbourhood.

- Prediction: The model classifies the listing into one of three room types — Entire home/apt, Private room, or Shared room — based on patterns learned from real NYC Airbnb data.

- Why it works: Each room type tends to have a distinct signature. Entire homes are usually priced higher and often belong to hosts managing multiple listings; shared rooms are typically cheaper and far less common. The model picks up on these correlations across price, location, and host behavior to make its prediction.

- Confidence scores: Instead of returning a single flat answer, the app shows probability scores for all three room types — so the result reflects how confident the model is, not just its top guess. This makes the output more transparent and trustworthy than a plain black-box label.

- Practical use cases:
  - Auto-categorizing listings with missing or unclear room type labels
  - Flagging inconsistencies — e.g. a listing marked "entire home" that behaves statistically like a shared room
  - Helping hosts, platforms, or analysts understand which features (price, location, host activity) most influence how a listing is structured

---

## 🧠 The Machine Learning Pipeline

### 1. Data Loading & Initial Exploration
- Dataset: NYC Airbnb listings, loaded via `kagglehub` into a Pandas DataFrame.
- Initial inspection with `df.head()`, `df.info()`, `df.describe()`, `df.shape`, and `df.isnull().sum()` to understand structure, types, and missing values.
- Target variable (`room_type`) checked for class distribution — revealed **class imbalance**, with `Shared room` as a clear minority class.

### 2. Exploratory Data Analysis (EDA)
- **Numerical distributions**: histograms for `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, and `availability_365` to check skew and outliers.
- **Categorical distribution**: count plot of `neighbourhood_group` across NYC boroughs.
- **Room type vs. price**: box plot to compare pricing across room types and surface outliers.
- **Correlation analysis**: heatmap of numerical features including latitude/longitude.
- **Geographical distribution**: scatter plot of longitude vs. latitude, colored by room type, to visualize spatial concentration patterns across the city.

### 3. Data Cleaning & Feature Engineering
- Dropped identifier/free-text columns with no generalizable predictive value: `id`, `name`, `host_id`, `host_name`, `last_review`.
- Imputed missing `reviews_per_month` values with 0 (listings with no reviews assumed to have no review rate).
- Capped extreme outliers in `price` and `minimum_nights` at the 99th percentile using `.clip(upper=quantile(0.99))`, reducing the influence of likely data-entry errors while preserving overall distribution shape.
- Split into features (`X`) and target (`y = room_type`).

### 4. Train/Test Split
- `train_test_split` with `test_size=0.33`, `random_state=42`.
- `stratify=y` used to preserve class proportions in both sets — important given the class imbalance.

### 5. Preprocessing Pipeline (`ColumnTransformer`)
Built as a single leak-proof pipeline so all transformations are learned only from training data:

| Feature type | Steps |
|---|---|
| Numerical (`latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`) | `SimpleImputer(strategy='median')` → `StandardScaler()` |
| Categorical (`neighbourhood_group`, `neighbourhood`) | `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')` |

### 6. Model Comparison
Four classifiers evaluated, each wrapped in a full pipeline with the preprocessor:

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree Classifier | Non-linear, prone to overfitting |
| Random Forest Classifier | Ensemble method |
| Gradient Boosting Classifier | Ensemble method |

- `class_weight="balanced"` applied where supported to counter the minority-class imbalance.
- Evaluated via **3-fold stratified cross-validation** on both `accuracy` and `f1_macro` (macro F1 treats all classes equally, which matters given the imbalance).
- **Random Forest and Gradient Boosting** produced the best macro F1 scores.

### 7. Hyperparameter Tuning
- **Random Forest Classifier** selected for tuning based on comparison results.
- Tuned with `RandomizedSearchCV` (`n_iter=10`), optimizing for `f1_macro`.
- Parameters searched:
  - `n_estimators`: 100, 150, 200, 300
  - `max_depth`: 8, 12, 15, 20, or unlimited
  - `min_samples_split`: 2, 5, 10

### 8. Final Model Evaluation
- Best pipeline evaluated on the held-out test set (`X_test`) using `accuracy_score` and `f1_score(average='macro')`.
- Confusion matrix generated and visualized to inspect per-class performance, with particular attention to the minority `Shared room` class.

### 9. Model Saving
- The full pipeline — preprocessing + tuned Random Forest — serialized with `joblib.dump()` as `Model_Pipeline.pkl`.
- Saving the entire pipeline (not just the model) ensures preprocessing is applied identically at inference time, avoiding train/serve skew.

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
