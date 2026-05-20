# ✈️ Airline Ticket Price Prediction

> Predicting Indian domestic flight prices using regression models trained on airline, routing, timing, and booking window features.

---

## 🧩 Problem Statement

Airline ticket prices are notoriously volatile — the same seat can cost dramatically different amounts depending on when you book, which airline you choose, what time you depart, and how long the flight is. Travelers are left guessing whether now is the right time to buy.

This project asks: **can we build a model that accurately predicts ticket prices using flight and booking features — and reveal which factors matter most?**

---

## 🗂️ Dataset

Three CSV files covering Indian domestic flights:

| File | Contents |
|---|---|
| `economy.csv` | Economy class flight records |
| `business.csv` | Business class flight records |
| `Clean_Dataset.csv` | Pre-cleaned dataset including `days_left` before departure |

The economy and business datasets were combined into a single `flights` dataframe, then merged with `Clean_Dataset.csv` to bring in the `days_left` booking window feature — the most critical predictor of price.

---

## 🔧 Approach

### Data Preparation

- Added a `class` column (`Economy` / `Business`) before concatenating the two datasets
- Converted `price` from a comma-formatted string to integer
- Converted `time_taken` from `"Xh Ym"` format to total minutes
- Parsed `dep_time` and `arr_time` into time objects; created `dep_float` (departure as decimal hour, e.g. 14.5 = 2:30 PM)
- Created `flight_id` by combining `ch_code` and `num_code` for merging
- Merged `flights` with `Clean_Dataset.csv` using a rank-based strategy to handle duplicate flight records correctly

### Feature Engineering

| Feature | Type | Description |
|---|---|---|
| `days_left` | Numeric | Days between booking and departure |
| `time_taken` | Numeric | Flight duration in minutes |
| `dep_hour_sin` | Numeric | Cyclical encoding of departure hour (sin) |
| `dep_hour_cos` | Numeric | Cyclical encoding of departure hour (cos) |
| `day_of_week` | Numeric | Day of week (0 = Monday, 6 = Sunday) |
| `airline` | Categorical (OHE) | Airline carrier |
| `class` | Categorical (OHE) | Economy or Business |
| `stop` | Categorical (OHE) | Number of stops |
| `from` | Categorical (OHE) | Departure city |
| `to` | Categorical (OHE) | Arrival city |

> Departure time was cyclically encoded (sin/cos) so the model understands that 11:59 PM and 12:00 AM are close together — a key improvement over raw hour encoding.

### Preprocessing Pipeline

- `OneHotEncoder` for all categorical columns
- `StandardScaler` for all numerical columns
- Wrapped in a `sklearn Pipeline` with `ColumnTransformer` for clean, leak-free preprocessing

### Models Evaluated

| Model | Notes |
|---|---|
| `LinearRegression` / `LogisticRegression` | Baseline linear model |
| `RandomForestRegressor` | 100 estimators, `random_state=42` |
| `GradientBoostingRegressor` | Sequential boosting baseline |

Evaluated on **RMSE**, **MSE**, and **R² score** on a held-out 20% test set.

---

## 📊 Key Findings

- **`days_left` is the strongest predictor** — prices rise sharply as departure approaches, confirming the classic book-early wisdom
- **Business class prices are dramatically higher** than economy across all airlines, and are more stable over time
- **Departure time affects price** — flights at certain hours (early morning, late night) tend to be cheaper
- **Flight duration correlates with price** — longer routes cost more, but the relationship is non-linear
- **Airline carrier is a significant factor** — premium carriers (e.g., Vistara) price substantially above budget carriers (e.g., AirAsia)
- The best model's **predicted vs. actual price curve** closely tracks real prices across the `days_left` booking window, with tightest fit in the 30–60 day range

> Run the notebook to see the exact RMSE and R² scores for each model — results are printed in the `results_df` table.

---

## 🗃️ Repository Structure

```
PredictAirlineTicketPrices/
├── code.ipynb          # Full pipeline: EDA, feature engineering, modeling, evaluation
├── business.csv        # Business class flight data
├── economy.csv         # Economy class flight data
└── Clean_Dataset.csv   # Pre-cleaned dataset with days_left feature
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/alexjbyoon/PredictAirlineTicketPrices.git
cd PredictAirlineTicketPrices

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook code.ipynb
```

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations and cyclical encoding |
| `matplotlib` / `seaborn` | EDA visualizations and result plots |
| `scikit-learn` | Preprocessing, pipelines, models, and metrics |
| `jupyter` | Notebook environment |

Python 3.8+ recommended.
