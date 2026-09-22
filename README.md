# EPL Match Outcome Prediction

A data science project analyzing 3 seasons (2022/23–2024/25) of English Premier 
League match data to uncover performance patterns and build a machine learning 
model that predicts match outcomes (Home Win / Draw / Away Win).

## Data

Match data sourced from [Kaggle: English Premier League Match Data 2000–2025](https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025), 
filtered to the last 3 seasons to reflect current team quality and playing styles.

## Key Findings

- **Home advantage is real** — home teams won roughly 150 more matches than away 
  teams across the period studied.
- **But it varies hugely by team** — e.g. Man City won ~77% of home games vs. 
  Ipswich's ~6%. Team quality drives outcomes far more than venue alone.
- **Team strength is consistent regardless of venue** — the same strong teams 
  (Man City, Liverpool, Arsenal) dominate both home and away, just at different rates.
- **Shots on target strongly correlates with winning** — teams that win a match 
  average ~3 more shots on target than their opponent.
- **Discipline (cards) has only a weak relationship** with match outcomes, 
  compared to shots on target or team form.

## Model

A Logistic Regression model was trained on pre-match features only (to avoid 
data leakage), including:
- Rolling recent form (points per game, last 5 matches)
- Rolling average goals scored/conceded (last 5 matches)
- Form similarity between the two teams (a proxy for how evenly matched they are)

**Results:**
- Accuracy: **51.4%** (vs. a baseline of 46.8% from always predicting "Home Win")
- The model reliably predicts home and away wins, but **fails to predict draws 
  entirely**. This is a known, documented challenge in soccer analytics — draws 
  don't have as distinct a statistical signature as decisive results.

## Limitations & Next Steps

- Draw prediction remains unsolved here; worth exploring other model types 
  (e.g. Random Forest) or features like head-to-head history.
- Limited to 3 seasons of data by design, to avoid outdated patterns from older 
  seasons affecting predictions.

## Tools Used

Python, pandas, scikit-learn, Jupyter Notebook

## How to Run

1. Clone this repo
2. Install dependencies: `pip install pandas scikit-learn jupyter`
3. Open `epl_match_prediction.ipynb` and run all cells


🔗 **[Try the live app](https://epl-match-prediction-model.streamlit.app/)**