import streamlit as st
import joblib
import pandas as pd

# Load the trained model and team stats
model = joblib.load('epl_model.pkl')
latest_stats = pd.read_csv('latest_team_stats.csv')

st.title('EPL Match Outcome Predictor')

teams = sorted(latest_stats['Team'].unique())

home_team = st.selectbox('Home Team', teams)
away_team = st.selectbox('Away Team', teams)

if st.button('Predict'):
    home_stats = latest_stats[latest_stats['Team'] == home_team].iloc[0]
    away_stats = latest_stats[latest_stats['Team'] == away_team].iloc[0]

    form_diff = abs(home_stats['Form_Last5'] - away_stats['Form_Last5'])
    goal_diff_form = abs(home_stats['AvgGoalsFor_Last5'] - away_stats['AvgGoalsFor_Last5'])

    features = pd.DataFrame([{
        'HomeForm': home_stats['Form_Last5'],
        'AwayForm': away_stats['Form_Last5'],
        'HomeAvgGoalsFor': home_stats['AvgGoalsFor_Last5'],
        'AwayAvgGoalsFor': away_stats['AvgGoalsFor_Last5'],
        'HomeAvgGoalsAgainst': home_stats['AvgGoalsAgainst_Last5'],
        'AwayAvgGoalsAgainst': away_stats['AvgGoalsAgainst_Last5'],
        'FormDiff': form_diff,
        'GoalDiffForm': goal_diff_form
    }])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    result_map = {'H': f'{home_team} wins', 'A': f'{away_team} wins', 'D': 'Draw'}
    st.subheader(result_map[prediction])

    # Show confidence for the predicted outcome
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100
    st.write(f"Confidence: {confidence:.1f}%")

    # Show full probability breakdown
    st.write("Full breakdown:")
    prob_df = pd.DataFrame({
        'Outcome': [result_map.get(c, c) for c in model.classes_],
        'Probability': [f"{p*100:.1f}%" for p in probabilities]
    })
    st.table(prob_df)

    # Show the stats behind the prediction
    st.write("---")
    st.write("**Stats used for this prediction:**")
    stats_df = pd.DataFrame({
        'Stat': ['Recent Form (pts/game)', 'Avg Goals Scored', 'Avg Goals Conceded'],
        home_team: [home_stats['Form_Last5'], home_stats['AvgGoalsFor_Last5'], home_stats['AvgGoalsAgainst_Last5']],
        away_team: [away_stats['Form_Last5'], away_stats['AvgGoalsFor_Last5'], away_stats['AvgGoalsAgainst_Last5']]
    })
    st.table(stats_df)