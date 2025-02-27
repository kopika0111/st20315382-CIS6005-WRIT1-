from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Team ID ranges (from your dataset)
team_range_1 = list(range(1101, 1481))  # Teams 1101 to 1480
team_range_2 = list(range(3101, 3481))  # Teams 3101 to 3480
all_teams = team_range_1 + team_range_2

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the user inputs for Team 1 and Team 2
        team1 = int(request.form["team1"])
        team2 = int(request.form["team2"])

        # Prepare the input data for prediction (wteamid and lteamid)
        X = pd.DataFrame({
            "wteamid": [team1],
            "lteamid": [team2],
            "daynum": [50],  # Assuming a fixed value for now
            "numot": [0],  # Assuming no overtime for now
            "total_points": [140],  # Example total points
            "point_diff": [0]  # Neutral point difference
        })

        # Scale the features
        X[["daynum", "numot", "total_points", "point_diff"]] = scaler.transform(
            X[["daynum", "numot", "total_points", "point_diff"]]
        )

        # Make prediction (using probability)
        pred_prob = model.predict_proba(X)[:, 1]  # Probability of team1 winning

        # Determine predicted winner and loser based on probability
        if pred_prob > 0.5:
            predicted_winner = team1
            predicted_loser = team2
            winning_score = int(np.random.normal(80, 10))  # Random score around 80
            losing_score = int(np.random.normal(60, 10))  # Random score around 60
        else:
            predicted_winner = team2
            predicted_loser = team1
            winning_score = int(np.random.normal(80, 10))  # Random score around 80
            losing_score = int(np.random.normal(60, 10))  # Random score around 60

        # Calculate the prediction probability as percentage
        prediction_percentage = round(pred_prob[0] * 100, 2)

        return render_template("index.html", all_teams=all_teams,
                               team1=team1, team2=team2,
                               predicted_winner=predicted_winner,
                               predicted_loser=predicted_loser,
                               winning_score=winning_score,
                               losing_score=losing_score,
                               prediction_percentage=prediction_percentage)

    return render_template("index.html", all_teams=all_teams)

if __name__ == "__main__":
    app.run(debug=True)
