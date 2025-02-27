import itertools
import pandas as pd
import pickle
import numpy as np

# Load trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define team ID ranges
team_range_1 = range(1101, 1481)  # Teams 1101 to 1480
team_range_2 = range(3101, 3481)  # Teams 3101 to 3480

# Generate all possible matchups within each group
matchups_1 = list(itertools.combinations(team_range_1, 2))
matchups_2 = list(itertools.combinations(team_range_2, 2))

# Combine all matchups
all_matchups = matchups_1 + matchups_2

# Create a DataFrame for the matchups
submission_df = pd.DataFrame(all_matchups, columns=["wteamid", "lteamid"])

# Add required features (assuming neutral values)
submission_df["daynum"] = 50  # Example fixed value, adjust as needed
submission_df["numot"] = 0
submission_df["total_points"] = 140  # Example total points, adjust if necessary
submission_df["point_diff"] = 0  # Neutral point difference

# Scale numerical features
submission_df[["daynum", "numot", "total_points", "point_diff"]] = scaler.transform(
    submission_df[["daynum", "numot", "total_points", "point_diff"]]
)

raw_probs = model.predict_proba(submission_df)
print(raw_probs[:10])  # Print first 10 predictions

print(model.classes_)  # Check class order

# Predict the probability of the first team winning
submission_df["Pred"] = model.predict_proba(submission_df)[:, np.argmax(model.classes_)]  # Pick highest class probability

# submission_df["Pred"] = (model.predict_proba(submission_df)[:, 1] - 0.5) * 2  # Scaled to [-1,1]
print(submission_df["Pred"])
# Create the ID column
submission_df["ID"] = submission_df.apply(lambda row: f"2025_{int(row['wteamid'])}_{int(row['lteamid'])}", axis=1)

# Select only the required columns
final_submission = submission_df[["ID", "Pred"]]

# Save to CSV
final_submission.to_csv("submission.csv", index=False)

print("✅ Submission file saved as submission.csv!")

