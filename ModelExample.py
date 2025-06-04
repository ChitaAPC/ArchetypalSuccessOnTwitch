import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub
import concurrent.futures
from KMaxoids.TimeSeriesKMaxoids import TimeSeriesKMaxoids
from pickle import load

#%% Load Scaler and K-Maxoids Model
model = TimeSeriesKMaxoids.load_model("model.pkl")
scaler = load(open('scaler.pkl', 'rb'))

#%% Load original data from Kaggle
dataset_path = kagglehub.dataset_download("rankirsh/evolution-of-top-games-on-twitch")
raw_df = pd.read_csv(f'{dataset_path}/Twitch_game_data.csv', encoding="ISO-8859-1")

#%% Pre-process data for model
# Process dataset
df = raw_df.copy()
df['Game'] = df['Game'].str.lower()
df['Month_since_start'] = ((df['Year'] - 2016) * 12) + df['Month']
df['Rank'] = 201 - df['Rank']  # Flip rank so 1 becomes 200 and 200 becomes 1

# Remove NaN values in 'Game' column
df = df.dropna(subset=['Game'])

# Define features to extract
columns = ['Rank', 'Hours_streamed', 'Hours_watched', 'Peak_viewers', 'Peak_channels', 'Streamers', 'Avg_viewers',
           'Avg_channels', 'Avg_viewer_ratio']
num_months = 12

def process_game_series(game):
    game_df = df[df['Game'] == game].sort_values(by='Month_since_start')
    if game_df.empty:
        return [], []

    series_list = []
    start_idx = 0
    start_months = []

    while start_idx < len(game_df):
        first_month = game_df['Month_since_start'].iloc[start_idx]
        start_months.append(first_month)
        subset = game_df[
            (game_df['Month_since_start'] >= first_month) & (game_df['Month_since_start'] < first_month + num_months)]

        if subset.empty:
            start_idx += 1
            continue

        series_data = {col: np.zeros(num_months) for col in columns}
        indices = subset['Month_since_start'] - first_month

        for col in columns:
            series_data[col][indices] = subset[col].values

        series_list.append(np.vstack([series_data[col] for col in columns]))

        start_idx += len(subset)  # Move to next appearance

    return series_list, start_months


# Process games in parallel
game_series = {}
game_starts = {}
with concurrent.futures.ThreadPoolExecutor() as executor:
    results, start_months_results = zip(*executor.map(process_game_series, df['Game'].unique()))

for game, series, month_series in zip(df['Game'].unique(), results, start_months_results):
    if series:
        game_series[game] = series
        game_starts[game] = month_series

# Convert dictionary to DataFrame
game_entries = []
game_labels = []
for game, series_list in game_series.items():
    for series in series_list:
        game_entries.append(series.T.flatten())
        game_labels.append(game)

games_df = pd.DataFrame(game_entries)
games_df.index = game_labels

scaled_data = scaler.transform(games_df)

#%% Predict data using model
y_pred = model.predict(scaled_data)

#%% Plot game clusters example
def flatten(xss):
    return [x for xs in xss for x in xs]

def plot_game_clusters(game_name, y_pred):
    cluster_labels = ["Supernova", "Short-Lived Fame", "Replay Royalty", "Esports Giants", "Momentum Builders",
                      "Roller Coasters", "Wildcards"]
    plt.figure(figsize=(12, 6))
    month_indexes = flatten(start_months_results)
    total_months = set()
    if game_name not in game_labels:
        print(f"Game '{game_name}' not found in dataset.")
        return

    indices = [i for i, label in enumerate(game_labels) if label == game_name]
    cluster_assignments = [y_pred[i] for i in indices]
    months = [int(month_indexes[i]) for i in indices]

    total_months.update(months)

    plt.plot(months, cluster_assignments, marker='o', linestyle='-', label=game_name)

    plt.xlabel("Time Period (Months/Years)")
    total_months = sorted(total_months)

    tick_labels = [f"{(2016 + (m // 12))}-{(m % 12) + 1:02d}" for m in total_months]
    plt.xticks(total_months, tick_labels, rotation=45)
    plt.yticks(range(len(cluster_labels)), cluster_labels)
    plt.title(f"\"{game_name}\" Archetypes Over Time")
    plt.show()


# Call functions to plot
plot_game_clusters("slay the spire", y_pred)