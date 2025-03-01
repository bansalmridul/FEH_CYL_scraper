import numpy as np
from scipy.optimize import minimize, Bounds
import pandas as pd
import itertools
import time


def least_squares_ratio(scores, matches, poll_results):
    least_squares = 0
    for pair, years in matches.items():
        i, j = pair
        expected = np.log(scores[i] / scores[j])
        r1_vals = poll_results[i, years].astype(float)
        r2_vals = poll_results[j, years]
        div_arr = np.divide(r1_vals, r2_vals).astype(float)
        squared_diffs = (np.log(div_arr) - expected) ** 2
        least_squares += squared_diffs.sum() / len(years)
    return least_squares


def least_squares_diff(scores, matches):
    return sum([((diffs - (scores[i] - scores[j])) ** 2).sum() / len(diffs) for (i, j), diffs in matches.items()])

def main(mode="diff"):
    df = pd.read_csv('ranked_popularity_elo.csv', header = 0)
    #print(df)
    poll_results = df.to_numpy()[:,3:] #only get year numbers
    num_players = 50 #len(df)
    poll_results = poll_results[:num_players]
    matches = dict()
    print(poll_results.shape)
    for i, j in itertools.combinations(range(num_players), 2):
        diffs = [poll_results[i][k] - poll_results[j][k] for k in range(9) if not (np.isnan(poll_results[i][k]) or np.isnan(poll_results[j][k]))]
        if len(diffs) == 0:
            continue
        matches[(i, j)] = np.array(diffs)
    print(f"got {len(matches)} matches")
    #print(matches)
    if mode == "diff":
        initial_params = np.zeros(num_players)
        result = minimize(least_squares_diff, initial_params, args=(matches), method='BFGS')
    if mode == "ratio":
        result = minimize(least_squares_ratio, initial_params, args=(matches, poll_results), method='BFGS')
    optimized_scores = result.x
    print(f"Final Error was: {result.fun}")
    print(result.message)
    for i, score in enumerate(optimized_scores):
        df.iloc[i, 2] = round(score, 2)
    temp_df = df.iloc[:num_players]
    temp_df.sort_values(by='SCORE', ascending=False, inplace=True)
    #df.to_csv("ranked_popularity_ls.csv", index=False)
    pd.set_option('display.max_rows', None)
    print(temp_df)

if __name__ == '__main__':
    t = time.perf_counter()
    main("diff")
    #df = pd.read_csv('ranked_popularity_elo.csv', header = 0)
    #print(df.iloc[0:20])
    print(f"Time: {time.perf_counter() - t}")
