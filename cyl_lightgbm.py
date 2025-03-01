import lightgbm as lgb
import pandas as pd
import itertools
import time, random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

def extract_matches():
    t = time.perf_counter()

    df = pd.read_csv('ranked_popularity_elo.csv', header = 0)
    print(df.iloc[0:20])

    pairs = []
    for (idx1, row1), (idx2, row2) in itertools.combinations(df.iterrows(), 2):
        name1, world1 = row1['CHARACTER'], row1['WORLD']
        name2, world2 = row2['CHARACTER'], row2['WORLD']
        for i in range(9):
            column = f"CYL {i+1} VOTES"
            if pd.isna(row1[column]) or pd.isna(row2[column]):
                continue
            if random.random() > 0.5: #randomly swap order of teams so that it doesn't overfit on name position
                pairs.append([name1, world1, name2, world2, i+1, row1[column] - row2[column]])
            else:
                pairs.append([name2, world2, name1, world1, i+1, row2[column] - row1[column]])
    pair_df = pd.DataFrame(pairs, columns=['name1', 'world1', 'name2', 'world2', 'year', 'score_diff'])

    pair_df.to_csv('feh_all_matches.csv')

    print(f"Time to create all matches: {time.perf_counter() - t}")


def main(train=False):
    #only run this once
    #extract_matches()

    df = pd.read_csv('feh_all_matches.csv', header = 0, index_col=0)

    le_names = LabelEncoder()
    le_worlds = LabelEncoder()
    df['name1'] = le_names.fit_transform(df['name1'])
    df['name2'] = le_names.transform(df['name2'])
    df['world1'] = le_worlds.fit_transform(df['world1'])
    df['world2'] = le_worlds.transform(df['world2'])
    with open('le_names.pkl', 'wb') as f:
        pickle.dump(le_names, f)
    with open('le_worlds.pkl', 'wb') as f:
        pickle.dump(le_worlds, f)

    X = df[['name1', 'world1', 'name2', 'world2', 'year']]
    y = df['score_diff']

    # Split the feature and target data into training and validation sets using a 80-20 split ratio.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=42)
    # Create LightGBM datasets for training and validation using X_train, y_train, X_valid, and y_valid.
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['name1', 'world1', 'name2', 'world2'])
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    if train:
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'num_iterations': 100000,
            'early_stopping_rounds': 10,
            'early_stopping_min_delta': 50,
        }
        print("Starting training...")
        train_start = time.perf_counter()
        model = lgb.train(params, train_data, valid_sets=[valid_data])
        print(f"Training Time: {time.perf_counter() - train_start}")
        model.save_model('feh_comparator.lgb')
    else:
        model = lgb.Booster(model_file='feh_comparator.lgb')
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    mse = mean_squared_error(y_valid.values, y_pred)
    r2 = r2_score(y_valid.values, y_pred)
    print(f"Validation MSE: {mse:.2f}")
    print(f"Validation r2: {r2:.6f}")

    lgb.plot_importance(model, figsize=(8, 4))
    plt.title("Feature Importance")
    plt.savefig('feh_plot.png')  # Save as PNG


if __name__ == '__main__':
    main(train=True)
