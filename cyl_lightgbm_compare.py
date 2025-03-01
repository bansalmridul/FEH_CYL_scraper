import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('ranked_popularity_elo.csv', header = 0)
unit2world = dict()
for _, row in df.iterrows():
    unit2world[row['CHARACTER']] = row['WORLD']


model = lgb.Booster(model_file='feh_comparator.lgb')
with open('le_names.pkl', 'rb') as f:
    le_names = pickle.load(f)
with open('le_worlds.pkl', 'rb') as f:
    le_worlds = pickle.load(f)

b = True
while b:
    try:
        inp_test = input(r"Enter {unit1} {unit2} {CYL}: ")
        inp_arr = inp_test.split(' ')
        u1, u2, year = inp_arr[0], inp_arr[1], int(inp_arr[2])
        input_arr = [u1, unit2world[u1], u2, unit2world[u2], year]
        print(f"Predicting: {input_arr}")
        transformed_names = le_names.transform([u1, u2])
        transformed_worlds = le_worlds.transform([unit2world[u1], unit2world[u2]])
        transformed_input = [transformed_names[0], transformed_worlds[0], transformed_names[1], transformed_worlds[1], year]
    except Exception as e:
        print("Error: Invalid Input. Shutting down")
        print(e)
        b = False
        continue
    pred = model.predict([transformed_input], num_iteration=model.best_iteration)
    if pred[0] > 0:
        print(f"The model predicts that {u1} beats {u2} by {pred[0]:.1f} votes")
    else:
        print(f"The model predicts that {u2} beats {u1} by {-pred[0]:.1f} votes")