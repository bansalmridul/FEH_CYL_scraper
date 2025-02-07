import pandas as pd
import itertools, random

k = 32
repititions = 10

def expected_score(r1, r2):
    denom = 1 + 10 ** ((r2 - r1)/400)
    return 1 / denom

def main():
    df = pd.read_csv('ranked_popularity_elo.csv', header = 0)
    print(df)

    match_stats = [0] * len(df)
    for i, row in df.iterrows():
        years = set()
        for j in range(3, 12):
            if not pd.isna(row.iloc[j]):
                years.add(j)
        match_stats[i] = years

    count = 1
    elos = [1500] * len(df)
    population = list(range(0, len(df)))
    random.shuffle(population)
    pairs = list(itertools.combinations(population, 2))
    for i1, i2 in pairs:
        if count % 10_000 == 0:
            print(f"{count} matches done")
        common = list(match_stats[i1].intersection(match_stats[i2]))
        if len(common) == 0:
            continue

        unit1, unit2 = df.iloc[i1], df.iloc[i2]
        year = random.choice(common)
        v1, v2 = unit1.iloc[year], unit2.iloc[year]
        s1, s2 = sqrt(v1/(v1 + v2)), sqrt(v2/(v1 + v2))
        r1, r2 = elos[i1], elos[i2]
        p1 = expected_score(r1, r2)
        p2 = 1 - p1
        elos[i1] += k * (s1 - p1)
        elos[i2] += k * (s2 - p2)
        count += 1

    for i, elo in enumerate(elos):
        df.iloc[i, 2] = round(elo,2)
    df.sort_values(by='SCORE', ascending=False, inplace=True)
    df.to_csv("ranked_popularity_elo.csv", index=False)


if __name__ == '__main__':
    main()
    df = pd.read_csv('ranked_popularity_elo.csv', header = 0)
    print(df.iloc[0:20])
