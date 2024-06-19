import pandas as pd

df = pd.read_csv("heart.csv")

objects_with_missing_values = {}
attrs_with_missing_values = []

for i in range(len(df)):
    for attr in df.columns.values:
        if df.loc[i][attr] == "?":
            if i not in objects_with_missing_values:
                objects_with_missing_values[i] = [attr]
            else:
                objects_with_missing_values[i].append(attr)

            if attr not in attrs_with_missing_values:
                attrs_with_missing_values.append(attr)

attrs_means = {}

for attr in attrs_with_missing_values:
    mean = 0
    count = 0

    for i in range(len(df)):
        if df.loc[i][attr] != '?':
            mean += float(df.loc[i][attr])
            count += 1

    attrs_means[attr] = mean/count



for obj, missing_attrs in objects_with_missing_values.items():
    for attr in missing_attrs:
        df.loc[obj, attr] = attrs_means[attr]


df.to_csv('processed-heart.csv')
