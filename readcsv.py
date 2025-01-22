import pandas as pd

df = pd.read_csv('/Users/ryanfriess/Desktop/projects/spam-classification/data/messages.csv')
print(df["Category"])

label_list = []
for label in df["Category"]:
    label_list.append(label)

# print(label_list)