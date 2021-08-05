import pandas as pd
from collections import Counter

def findCommon(list1, list2):
    return list(set(list1).intersection(list2))

df  = pd.read_csv('data/dataset.csv')
empty = []

for i in range(1, 17): 
    empty += df['Symptom_' + str(i)].tolist()
empty = {k: v for k, v in sorted(Counter(empty).items(), key=lambda item: item[1], reverse=True)}
[print(k,':',v) for k, v in empty.items()]

findCommon()
