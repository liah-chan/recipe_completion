import csv
import numpy as np

with open('../recipes/filtered/ingredients.csv') as f:
    r = csv.reader(f)
    ings = [row[0] for row in r]

with open('../recipes/filtered/recipes.csv') as f:
    r = csv.reader(f)
    recipes = {int(row[0]): set(row[2:]) for row in r}

n = len(recipes)
occur = {}
cooccur = {}
for recid, rec in recipes.items():
    for ing1 in rec:
        if ing1 not in occur:
            occur[ing1] = 0
        occur[ing1] += 1
        if ing1 not in cooccur:
            cooccur[ing1] = {}
        for ing2 in rec:
            if ing2 not in cooccur[ing1]:
                cooccur[ing1][ing2] = 0
            cooccur[ing1][ing2] += 1

npmis = {}
for ing1 in ings:
    for ing2 in ings:
        if ing2 in cooccur[ing1]:
            npmi = np.log((occur[ing1] / n) * (occur[ing2] / n)) / (2 * np.log((cooccur[ing1][ing2] / n)))
        else:
            npmi = 0.0
        print('{},{},{}'.format(ing1, ing2, npmi))
