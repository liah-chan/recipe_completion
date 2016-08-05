
import csv
from scipy.misc import comb
import itertools

with open('recipes/filtered/ingredients.csv') as f:
    ings = [row[0] for row in csv.reader(f)]
    
with open('recipes/filtered/recipes.csv') as f:
    recs = [(row[0], set(row[2:])) for row in csv.reader(f)]

cooccurs = {}
with open('cooccurs/cooccurs.csv') as f:
    for row in csv.reader(f):
        ing1, ing2, score = row
        score = float(score)
        if ing1 not in cooccurs:
            cooccurs[ing1] = {}
        cooccurs[ing1][ing2] = score

def make_profits(contexts_file, use_cooccurs, out, out_means=None):
    contexts = {}
    with open(contexts_file) as f:
        for row in csv.reader(f):
            ing1, ing2, score = row
            score = float(score)
            if ing1 not in contexts:
                contexts[ing1] = {}
            contexts[ing1][ing2] = score
        
    mean_coocs = {}
    for recid, rec in recs:
        mean_coocs[recid] = 0.0
        for ing1, ing2 in itertools.combinations(rec, 2):
            mean_coocs[recid] += cooccurs[ing1][ing2]
        mean_coocs[recid] /= comb(len(rec), 2)
    
    def profit(rec, i):
        recid, recings = rec
        left = recings - {i}
        rec_contexts = [contexts[i][ing] for ing in left]
        mean_context = (max(rec_contexts) + min(rec_contexts)) / 2.0
        if use_cooccurs:
            mean_cooc = sum([cooccurs[ing][i] for ing in left]) / len(left)
            mean_cooc_diff = mean_cooc - mean_coocs[recid]
            return mean_cooc_diff + mean_context
        else:
            return mean_context
        
    profits = []
    def work(recs):
        for rec in recs:
            recid = rec[0]
            for ing in ings:
                prof = profit(rec, ing)
                profits.append((recid, ing, prof))

    import threading
    
    pw = 16
    if len(recs) > pw:
        pwd = len(recs) // pw
        batches = [recs[i:i+pwd] for i in range(0, len(recs), pwd)]
    else:
        batches = [recs]
    threads = []
    for batch in batches:
        t = threading.Thread(target=work, args=(batch,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    profs = list(zip(*profits))[2]
    max_prof = max(profs)
    min_prof = min(profs)
    
    def scale(x):
        return 2 * ((x - min_prof) / (max_prof - min_prof)) - 1
    
    scaled_profits = {}
    
    for recid, ing, prof in profits:
        prof = scale(prof)
        if recid not in scaled_profits:
            scaled_profits[recid] = {}
        scaled_profits[recid][ing] = prof
        print('{},{},{}'.format(recid, ing, prof), file=out)
    
    if out_means:
        for recid, rec in recs:
            mean_profit = sum([scaled_profits[recid][ing] for ing in rec]) / len(rec)
            print('{},{}'.format(recid, mean_profit), file=out_means)

def to_bool(s):
   return {
        'True': True,
        'true': True,
        '1': True,
        'False': False,
        'false': False,
        '0': False
    }[s]

if __name__ == '__main__':
    """
    Usage: python3 profits.py CONTEXT_FILE USE_COOCCURS
    """
    import sys
    with open('profits.csv', 'w') as f1, open('mean_profits.csv', 'w') as f2:
        make_profits(sys.argv[1], to_bool(sys.argv[2]), f1, f2)

