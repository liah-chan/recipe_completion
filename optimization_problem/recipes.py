"""
Set of APIs to interact with the recipes data.
"""

import logging
import pymzn
import csv

data_dir = 'data/'
dzn_dir = 'dzn/'

class Recipes(object):
    
    def __init__(self):
        self.log = logging.getLogger(__name__)
        
        ingredients_csv = data_dir + 'recipes/filtered/ingredients.csv'
        recipes_csv = data_dir + 'recipes/filtered/recipes.csv'
        profits_csv = data_dir + 'profits.csv'
        mean_profits_csv = data_dir + 'mean_profits.csv'
        
        self._load_ingredients(ingredients_csv)
        self._load_recipes(recipes_csv)
        self._load_profits(profits_csv)
        self._load_mean_profits(mean_profits_csv)
        
        self.recipes_dzn = dzn_dir + 'recipes.dzn'
        self.profits_dzn = dzn_dir + 'profits.dzn'
        
        self.dzn_files = [self.recipes_dzn, self.profits_dzn]
        
    def _load_ingredients(self, ingredients_csv):
        self.log.debug('Loading ingredients from: {}'.format(ingredients_csv))
        self.ings = []          # List of ingredient names
        self.ingset = set()     # Set of ingredient names
        self.ingidxs = []       # List of ingredient indices
        self.ing2ingidx = {}    # Map from ingredient names to indices
        self.ingidx2ing = {}    # Map from ingredient indices to names
        self.ingtypes = {}      # Map from ingredient indices to types
        with open(ingredients_csv) as f:
            for i, row in enumerate(csv.reader(f)):
                ing = row[0]
                ingidx = i + 1
                ingtype = row[1]
                self.ings.append(ing)
                self.ingset.add(ing)
                self.ingidxs.append(ingidx)
                self.ingtypes[ingidx] = ingtype
                self.ing2ingidx[ing] = ingidx
                self.ingidx2ing[ingidx] = ing
    
    def _load_recipes(self, recipes_csv):
        self.log.debug('Loading recipes from: {}'.format(recipes_csv))
        self.recs = {}              # Map from recipe indices to set of ingredient indices
        self.rectypes = {}          # Map from recipe indices to types (cousines)
        self.recid2recidx = {}      # Map from recipe ids to indices
        self.recidx2recid = {}      # Map from recipe indices to ids
        with open(recipes_csv) as f:
            for i, row in enumerate(csv.reader(f)):
                recid = int(row[0])
                recidx = i + 1
                rectype = row[1]
                recingidxs = [self.ing2ingidx[ing] for ing in row[2:]]
                self.recs[recidx] = set(recingidxs)
                self.rectypes[recidx] = rectype
                self.recid2recidx[recid] = recidx
                self.recidx2recid[recidx] = recid
    
    def _load_profits(self, profits_csv):
        self.log.debug('Loading profits from: {}'.format(profits_csv))
        self.profits = {}       # Map from a recipe index and an ingredient index to a score (float)
        with open(profits_csv) as f:
            for row in csv.reader(f):
                recid = int(row[0])
                ing = row[1]
                if ing in self.ingset and recid in self.recid2recidx:
                    recidx = self.recid2recidx[recid]
                    ingidx = self.ing2ingidx[ing]
                    score = float(row[2])
                    if recidx not in self.profits:
                        self.profits[recidx] = {}
                    self.profits[recidx][ingidx] = score

    def _load_mean_profits(self, mean_profits_csv):
        self.log.debug('Loading mean profits from: {}'.format(mean_profits_csv))
        self.mean_profits = {}      # Map from a recipe index to an average score (float)
        with open(mean_profits_csv) as f:
            for row in csv.reader(f):
                recid = int(row[0])
                if recid in self.recid2recidx:
                    recidx = self.recid2recidx[recid]
                    score = float(row[1])
                    self.mean_profits[recidx] = score

    def _is_none(self, ingidx):
        return ingidx > len(self.ings)

    def rank(self, recipe_id, removed_ingredient):
        tmplidx = self.recid2recidx[recipe_id]
        remingidx = self.ing2ingidx[removed_ingredient]
        tmpl = self.recs[tmplidx]
        data = {'N_INGREDIENTS': len(self.ings), 
                'template': tmpl - {remingidx},
                'template_profits': self.profits[tmplidx],
                'template_mean_profit': self.mean_profits[tmplidx]}
        model = pymzn.Model('completion.mzn')
        solns = []
        while True:
            soln = pymzn.minizinc(model, data=data, parallel=0, 
                                  output_vars=['added', 'compatibility'])[0]
            comp = soln['compatibility']
            added = soln['added'][0]
            if self._is_none(added):
                solns.append(('None', comp))
                break
            if added == remingidx:
                solns.append((self.ingidx2ing[added], comp))
                break
            solns.append((self.ingidx2ing[added], comp))
            model.constraint('not member(added, {})'.format(added))
        return solns

