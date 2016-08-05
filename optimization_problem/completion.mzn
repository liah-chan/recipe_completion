include "globals.mzn";

int: D = 1;
int: N_INGREDIENTS;
set of int: INGREDIENTS = 1..N_INGREDIENTS;
set of int: INGREDIENTS_D = 1..N_INGREDIENTS+D;

set of INGREDIENTS: template;
float: template_mean_profit;
array[INGREDIENTS] of float: template_profits;
array[INGREDIENTS_D] of float: template_profits_D = [if ing <= N_INGREDIENTS then template_profits[ing] else 0.0 endif | ing in INGREDIENTS_D];

array[1..D] of var INGREDIENTS_D: added;
var 0..D: added_n = among(added, INGREDIENTS);

constraint forall(i in 1..added_n)(not (added[i] in template));
constraint alldifferent(added);
constraint increasing(added);

array[1..D] of var float: added_profits = [template_profits[added[i]] | i in 1..D];
var float: profit_sum = sum(added_profits);

var float: compatibility = profit_sum - added_n * template_mean_profit;

solve maximize compatibility;
