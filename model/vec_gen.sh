#! /bin/bash

time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
-output ./model_1 -cbow 0 \
-size 10 -window 32  \
-negative 17 -hs 0 -sample 1e-4 -threads 10 \
-min-count 75 -binary 0 -iter 15;

time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
-output ./model_2 -cbow 0 \
-size 10 -window 32  \
-negative 17 -hs 1 -sample 1e-4 -threads 10 \
-min-count 75 -binary 0 -iter 15;

time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
-output ./model_3 -cbow 1 \
-size 10 -window 32  \
-negative 17 -hs 0 -sample 1e-4 -threads 10 \
-min-count 75 -binary 0 -iter 15;

time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
-output ./model_4 -cbow 1 \
-size 10 -window 32  \
-negative 17 -hs 1 -sample 1e-4 -threads 10 \
-min-count 75 -binary 0 -iter 15;

# time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
# -output ./classes_3 -cbow 0 \
# -size 35 -window 32  \
# -negative 17 -hs 0 -sample 1e-4 -threads 10 \
# -min-count 75 -classes 3 -binary 0 -iter 15;

# time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
# -output ./classes_12 -cbow 0 \
# -size 35 -window 32  \
# -negative 17 -hs 0 -sample 1e-4 -threads 10 \
# -min-count 75 -classes 12 -binary 0 -iter 15;