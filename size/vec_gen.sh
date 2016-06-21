#! /bin/bash

# for i in {1..20};do
# 	size=$((i*5));
# 	time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
# 	-output ./model_${size} -cbow 1 \
# 	-size $size -window 32  \
# 	-negative 1 -hs 1 -sample 1e-4 -threads 10 -min-count 75 -binary 0 -iter 15;
# done;

for i in {1..20};do
	size=$((i*5));
	time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
	-output ./classes_model_${size} -cbow 1 \
	-size $size -window 32  \
	-negative 1 -hs 1 -sample 1e-4 -threads 10 -min-count 75 -classes 9 -binary 0 -iter 15;
done;