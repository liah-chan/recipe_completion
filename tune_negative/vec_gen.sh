#! /bin/bash

for negative in {1..30};do
	time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
	-output ./model_${negative} -cbow 0 \
	-size 5 -window 11  \
	-negative $negative -hs 0 -sample 1e-4 -threads 10 -min-count 120 -binary 0 -iter 15;	
done;

# for negative in {1..30};do
# 	time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
# 	-output ./model_${negative} -cbow 0 \
# 	-size 35 -window 32  \
# 	-negative $negative -hs 0 -sample 1e-4 -threads 10 \
# 	-min-count 120 -binary 0 -iter 15;	
# done;

	# time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
	# -output ./classes_3 -cbow 0 \
	# -size 35 -window 32  \
	# -negative 3 -hs 0 -sample 1e-4 -threads 10 \
	# -min-count 120 -classes 8 -binary 0 -iter 15;	

	# time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
	# -output ./model_3 -cbow 0 \
	# -size 35 -window 32  \
	# -negative 3 -hs 0 -sample 1e-4 -threads 10 \
	# -min-count 120 -binary 0 -iter 15;