#! /bin/bash

for negative in {5..20}; do	
	time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
	-output ../vectors/new/model_${negative}_16 -cbow 0 \
	-size 200 -window 16  \
	-negative $negative -hs 0 -sample 0 -threads 20 -min-count 170 -binary 0 -iter 15;	
done;

# #use skip-gram with hs(no neg sampling)
# for i in {25..30}; do		
# 		size=$((i*10))
# 		echo size: $size negative_samples: $negative
# 		time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
# 		-output ../word2vec_vec_cbow/model_${size}_${negative}.bin -cbow 0 \
# 		-size $size -window 16  \
# 		-negative 0 -hs 1 -sample 1e-5 -threads 20 -min-count 170 -binary 0 -iter 15;
# 	done;
# done;

# #use skip-gram with neg sampling

# for i in {25..30}; do
# 	for negative in {1..20}; do	
# 		for window in {3..16}; do
# 			size=$((i*10))
# 			echo size: $size negative_samples: $negative
# 			time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
# 			-output ../word2vec_vec_sg/model_${size}_${negative}_{$window} -cbow 0 \
# 			-size $size -window $window  \
# 			-negative 0 -hs 0 -sample 1e-5 -threads 20 -min-count 170 -binary 0 -iter 15;
# 		done;
# 	done;
# done;