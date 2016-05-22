
for window in {3..32};do
	time ../word2vec/trunk/word2vec -train ../data/recipes_as_input.txt \
	-output ./model_${window} -cbow 0 \
	-size 5 -window ${window}  \
	-negative 1 -hs 0 -sample 1e-4 -threads 10 \
	-min-count 120 -binary 0 -iter 15;
done;
