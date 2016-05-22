#! /bin/bash/

orignial_ingredient=../data/original/ingredients.csv;
model_dir=../tune_size/;
model_file=${model_dir}model_10;
category_file=${model_dir}category.txt
final_file=${model_file}_with_cate

#concatinating the vector file wiht food category
if [ -f "$category_file" ]; then
	rm $category_file
fi
ingredients_in_model=($(cat $model_file | sed '1d;2d' | cut -d ' ' -f 1));
for ((i=0; i<${#ingredients_in_model[@]}; i++));do
	#ingredient_category = ${ingredients_in_model[i]}
	awk -F"," -v ingredient="${ingredients_in_model[i]}" '$1==ingredient {print $2}' $orignial_ingredient >>$category_file;
done
#final_file:
#food_category ingredient vec_1 vec_2 vec_3 ...
cat $model_file| sed '1d;2d' | paste -d' ' $category_file - >$final_file

#calculate for each column in the vector file,starting from -k3, which food categry is the majority
cat $final_file | sort -k3nr | cut -d ' ' -f 1 | head -100 |sort| uniq -c |sort -nr
