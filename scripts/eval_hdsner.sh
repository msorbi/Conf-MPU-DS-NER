#!/bin/bash

(
# set attributes
dataset_prefix="data/hdsner-"

datasets="`echo ${dataset_prefix}*`"

# move to directory and activate evaluation environment
cd hdsner-utils/
conda activate hdsner

# execute on all datasets
for split in valid test
do
    for dataset in ${datasets}
    do
        if [ -d "../${dataset}" ]
        then
            dataset_name="`basename "${dataset}"`"
            pred_file="`find ../predicted_data/ -name "pred_${split}_conf_mPU_${dataset_name}_*" | head -1`"
            output_file="../${dataset}/pred_${split}.json"
            python3 src/eval.py \
                --true "../${dataset}/${split}.txt" \
                --pred <(python3 src/convert_index.py --input <(cut "${pred_file}" -d ' ' -f 1,3)) \
                --output "$output_file" \
                --n 1 \
                --field-delimiter ' ' \
            > /dev/null
            echo "$output_file" # this is going to python below
        fi
    done | python3 src/eval_summary.py --output "../data/hdsner_report_${split}.json"
done

# deactivate environment and return to project directory
conda deactivate
cd ..

)
