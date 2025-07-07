#!/bin/bash

(
# set attributes
dataset_prefix="data/hdsner-"
nl=$'\n'

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
            pred_file="`find ../predicted_data/ -name "pred_${split}_conf_mPU_${dataset}_*" | head -1`"
            output_file="../${dataset}/pred_${split}.json"
            python3 src/eval.py \
                --true "../${dataset}/${split}.txt" \
                --pred <(cut "${pred_file}" -d ' ' -f 1,2) \
                --output "$output_file" \
                --n 1 \
                --field-delimiter ' ' \
            > /dev/null
            echo "$output_file" # this is going to python below
        fi
    done | \
python3 -c "import sys ${nl}\
import json ${nl}\
summary = {} ${nl}\
for f in sys.stdin: ${nl}\
    with open(f.strip(), 'r') as fp: ${nl}\
        x = json.load(fp) ${nl}\
    summary[f.strip().split('/')[-2]] = x ${nl}\
with open(\"../data/hdsner_report_${split}.json\", 'w') as fp: ${nl}\
    json.dump(obj=summary, fp=fp) ${nl}\
"
done

# deactivate environment and return to project directory
conda deactivate
cd ..

)
