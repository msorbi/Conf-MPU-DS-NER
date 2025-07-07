#!/bin/bash
if [ $# -ne 1 ] || ([ "$1" != "supervised" ] && [ "$1" != "distant" ])
then
    echo "usage: $0 (supervised|distant)"
    exit 1
fi
setting="$1"

if [ ! -f data/glove.6B.100d.txt ]
then
    (
        wget --no-check-certificate -P "data/" "https://nlp.stanford.edu/data/glove.6B.zip"
        cd data
        unzip glove.6B.zip
        cd ..
    )
fi

source="hdsner-utils/data/${setting}/ner_medieval_multilingual/FR/"
output_dir="data"
dataset_prefix="hdsner-"
if [ "${setting}" == "supervised" ]
then
    output_suffix="_Fully"
else
    output_suffix="_Dict_0.1"
fi

# copy and format datasets
python3 src/format_hdsner_datasets.py \
    --input-dir "${source}" \
    --output-dir "${output_dir}" \
    --output-suffix "${output_suffix}"

# execute on all datasets
for dataset in ${output_dir}/${dataset_prefix}*${dataset_suffix}
do
    dataset_name="`basename ${dataset}`"
    echo "${dataset_name}"
    time \
    (
        echo "Step 1"
        python pu_main.py --type bnPU --dataset "${dataset_name}" --flag Entity --m 15 --determine_entity True --epochs 2 # TODO: --epochs 100
        echo "$?"
        echo "Step 2"
        python pu_main.py --type bnPU --dataset "${dataset_name}" --add_probs True --flag ALL --added_suffix entity_prob --model_path "saved_model/bnPU_${dataset_name}_Entity_NA_lr_0.0001_cn_2_loss_SMAE_m_15.0_ws_NA_eta_NA_percent_1.0_trail_1" --epochs 2 # TODO: remove --epochs
        echo "$?"
        echo "Step 3"
        python pu_main.py --type conf_mPU --dataset "${dataset_name}" --flag ALL --suffix entity_prob --m 15 --eta 0.5 --lr 0.0005 --loss MAE --epochs 2 # TODO: --epochs 100
        echo "$?"
    ) \
    > "${dataset}/stdout.txt" 2> "${dataset}/stderr.txt"
done
