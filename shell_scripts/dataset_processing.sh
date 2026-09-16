task=$1
dataset=$2
norm=$3

python src/dataset_processing.py \
    --task ${task} \
    --net LMGM \
    --norm ${norm} \
    --df ${dataset}

python src/dataset_processing.py \
    --task ${task} \
    --net ANN \
    --norm ${norm} \
    --df ${dataset}