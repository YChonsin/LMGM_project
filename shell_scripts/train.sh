net=$1
datasetfolder=$2

python src/train.py \
    --net ${net} \
    --dpath ${datasetfolder} \
    --gpunum 1 \
    --batchSize 4 \
    --dtype 3d \
    --epoch 20

