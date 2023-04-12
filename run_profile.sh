export DATASET_DIR="/data/train/"
export LOG_FILE="/results/resnet.log"

python main.py \
       --epochs 1 \
       --batch_size 64 \
       --learning_rate 0.001