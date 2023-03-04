# train on QNRF
python train.py --tag CHSNet-qnrf --no-wandb --device 0 --max-noisy-ratio 0.05 --max-weight-ratio 0.5 --scheduler cosine --dcsize 4 --batch-size 8 --lr 4e-5 --data-dir ../DATASET/QNRF-trainfull-test-dmapfix15 --val-start 200 --val-epoch 5
# train on SHA
python train.py --tag CHSNet-sha --no-wandb  --device 1 --max-noisy-ratio 0.10 --max-weight-ratio 1.0 --scheduler cosine --dcsize 2 --batch-size 8 --lr 4e-5 --data-dir ../DATASET/SHA-train-test-dmapfix15      --val-start 200 --val-epoch 5
# train on SHB
python train.py --tag CHSNet-shb --no-wandb  --device 2 --max-noisy-ratio 0.05 --max-weight-ratio 1.0 --scheduler cosine --dcsize 4 --batch-size 8 --lr 4e-5 --data-dir ../DATASET/SHB-train-test-dmapfix15      --val-start 200 --val-epoch 5


