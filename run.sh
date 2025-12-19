#!/bin/bash

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
GPU_ID=0
DATASET="cub"
BATCH_SIZE=64           
NUM_OLD_CLASSES=100      
PROP_TRAIN_LABELS=0.8   
LR=0.0001                 
LAMBDA_LEJEPA=0.05
SUP_WEIGHT=0.35    
EPOCHS_OFFLINE=100
EPOCHS_ONLINE=50
SESSIONS=5
SEED=0

# CUB settings
ONLINE_OLD_SEEN_NUM=5     
ONLINE_NOVEL_SEEN_NUM=5   
ONLINE_NOVEL_UNSEEN_NUM=25 

EXP_ROOT_DIR="dev_outputs_Happy" 

export CUDA_VISIBLE_DEVICES=$GPU_ID

# ============================================================
# GIAI ĐOẠN 1: OFFLINE TRAINING
# ============================================================
echo ""
echo "[GIAI ĐOẠN 1] Bắt đầu Training Offline..."

python train_happy.py \
    --dataset_name $DATASET \
    --train_session offline \
    --batch_size $BATCH_SIZE \
    --grad_from_block 11 \
    --lr $LR \
    --epochs_offline $EPOCHS_OFFLINE \
    --num_old_classes $NUM_OLD_CLASSES \
    --prop_train_labels $PROP_TRAIN_LABELS \
    --lambda_lejepa $LAMBDA_LEJEPA \
    --sup_weight $SUP_WEIGHT \
    --transform imagenet \
    --seed $SEED \
    --exp_root $EXP_ROOT_DIR \
    --eval_funcs v2 \
    --online_old_seen_num $ONLINE_OLD_SEEN_NUM \
    --online_novel_seen_num $ONLINE_NOVEL_SEEN_NUM \
    --online_novel_unseen_num $ONLINE_NOVEL_UNSEEN_NUM

if [ $? -ne 0 ]; then
    echo "[LỖI] Training Offline thất bại. Dừng script."
    exit 1
fi

echo "[HOÀN TẤT] Giai đoạn 1 thành công."

# ============================================================
# TỰ ĐỘNG TÌM ID
# ============================================================
OFFLINE_LOG_PATH="${EXP_ROOT_DIR}_offline/${DATASET}"
LATEST_OFFLINE_ID=$(ls -t "$OFFLINE_LOG_PATH" | head -n 1)

if [ -z "$LATEST_OFFLINE_ID" ]; then
    echo "[LỖI] Không tìm thấy thư mục checkpoint offline."
    exit 1
fi

echo ""
echo ">>> Đã tìm thấy Offline ID mới nhất: $LATEST_OFFLINE_ID"
sleep 3

# ============================================================
# GIAI ĐOẠN 2: ONLINE CONTINUAL LEARNING
# ============================================================
echo ""
echo "[GIAI ĐOẠN 2] Bắt đầu Training Online..."

python train_happy.py \
    --dataset_name $DATASET \
    --train_session online \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs_online_per_session $EPOCHS_ONLINE \
    --num_old_classes $NUM_OLD_CLASSES \
    --continual_session_num $SESSIONS \
    --lambda_lejepa $LAMBDA_LEJEPA \
    --sup_weight $SUP_WEIGHT \
    --init_new_head \
    --prop_train_labels $PROP_TRAIN_LABELS \
    --load_offline_id "$LATEST_OFFLINE_ID" \
    --transform imagenet \
    --seed $SEED \
    --exp_root $EXP_ROOT_DIR \
    --eval_funcs v2 \
    --online_old_seen_num $ONLINE_OLD_SEEN_NUM \
    --online_novel_seen_num $ONLINE_NOVEL_SEEN_NUM \
    --online_novel_unseen_num $ONLINE_NOVEL_UNSEEN_NUM

if [ $? -ne 0 ]; then
    echo "[LỖI] Training Online thất bại."
    exit 1
fi

echo ""
echo "CHÚC MỪNG! TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT."