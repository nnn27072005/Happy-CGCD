#!/bin/bash

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
GPU_ID=0
DATASET="cub"
BATCH_SIZE=64            # CUB ảnh to, để 64 cho an toàn VRAM, nếu GPU khỏe >24GB thì lên 128
NUM_OLD_CLASSES=100      # 100 lớp cũ
PROP_TRAIN_LABELS=0.8    # Tỷ lệ label
LR=0.01                  # LR thấp cho LeJEPA
LAMBDA_LEJEPA=0.1        # Trọng số LeJEPA
EPOCHS_OFFLINE=100
EPOCHS_ONLINE=50
SESSIONS=5
SEED=0

# Đường dẫn output gốc (phải khớp với exp_root_happy trong config.py của bạn)
# Mặc định trong code thường là ./outputs hoặc dev_outputs_Happy
EXP_ROOT_DIR="/kaggle/working/dev_outputs_Happy" 

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================================"
echo "BẮT ĐẦU CHẠY HAPPY + LEJEPA TRÊN DATASET: $DATASET"
echo "GPU: $GPU_ID | Batch: $BATCH_SIZE | LR: $LR"
echo "========================================================"

# ============================================================
# GIAI ĐOẠN 1: OFFLINE TRAINING (Học 100 lớp cũ)
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
    --transform imagenet \
    --seed $SEED \
    --exp_root $EXP_ROOT_DIR

if [ $? -ne 0 ]; then
    echo "[LỖI] Training Offline thất bại. Dừng script."
    exit 1
fi

echo "[HOÀN TẤT] Giai đoạn 1 thành công."

# ============================================================
# TỰ ĐỘNG TÌM ID CỦA OFFLINE MODEL VỪA CHẠY
# ============================================================
# Đường dẫn nơi lưu log offline (Happy thường lưu dạng: exp_root_offline/dataset/ID)
OFFLINE_LOG_PATH="${EXP_ROOT_DIR}_offline/${DATASET}"

# Tìm thư mục mới nhất được tạo ra trong đường dẫn đó
LATEST_OFFLINE_ID=$(ls -t "$OFFLINE_LOG_PATH" | head -n 1)

if [ -z "$LATEST_OFFLINE_ID" ]; then
    echo "[LỖI] Không tìm thấy thư mục checkpoint offline nào trong $OFFLINE_LOG_PATH"
    exit 1
fi

echo ""
echo ">>> Đã tìm thấy Offline ID mới nhất: $LATEST_OFFLINE_ID"
echo ">>> Chuẩn bị chuyển sang giai đoạn Online..."
sleep 3

# ============================================================
# GIAI ĐOẠN 2: ONLINE CONTINUAL LEARNING (5 Sessions)
# ============================================================
echo ""
echo "[GIAI ĐOẠN 2] Bắt đầu Training Online (Continual)..."

python train_happy.py \
    --dataset_name $DATASET \
    --train_session online \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs_online_per_session $EPOCHS_ONLINE \
    --num_old_classes $NUM_OLD_CLASSES \
    --continual_session_num $SESSIONS \
    --lambda_lejepa $LAMBDA_LEJEPA \
    --init_new_head \
    --prop_train_labels $PROP_TRAIN_LABELS \
    --load_offline_id "$LATEST_OFFLINE_ID" \
    --transform imagenet \
    --seed $SEED \
    --exp_root $EXP_ROOT_DIR

if [ $? -ne 0 ]; then
    echo "[LỖI] Training Online thất bại."
    exit 1
fi

echo ""
echo "========================================================"
echo "CHÚC MỪNG! TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT."
echo "========================================================"