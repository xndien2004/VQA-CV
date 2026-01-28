export CUDA_VISIBLE_DEVICES=1

python3 - << 'EOF'
import torch, time

device = "cuda"

# =========================
# 1. Chiếm ~46GB VRAM
# float32 = 4 bytes
# 46GB ≈ 46 * 1024^3 bytes
# => ~11.5 tỷ phần tử
# =========================

NUM_TENSORS = 40
SIZE = 16384  # 16384 x 16384 x 4B ≈ 1GB mỗi tensor

buffers = [
    torch.randn(SIZE, SIZE, device=device)
    for _ in range(NUM_TENSORS)
]

print("🔥 VRAM ~46GB allocated")

# =========================
# 2. Giữ GPU bận nhẹ (để không bị swap out)
# =========================
while True:
    buffers[0] = buffers[0] @ buffers[1]
    torch.cuda.synchronize()
    time.sleep(0.02)
EOF

# 418731