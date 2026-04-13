from ultralytics import YOLO

# 1️⃣ 加载模型（换成你的路径）
model = YOLO("/home/root123/Code/Code/ultralytics-main-26/ultralytics/fgfd_yolo26/YOLO26_Star/weights/best.pt").model

# 2️⃣ 总参数量
total_params = sum(p.numel() for p in model.parameters())

# 3️⃣ 可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 4️⃣ 打印（转成M更直观）
print(f"Total Params: {total_params} ({total_params/1e6:.2f}M)")
print(f"Trainable Params: {trainable_params} ({trainable_params/1e6:.2f}M)")