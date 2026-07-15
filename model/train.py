from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # 加载预训练模型

results = model.train(
    data="kitti.yaml",    # 数据集配置
    epochs=100,           # 训练100轮
    imgsz=640,            # 图片尺寸640×640
    batch=16,             # 每次迭代处理16张图片（批次大小）
    device=0,             # 使用第0块GPU
    workers=8,            # 用8个线程加载数据
    patience=50,          # 如果50轮没提升就提前停止训练
    classes=[0, 1, 2]     # 只训练 car, van, truck 这三个类别
)
