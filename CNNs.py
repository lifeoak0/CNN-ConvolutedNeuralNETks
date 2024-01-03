import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

# 检查是否有可用的 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN 模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # 根据卷积层输出和图像尺寸计算全连接层的输入尺寸
        num_features = 25 * 25 * 128
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 25 * 25 * 128)  # 修改展平操作
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleCNN().to(device)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据转换：统一尺寸、转换为张量、归一化
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_data = datasets.ImageFolder(r'T:\neuraldata\DOGCATarchive\training_set', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

# 开始训练
train(model, train_loader, criterion, optimizer, num_epochs=10)

# 保存和加载模型
torch.save(model.state_dict(), r'T:\neuralmodel\firstversioncnn.pth')
model.load_state_dict(torch.load(r'T:\neuralmodel\firstversioncnn.pth'))

# 预测函数
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image)
        return torch.round(prediction).item()

# 预测新图像
prediction = predict_image(r'T:\neuraldata\effitest\222.jpg')
print("Predicted class:", "Cat" if prediction == 0 else "Dog")
