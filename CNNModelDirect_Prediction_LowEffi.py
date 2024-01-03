import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 确保使用 GPU 进行预测（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 CNN 模型结构
class SimpleCNN(nn.Module):
            # ...（模型定义保持不变）

model = SimpleCNN()
# 确保模型和所有计算都在 GPU 上执行（如果可用）
model.to(device)
# 加载训练好的模型参数
model.load_state_dict(torch.load(r'T:\neuralmodel\firstversioncnn.pth', map_location=device))
model.eval()

# 图像预处理的优化
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 优化后的预测函数
def predict_image(image_path, model, transform, device):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # 确保图像和模型在同一个设备上
    with torch.no_grad():
        prediction = model(image)
        predicted_class = torch.round(prediction).item()
    return "Cat" if predicted_class == 1 else "Dog"

# 使用模型进行预测
image_path = r'T:\neuraldata\effitest\222.jpg'
prediction = predict_image(image_path, model, transform, device)
print(f'Predicted class: {prediction}')
