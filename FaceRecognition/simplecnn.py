# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
#
# # GPU 사용 가능 여부 확인
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 데이터셋 디렉토리 경로
# dataset_dir = "dataset"
#
# # 데이터 전처리 변환
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 이미지 크기 조정
#     transforms.ToTensor(),  # 이미지를 텐서로 변환
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
# ])
#
# # 데이터셋 로드
# dataset = ImageFolder(dataset_dir, transform=transform)
#
# # DataLoader를 사용하여 데이터셋을 배치 단위로 로드
# batch_size = 32
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#
# # 간단한 CNN 모델 정의
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(32 * 56 * 56, 512)
#         self.fc2 = nn.Linear(512, 10)  # 분류할 클래스 수에 맞게 조정
#
#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = x.view(-1, 32 * 56 * 56)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # 모델 초기화 및 GPU 사용 설정
# model = SimpleCNN().to(device)
#
# # 손실 함수 및 최적화 알고리즘 정의
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 모델 학습
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#
#     for images, labels in data_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * images.size(0)
#
#     epoch_loss = running_loss / len(dataset)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#
# # 학습된 모델 저장
# torch.save(model.state_dict(), "simple_cnn_model.pth")
#
# # 모델 평가
# model.eval()
# correct = 0
# total = 0
#
# with torch.no_grad():
#     for images, labels in data_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# accuracy = correct / total
# print(f"Accuracy on the test set: {accuracy:.4f}")
