# --- 必要ライブラリ ---
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import sys
sys.path.append('/content')  # モジュール探索パスに /content を追加

from loss_function import WeightedMSELoss

# --- 1D Conv -> RGB画像変換ユニット ---
class PreprocessToImage(nn.Module):
    def __init__(self):
        super().__init__()
        def block():
            return nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(64)
            )
        self.low = block()
        self.mid = block()
        self.high = block()

    def forward(self, x):  # x: (B, 1, 3000)
        low = x[:, :, 0:80]
        mid = x[:, :, 80:350]
        high = x[:, :, 350:]

        low = self.low(low).unsqueeze(1)   # (B, 1, 64, 64)
        mid = self.mid(mid).unsqueeze(1)
        high = self.high(high).unsqueeze(1)

        return torch.cat([low, mid, high], dim=1)  # (B, 3, 64, 64)

# --- ResNet本体（ResNet18の簡易版） ---
from torchvision.models import resnet18

def get_resnet18(num_classes):
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 入力3ch対応
    return model

# --- データ読み込み ---
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

X = data['x'].float().to("cpu")  # (B, 1, 3000)
Y = data['y'].float().to("cpu")

# --- モデルと学習準備 ---
pre = PreprocessToImage()
model = get_resnet18(num_classes=2)

criterion = WeightedMSELoss(weight1=10, weight2=1)
optimizer = optim.Adam(list(pre.parameters()) + list(model.parameters()), lr=0.001)

# --- 学習ループ（簡易版） ---
epochs = 5
batch_size = 128

for epoch in range(epochs):
    permutation = torch.randperm(X.size(0))
    for i in range(0, X.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X[indices], Y[indices]

        optimizer.zero_grad()
        image_input = pre(batch_x)
        outputs = model(image_input)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# --- モデル保存 ---
torch.save({
    'preprocess': pre.state_dict(),
    'resnet': model.state_dict(),
}, 'model.pth')
print("✅ モデルの重みを 'model.pth' に保存しました。")

# --- 推論テスト ---
pre.load_state_dict(torch.load('model.pth')['preprocess'])
model.load_state_dict(torch.load('model.pth')['resnet'])
model.eval()
pre.eval()

with torch.no_grad():
    image_input = pre(X)
    preds = model(image_input)
    for i in range(5):
        ea, eb = preds[i].tolist()
        print(f"Sample {i}: 推定された Eα = {ea:.4f}, Eβ = {eb:.4f}")
