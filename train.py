# ─── 必要なライブラリのインポート ───────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# 自分のモデル定義を含むファイルをインポート
from model import CNN1d_with_resnet

# ─── デバイス選択（GPUがあるならGPU）───────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

# ─── データ読み込み ─────────────────────────────────────────
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)
X = data['x'].float().to(device)  # 例: shape (サンプル数, 1, 3000)
Y = data['y'].float().to(device)  # shape (サンプル数, 2)

print("データ形状 X:", X.shape, " Y:", Y.shape)

# ─── モデルと損失関数、オプティマイザの準備 ───────────────
model = CNN1d_with_resnet().to(device)
criterion = nn.MSELoss()  # あとはWeightedMSELossに差し替えOK
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── 学習ループ ─────────────────────────────────────────
epochs = 100
batch_size = 32
num_samples = X.size(0)

for epoch in range(1, epochs + 1):
    perm = torch.randperm(num_samples)
    running_loss = 0.0

    for i in range(0, num_samples, batch_size):
        idx = perm[i:i+batch_size]
        batch_x = X[idx]
        batch_y = Y[idx]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / num_samples
    print(f"Epoch {epoch}/{epochs}  平均損失: {epoch_loss:.6f}")

# ─── 学習済みモデルの保存 ─────────────────────────────────
torch.save(model.state_dict(), 'model.pth')
print("✅ 学習が完了し、'model.pth' を保存しました。")

# ─── 確認：読み込み・推論テスト ─────────────────────────────
model2 = CNN1d_with_resnet().to(device)
model2.load_state_dict(torch.load('model.pth', map_location=device))
model2.eval()
with torch.no_grad():
    dummy = torch.randn(3, 1, X.size(2)).to(device)
    out = model2(dummy)
    print("推論テスト出力サイズ:", out.shape)
    print("推論結果例:", out[:2])
