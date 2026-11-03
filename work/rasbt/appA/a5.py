# %% Step1：モデルを定義して中身を表示（本の A.5 そのまま）
###########################################################

import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()                      # ← 必ず最初に
        self.layers = torch.nn.Sequential(      # ← 部品を self に持たせる
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):                       # ← データの流れを書く
        logits = self.layers(x)
        return logits

model = NeuralNetwork(50, 3)
print(model)

# %% Step2：パラメータ数と、その内訳
###########################################################
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("パラメータ数:", num_params)

for name, p in model.named_parameters():
    print(name, tuple(p.shape), p.numel())

# %% 　Step3：シードを固定して、重みの初期値を再現する
###########################################################
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight.shape)
print(model.layers[0].weight[0, :5])   # 先頭の 5 個だけ

# %% ステップ 4：データを入れて出力を見る
###########################################################

torch.manual_seed(123)
X = torch.rand((1, 50))          # 1 件 × 50 次元

out = model(X)                   # ← model.forward(X) とは書かない
print(out)                       # grad_fn が付いている

with torch.no_grad():            # 学習しない時はこれ
    out = model(X)
print(out)                       # grad_fn が消える

with torch.no_grad():
    probs = torch.softmax(model(X), dim=1)
print(probs, probs.sum())        # 合計 1 の確率になる

X5 = torch.rand((5, 50))         # 5 件まとめて入れると？
print(model(X5).shape)

# %% ステップ 5：宿題。Sequential を使わずに書き直す
###########################################################
class MyNet(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_inputs, 30)
        self.fc2 = torch.nn.Linear(30, 20)
        self.fc3 = torch.nn.Linear(20, num_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

my = MyNet(50, 3)
print(my)
print("パラメータ数:", sum(p.numel() for p in my.parameters()))
print(my(X).shape)

# %% ステップ 6：わざと間違えてみる（つまずきの予防）
###########################################################
# (a) super().__init__() を忘れると？
class BadNet1(torch.nn.Module):
    def __init__(self):
        self.fc1 = torch.nn.Linear(50, 3)

try:
    BadNet1()
except Exception as e:
    print(type(e).__name__, ":", e)

# (b) 層を普通の Python リストに入れると？
class BadNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [torch.nn.Linear(50, 30), torch.nn.Linear(30, 3)]
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

print("リスト版:", sum(p.numel() for p in BadNet2().parameters()))

# (c) ModuleList に入れると？
class GoodNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(50, 30), torch.nn.Linear(30, 3)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

print("ModuleList 版:", sum(p.numel() for p in GoodNet2().parameters()))

# %% ステップ 7：わざと間違えてみる（つまずきの予防）⇒修正
###########################################################
# (a) super().__init__() を忘れると？
class BadNet1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(50, 3)

try:
    BadNet1()
except Exception as e:
    print(type(e).__name__, ":", e)

# (b) 層を普通の Python リストに入れると？
class BadNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(50, 30)
        self.fc2 = torch.nn.Linear(30, 20)
        self.fc3 = torch.nn.Linear(20, 3)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

print("リスト版:", sum(p.numel() for p in BadNet2().parameters()))

# (c) ModuleList に入れると？
class GoodNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(50, 30), torch.nn.Linear(30, 3)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

print("ModuleList 版:", sum(p.numel() for p in GoodNet2().parameters()))
# %%

# %% ステップ 7：わざと間違えてみる（つまずきの予防）⇒正解
###########################################################
# (a) super().__init__() を忘れると？
class BadNet1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(50, 3)

try:
    BadNet1()
except Exception as e:
    print(type(e).__name__, ":", e)

# (b) 層を普通の Python リストに入れると？
class BadNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(50, 30)
        self.fc2 = torch.nn.Linear(30, 20)
        self.fc3 = torch.nn.Linear(20, 3)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

print("リスト版:", sum(p.numel() for p in BadNet2().parameters()))

# (c) ModuleList に入れると？
class GoodNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(50, 30),
            torch.nn.Linear(30, 20),
            torch.nn.Linear(20, 3),
        ])
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:   # 最後の層の後には ReLU を入れない
                x = torch.relu(x)
        return x

g = GoodNet2()
print("ModuleList 版:", sum(p.numel() for p in g.parameters()), g(X).shape)

# %%
###########################################################
# %%
torch.manual_seed(123)
layer = torch.nn.Linear(50, 30)
W, b = layer.weight, layer.bias
X = torch.rand((1, 50))

y1 = layer(X)                 # PyTorch に計算させる
y2 = X @ W.T + b              # 自分で計算する（.T で向きを合わせる）
print(y1.shape, y2.shape)
print(torch.allclose(y1, y2)) # 同じ値か？

# 出力ニューロン 0 番 = W の 0 行目（50 個）と入力の掛け算の合計 + b[0]
y0 = (X[0] * W[0]).sum() + b[0]
print(y0.item(), y1[0, 0].item())
print(W[0].shape)
# %%
