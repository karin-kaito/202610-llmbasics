# %% ステップ 1：小さいデータを作る（本の A.6 そのまま）
###########################################################
import torch

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

print(X_train.shape, y_train.shape)   # 5 件 × 2 次元、ラベル 5 個
print(X_test.shape, y_test.shape)

# %% ステップ 2：Dataset を作る（本の A.6 そのまま）
###########################################################
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):          # ← データを self に持たせるだけ
        self.features = X
        self.labels = y

    def __getitem__(self, index):      # ← ds[index] と書くと呼ばれる。「1 件」を返す
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):                 # ← len(ds) と書くと呼ばれる。件数を返す
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))                   # __len__ が呼ばれる
print(train_ds[0])                     # __getitem__(0) が呼ばれる
x, y = train_ds[3]
print("x,x.shape:",x, x.shape)          # 1 件ぶん：x は (2,)、y は 0 次元
print("y,y.shape:",y, y.shape)          # 1 件ぶん：x は (2,)、y は 0 次元

# %% ステップ 3：DataLoader でまとめて取り出す（本の A.6 そのまま）
###########################################################
from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,      # 2 件ずつまとめる
    shuffle=False,      # 順番を混ぜる
    num_workers=0      # 0 のままでよい
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,     # テストは混ぜない
    num_workers=0
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)

print(len(train_loader))               # バッチの数（件数ではない）


# %%
# %% ステップ 8：形の規則を 1 本の流れで確かめる
###########################################################
# 前提：ステップ 1〜3 を実行済み（train_ds、train_loader がある）

# (1) Dataset から出るのは「1 件」。バッチの次元はまだ無い
x1, y1 = train_ds[0]
print("1 件      :", tuple(x1.shape), tuple(y1.shape))     # (2,) と ()

# (2) DataLoader が、先頭に「件数」の次元を 1 つ足す
xb, yb = next(iter(train_loader))
print("1 バッチ  :", tuple(xb.shape), tuple(yb.shape))     # (2, 2) と (2,)
#                    ↑ (件数, 特徴の数)        ↑ (件数,)

# (3) Linear が見るのは「最後の次元」だけ。ここが in_features と合えばよい
layer = torch.nn.Linear(2, 3)          # 特徴 2 個 → 出力 3 個
print("Linear 後 :", tuple(layer(xb).shape))               # (2, 3) = (件数, 出力の数)
print("1 件でも  :", tuple(layer(x1).shape))               # (3,)
print("5 件でも  :", tuple(layer(X_train).shape))          # (5, 3)

# (4) 最後の次元が合わないと落ちる
try:
    # torch.nn.Linear(4, 3)(xb)          # 特徴 4 個を期待しているのに 2 個
    torch.nn.Linear(4, 3)(xb)          # 特徴 4 個を期待しているのに 2 個
except Exception as e:
    print(type(e).__name__, ":", e)

# (5) 件数（先頭の次元）は幾つでもよい。バッチごとに違ってもよい
for xb, yb in train_loader:
    print(tuple(xb.shape), "→", tuple(layer(xb).shape))
# %%
# %% ステップ 9-1：形（shape）の読み方。「[ の数」と「中の個数」を数える
###########################################################
import torch

a = torch.tensor(5.0)                        # [ が 0 個：数字 1 個だけ
b = torch.tensor([-1.2, 3.1])                # [ が 1 重：数字が 2 個並ぶ
c = torch.tensor([[-1.2, 3.1],
                  [-0.9, 2.9],
                  [-0.5, 2.6]])              # [ が 2 重：「2 個並び」が 3 行

print(a, a.shape)
print(b, b.shape)
print(c, c.shape)
# %%
# %% ステップ 9-2：1 件を縦に積むと、バッチになる
###########################################################
x0 = torch.tensor([-1.2, 3.1])               # 1 件め（数字 2 個）
x1 = torch.tensor([-0.9, 2.9])               # 2 件め（数字 2 個）
print(x0.shape, x1.shape)

xb = torch.stack([x0, x1])                   # 縦に積む
print(xb)
print(xb.shape)                              # (2, 2) = (件数, 1 件あたりの数字の個数)

print(xb[0])                                 # 積んだものの 0 行め = 1 件め
print(xb[1])                                 # 1 行め = 2 件め
# %%
# %% ステップ 9-3：Linear は 1 行ずつ、別々に計算している
###########################################################
torch.manual_seed(123)
layer = torch.nn.Linear(2, 3)                # 数字 2 個を入れると、数字 3 個が出る

print(layer(x0))                             # 1 件めだけ入れる → 数字 3 個
print(layer(x1))                             # 2 件めだけ入れる → 数字 3 個
print(layer(xb))                             # 2 件まとめて入れる → 「3 個並び」が 2 行
print(layer(xb).shape)                       # (2, 3) = (件数, 出てくる数字の個数)
# %%
