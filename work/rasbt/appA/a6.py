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
# %% ステップ 4：設定を少し変えて試す
###########################################################
# (a) drop_last=True：半端な最後のバッチを捨てる
torch.manual_seed(123)
loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
for idx, (x, y) in enumerate(loader):
    print(f"Batch {idx+1}:", x.shape, y)
print("バッチの数:", len(loader))

# (b) 同じ loader を 3 周（3 エポック）回す：順番は毎回変わる？
torch.manual_seed(123)
loader = DataLoader(train_ds, batch_size=2, shuffle=True)
for epoch in range(3):
    print("epoch(b)", epoch, [y.tolist() for x, y in loader])

# (c) shuffle=False だと？
loader = DataLoader(train_ds, batch_size=2, shuffle=False)
for epoch in range(2):
    print("epoch(c)", epoch, [y.tolist() for x, y in loader])

# (d) batch_size を 1、3、5、10 に変えると、バッチの形と数は？
for bs in [1, 3, 5, 10]:
    loader = DataLoader(train_ds, batch_size=bs, shuffle=False)
    print("batch_size", bs, "→", [tuple(x.shape) for x, y in loader])

# %%
# %% ステップ 5：DataLoader がやっていることを、自分の手でやる
###########################################################
# やっていることは 3 つ：
#   1) 番号（インデックス）を並べる（shuffle なら混ぜる）
#   2) 番号を batch_size 個ずつに区切り、ds[番号] で 1 件ずつ取る
#   3) 取った 1 件ずつを torch.stack で 1 つのテンソルに重ねる
torch.manual_seed(123)
order = torch.randperm(len(train_ds)).tolist()     # 1) 混ぜた番号
print("順番:", order)

batch_size = 2
for start in range(0, len(order), batch_size):
    idxs = order[start:start + batch_size]         # 2) 番号を区切る
    items = [train_ds[i] for i in idxs]            #    1 件ずつ取る
    xb = torch.stack([x for x, y in items])        # 3) 重ねる
    yb = torch.stack([y for x, y in items])
    print(idxs, tuple(xb.shape), yb)

# stack は「新しい次元を前に足して重ねる」：(2,) を 2 個 → (2, 2)
a = train_ds[0][0]
b = train_ds[1][0]
print(a.shape, torch.stack([a, b]).shape)
# %%

# %% ステップ 6：わざと間違えてみる（つまずきの予防）
###########################################################
# (a) __len__ を書き忘れると？
class NoLen(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    # def __len__(self):                 # ← len(ds) と書くと呼ばれる。件数を返す
    #     return self.labels.shape[0]


ds = NoLen(X_train, y_train)
print(ds[0])                           # 1 件取るだけなら動く
try:
    DataLoader(ds, batch_size=2, shuffle=True)
except Exception as e:
    print(type(e).__name__, ":", e)

# (b) 1 件ごとに長さが違うと？（文章の長さがバラバラ、と同じ状況）
class Ragged(Dataset):
    def __init__(self):
        # self.data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
        self.data = [torch.tensor([2, 4]), torch.tensor([4, 5]), torch.tensor([6])]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

try:
    for xb in DataLoader(Ragged(), batch_size=2):
        print(xb)
except Exception as e:
    print(type(e).__name__, ":", str(e)[:80])

# batch_size=1 なら重ねないので動く
for xb in DataLoader(Ragged(), batch_size=1):
    print(xb)
# %%

# %% ステップ 7：ch02 の予告。数字の列を「入力」と「1 つずらした正解」に切る
###########################################################
class WindowDataset(Dataset):
    def __init__(self, ids, context_length):
        self.inputs = []
        self.targets = []
        for i in range(len(ids) - context_length):
            self.inputs.append(torch.tensor(ids[i:i + context_length]))
            self.targets.append(torch.tensor(ids[i + 1:i + context_length + 1]))

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

ids = list(range(10, 20))              # トークン ID のつもり：10, 11, ..., 19
ds = WindowDataset(ids, context_length=4)
print(len(ds))
print(ds[0])                           # 入力 [10,11,12,13] → 正解 [11,12,13,14]
print(ds[5])

loader = DataLoader(ds, batch_size=3, shuffle=False)
for x, y in loader:
    print(tuple(x.shape), tuple(y.shape))
    print(x)
    print(y)
    break                              # 最初のバッチだけ見る
# %%
