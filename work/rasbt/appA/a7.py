# A0-6：③ 付録 A の A.7（学習ループ）と A.8（保存と読み込み）
# 教材：refs/LLMs-from-scratch/appendix-A/01_main-chapter-code/code-part1.ipynb
#       （2026-11-09 時点の教材リポと照合、PyTorch 2.15.0 で動作確認済み）
# 進め方：上から 1 セルずつ「Run Cell」。1 セル 1 話題。

# %% ステップ 0：準備（A.5 のモデル と A.6 のデータ を再掲。読むだけでよい）
###########################################################
import torch
from torch.utils.data import Dataset, DataLoader

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

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

torch.manual_seed(123)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),   # 数字 2 個 → 30 個
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),           # 30 個 → 20 個
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),  # 20 個 → 2 個（クラス 0 と 1 の点数）
        )
    def forward(self, x):
        return self.layers(x)

print("準備 OK")

# %% ステップ 1-1：モデルに 1 バッチ入れると、何が出る？
###########################################################
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

features, labels = next(iter(train_loader))  # バッチを 1 つだけ取り出す
print("features:", features)                 # (2, 2)：2 件 × 数字 2 個
print("labels  :", labels)                   # 正解のクラス番号が 2 個

logits = model(features)
print("logits  :", logits)                   # (2, 2)：2 件 × 「クラス 0 の点数, クラス 1 の点数」
print(logits.shape)
# まだ学習していないので、点数はでたらめ。

# %% ステップ 1-2：損失（loss）は「どれだけ外れているか」を表す数字 1 個
###########################################################
import torch.nn.functional as F

loss = F.cross_entropy(logits, labels)
print("loss:", loss)                         # 数字 1 個。小さいほど良い

# cross_entropy を手でやってみる（同じ数字になるか確認）
probas = torch.softmax(logits, dim=1)        # (a) 点数 → 確率（1 件ごとに合計 1）
print("確率:", probas)
p0 = probas[0, labels[0]]                    # (b) 1 件め：正解クラスの確率
p1 = probas[1, labels[1]]                    #     2 件め：正解クラスの確率
print("正解クラスの確率:", p0, p1)
my_loss = (-torch.log(p0) - torch.log(p1)) / 2   # (c) -log をとって平均
print("手で計算:", my_loss)
# 正解の確率が 1 に近いほど -log は 0 に近い → loss が小さい

# %% ステップ 1-3：1 回だけ学習する（3 行で重みが少し動く）
###########################################################
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # 重みを動かす係。lr = 1 回に動かす量

w = model.layers[0].weight                   # 1 つめの Linear の重み (30, 2)
print("前の重み 1 個:", w[0, 0].item())
print("前の grad   :", w.grad)               # まだ None

logits = model(features)
loss = F.cross_entropy(logits, labels)
print("前の loss:", loss.item())

optimizer.zero_grad()                        # ① grad を 0 に戻す
loss.backward()                              # ② grad を計算する（各重みを「どっちに動かせば loss が減るか」）
print("grad 1 個:", w.grad[0, 0].item())
optimizer.step()                             # ③ 重み ← 重み − lr × grad

print("後の重み 1 個:", w[0, 0].item())       # 前の重み − 0.5 × grad になっている
loss_after = F.cross_entropy(model(features), labels)
print("後の loss:", loss_after.item())       # 同じバッチなら下がる

# %% ステップ 1-4：わざと間違える。zero_grad を忘れると？
###########################################################
torch.manual_seed(123)
m = NeuralNetwork(2, 2)
w = m.layers[0].weight

for i in range(3):
    loss = F.cross_entropy(m(features), labels)
    loss.backward()                          # zero_grad なしで backward だけ 3 回
    print(i, "grad 1 個:", w.grad[0, 0].item())
# grad は上書きではなく「足し算」でたまっていく → 毎回 zero_grad で 0 に戻す

# %% ステップ 2：本の学習ループ（ステップ 1-3 の 3 行を、バッチ × エポックで繰り返すだけ）
###########################################################
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):              # エポック = データ全部を 1 周
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()                # ①
        loss.backward()                      # ②
        optimizer.step()                     # ③

        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

    model.eval()
# loss が下がっていけば学習できている
# 試す：num_epochs を 1 や 10 に、lr を 0.01 や 5.0 に変えて、loss の動きを見る

# %% ステップ 3-1：学習したモデルで予測する
###########################################################
model.eval()                                 # 「評価モード」にする（この小さいモデルでは結果は同じ。習慣として書く）
with torch.no_grad():                        # 予測だけなので grad の記録をしない（速い・省メモリ）
    outputs = model(X_train)
print(outputs)                               # (5, 2)：5 件 × 2 クラスの点数

# %% ステップ 3-2：点数 → 確率 → 予測クラス
###########################################################
torch.set_printoptions(sci_mode=False)       # 0.0001 を 1e-04 と書かない設定
probas = torch.softmax(outputs, dim=1)       # dim=1：1 行（1 件）ごとに合計 1 にする
print(probas)
print(probas.sum(dim=1))                     # 1 件ごとの合計 → 全部 1

predictions = torch.argmax(probas, dim=1)    # 1 行ごとに「いちばん大きい列の番号」
print(predictions)

print(torch.argmax(outputs, dim=1))          # softmax を通さなくても同じ答え（大小の順は変わらないから）

# %% ステップ 3-3：正解と比べて数える
###########################################################
print(predictions)
print(y_train)
compare = predictions == y_train             # 1 件ずつ一致したか（True / False）
print(compare)
print(torch.sum(compare))                    # True を 1 として数える → 正解数

# %% ステップ 4：正解率を計算する関数（本の compute_accuracy）
###########################################################
def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0.0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()

print("train:", compute_accuracy(model, train_loader))
print("test :", compute_accuracy(model, test_loader))
# 1.0 = 100% 正解
# 注意：train_loader は drop_last=True なので 5 件中 4 件しか数えていない

# %% ステップ 5-1：保存する（A.8）
###########################################################
sd = model.state_dict()                      # 「名前 → 重みテンソル」の辞書
for name, t in sd.items():
    print(name, tuple(t.shape))
# 保存されるのは「重みの数字」だけ。クラスの設計図（NeuralNetwork）は保存されない

torch.save(sd, "model.pth")                  # 今いるフォルダに model.pth ができる（.gitignore で commit 対象外）

# %% ステップ 5-2：読み込む（新しいモデルを作って、重みを流し込む）
###########################################################
torch.manual_seed(999)
new_model = NeuralNetwork(2, 2)              # ① 同じ設計図で作る（中身は別のでたらめな重み）
with torch.no_grad():
    print("読む前:", new_model(X_test))       # 点数が model と違う

state = torch.load("model.pth", weights_only=True)   # ② ファイル → 辞書
new_model.load_state_dict(state)                     # ③ 辞書 → モデルに流し込む
with torch.no_grad():
    print("読んだ後:", new_model(X_test))
    print("元の model:", model(X_test))      # 同じ数字になる
print("test 正解率:", compute_accuracy(new_model, test_loader))

# %% ステップ 5-3：わざと間違える。設計図が違うモデルに読み込むと？
###########################################################
wrong = NeuralNetwork(2, 3)                  # 出力を 3 個にした（保存したのは 2 個）
try:
    wrong.load_state_dict(torch.load("model.pth", weights_only=True))
except Exception as e:
    print(type(e).__name__, ":", str(e)[:300])
# 本のコメント「needs to match the original model exactly」の意味：
# 形が 1 か所でも違うと読み込めない。だから設計図（クラスと引数）は自分で同じものを用意する
# %%
