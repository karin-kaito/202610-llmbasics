# %% ステップ 10-1：あの 1 行は、2 行を 1 行に縮めた書き方
###########################################################
import torch

xb = torch.tensor([[-1.2, 3.1 ,1.,1.],
                   [-0.9, 2.9,1.,1.]])             # 2 件。1 件あたり数字 2 個

xb = torch.tensor([[-1.2, 3.1 ],
                   [-0.9, 2.9 ]])             # 2 件。1 件あたり数字 2 個

layer = torch.nn.Linear(4, 3)                # ① 層を作る（数字 4 個 → 3 個）
try:
    layer(xb)                                # ② その層に xb を入れる
except Exception as e:
    print(type(e).__name__, ":", e)
# %%
# %% ステップ 10-2：Linear(4, 3) の中身を見る
###########################################################
torch.manual_seed(123)
layer = torch.nn.Linear(4, 3)
print(layer.weight)                          # 3 行 × 4 個
print(layer.weight.shape)
# %%
# %% ステップ 10-3：数字 4 個なら、重み 4 個と 1 対 1 で掛けられる
###########################################################
x4 = torch.tensor([1.0, 2.0, 3.0, 4.0])      # 数字 4 個
w0 = layer.weight[0]                         # 出力 0 番の重み（4 個）
print(x4)
print(w0)
print(x4 * w0)                               # 4 個と 4 個：1 対 1 で掛けられる
print(layer(x4))                             # 層に入れても動く → 数字 3 個
# %%
# %% ステップ 10-4：数字 2 個だと、掛ける相手が足りない
###########################################################
x2 = torch.tensor([-1.2, 3.1])               # 数字 2 個（xb の 1 件め）
x2 = torch.tensor([-1.2, 3.1, 1., 1.])               # 数字 2 個（xb の 1 件め）
try:
    print(x2 * w0)                           # 2 個と 4 個：相手が合わない
except Exception as e:
    print(type(e).__name__, ":", e)

try:
    layer(x2)                                # 層に入れても同じ理由で落ちる
except Exception as e:
    print(type(e).__name__, ":", e)
# %%
# %% ステップ 10-5：直し方は 2 通り
###########################################################
print(torch.nn.Linear(2, 3)(xb).shape)       # (a) 層をデータに合わせる：4 → 2

xb4 = torch.rand((2, 4))                     # (b) データを層に合わせる：1 件 4 個
print(torch.nn.Linear(4, 3)(xb4).shape)
# %%
