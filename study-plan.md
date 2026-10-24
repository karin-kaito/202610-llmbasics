# LLM 基礎学習プラン（Gemma 4 理解のための基礎担当）

作成: 2026-10-24。事実は各リポジトリを 2026-10-24 に clone して確認した（下の「確認した事実」）。
確認できなかったものは「未確認」と書いている。

---

## 0. 方針（一言）

**③ LLMs-from-scratch を背骨（写経して動かす）にする。①は同じテーマを日本語で読み返す辞書、②は後半（LoRA・SFT・DPO）で合流、④ nanoGPT は区切りの復習として読む。**

理由：
- ③のリポには **`ch05/17_gemma4/`（Gemma 4 E2B/E4B を純 PyTorch で実装したノートブック）** と、そこに至る部品（`ch04/04_gqa`, `06_swa`, `10_kv-sharing`, `ch05/12_gemma3`）がある。GPT-2 から Gemma 4 まで同じ書き方の一本道で行けるのは４本の中で③だけ。
- ③の本編は「普通のノート PC で動く」と README に書いてある。15〜30 分の区切りでも進めやすい。
- ①は一読済みで、トークナイザ → Attention → GPT-2 → RoPE/SwiGLU/RMSNorm/KV キャッシュ → DPO まで、③と同じテーマを日本語で扱う。新しく写経するより「③で詰まったら①の該当箇所を読む」使い方の方が効く。
- ④ nanoGPT は README で本人が「very old and deprecated」と書き、後継の nanochat を案内している（2025年11月）。主教材にはしない。`model.py` と `train.py`（各 300 行ほど）を「通しで読めるか」の確認に使う。

---

## 1. 順番と並行の仕方

並行するのは **「写経 1 本 ＋ 読み返し 1 本」まで**。同時に写経を 2 本進めない。

| 段階 | 写経して動かす（主） | 並行して読む（従） | 合流点・ゴール |
|---|---|---|---|
| **A. GPT-2 を自分で組む** | ③ ch02〜ch05 | ① ch01（トークナイザ）、ch02（Attention〜GPT-2）、ch03（事前学習・生成） | ③ ch05.5 で OpenAI の GPT-2 重みを自作モデルに読み込めたら完了 |
| **A'. 復習** | ④ nanoGPT `model.py` を読む → shakespeare_char を小さい設定で回す | — | ③ の GPT と nanoGPT の対応表を自分で書けたら完了 |
| **B. GPT-2 → Gemma 4 の橋渡し** ★最重要 | ① ch05（RoPE / SwiGLU / RMSNorm / KV キャッシュ）→ ③ ch04 ボーナス（KV cache, GQA, SWA, KV-sharing）→ ③ ch05/12_gemma3 → ③ ch05/17_gemma4 | Gemma 4 側の学習（別の場）で出てきた用語を、ここで実装と結び付ける | Gemma 4 ノートの各クラスが GPT-2 のどこから変わったか説明できたら完了 |
| **C. 事後学習（ファインチューニング）** | ③ ch06〜07、付録 E（LoRA）→ ② 4.4 LoRA、5.2 SFT、5.3 DPO | ① ch06（AdamW・混合精度・DPO） | Kaggle や会社で Gemma を LoRA で調整する話が読めたら完了 |
| **D. 余裕があれば** | ② ch2（Transformer の原形・エンコーダ／デコーダ）、ch4.1〜4.3（分散学習） | ① ch09（DDP） | — |

②の扱い：途中まで読んだ②は**段階 A の間は止める**。②の ch2 は「元祖 Transformer（エンコーダ／デコーダ）」なので、GPT 系（デコーダのみ）に慣れた後の方が読みやすい。②の強みは後半（LoRA・SFT・DPO の Colab ノート、日本語データ）なので段階 C で合流させる。

---

## 2. chat と Claude Code の使い分け

| やること | 使う道具 | 例 |
|---|---|---|
| 理論の章・図の意味を読む（③ ch01、各章の前置き、①の読み返し） | **chat** | 「Q·Kᵀ を √d で割る理由を、数値例で」 |
| 写経・実行・エラー対応 | **Claude Code** | 「`work/ch03` の自分の実装と公式 `ch03.ipynb` の差分を出して、どこで形が合わなくなったか」 |
| 複数ファイルを並べて読み比べる（段階 A', B） | **Claude Code**（リポを手元に clone して） | 「③ ch04 `gpt.py` の TransformerBlock と `17_gemma4` の `Gemma4DenseBlock` を表で比べて」 |
| 理解度チェック | **chat** | 「この章の内容で 5 問出して。答えは後で」 |

ルール：**自分で写経したコードは Claude Code に書かせない**。Claude Code は「動かない理由を探す」「公式との差分を見る」「読み比べ表を作る」に使う。

---

## 3. ローカルかクラウドか

**未確認のこと：自宅 PC の GPU（型番・VRAM）。** これが分かれば確定できる。以下は分岐で書く。

| 教材・章 | README／コードから分かったこと | おすすめ |
|---|---|---|
| ③ 本編 ch02〜07 | 「普通のノート PC で動く・特別なハードは不要」「M3 MacBook Air で全章テスト済み」「GPU があれば自動で使う」。PyTorch ≥ 2.2.2 | **ローカル** |
| ③ `ch05/17_gemma4` | Gemma 4 E2B/E4B の実重みを Hugging Face から取得。**HF でライセンス同意とアクセストークンが必要**。必要メモリは README に記載なし（**未確認**） | GPU が小さければ **Colab**。参考：同じ作者の Gemma 3 ノートは出力に「GPU memory used: 0.96 GB」とあるが、Gemma 4 の値は**未確認** |
| ① ch01〜07 | PyTorch 2.x（requirements では torch ≥ 2.5.0）、Python ≥ 3.10。CUDA / Apple MPS / CPU を自動選択 | **ローカル**（重い事前学習だけ Colab） |
| ① ch09 | `torchrun` 前提の DDP（複数 GPU、`nccl`）のコード | 読むだけ。動かすなら複数 GPU のクラウド |
| ② ch2〜3、4.4、5.x | 各ノートに「Open in Colab」バッジあり。**Python は 3.12 限定（≥3.12,<3.13）**、torch ≥ 2.1.1、`peft==0.14.0`、`trl==0.9.6` など版の固定が多い。ch4.4 は `rinna/japanese-gpt2-medium` を使う | **Colab**（版合わせの手間が少ない） |
| ② 4.2・4.3 | `torchrun` / `deepspeed` で動かす分散学習の `.py` | 読むだけ（Windows で DeepSpeed が動くかは**未確認**） |
| ④ nanoGPT | shakespeare_char は A100 1 枚で約 3 分、CPU では小さい設定で約 3 分（README）。GPT-2 124M の再現は 8×A100 40GB で約 4 日。README には `torch.compile` が「Windows では使えないことがある → `--compile=False`」とある（2022 年時点の記述。今も当てはまるかは**未確認**） | **ローカル**で小さい設定のみ |

Colab Pro について：使える GPU の種類やコンピューティングユニットの消費量は、情報源によって数字が食い違っている。公式ページでは**未確認**。契約前に Google の公式ページで確認すること。

---

## 4. コードと記録の置き場所

```
E:\study\llm-basics\            ← このリポジトリ（202610-llmbasics）を clone する場所
├─ study-plan.md                ← この文書
├─ progress.md                  ← 完了／未完了／次にやること
├─ refs\                        ← 教材リポの clone（読むだけ・編集しない・.gitignore で除外）
│   ├─ deep-learning-from-scratch-6\
│   ├─ book-llm-from-scratch\
│   ├─ LLMs-from-scratch\
│   └─ nanoGPT\
├─ work\                        ← 自分の写経（教材ごと・章ごと）
│   ├─ rasbt\ch03\attention.py
│   └─ dlfs6\ch05\rope.py
└─ notes\                       ← 1 回ごとのメモ・用語集・比較表
    ├─ glossary.md              ← Gemma 4 側で出た用語 ↔ 実装箇所
    └─ gpt2-vs-gemma4.md        ← 段階 B の成果物
```

- Kaggle 用リポとは**別リポ・別フォルダ**。
- `refs\` の教材は編集しない（更新を `git pull` で取り込めるように）。手を加えるときは `work\` にコピーしてから。
- Windows 環境なら、Python の版を教材ごとに分けるために `uv` を使う（①②のリポは `uv.lock` を同梱）。

---

## 5. 1 回 15〜30 分の区切り案

1 回の型：**① 読む 5 分 → ② 写経・実行 15〜20 分 → ③ メモ 3 行（分かった／詰まった／次）を progress.md に書く**。
1 回で終わらなかったら、その節を 2 回に分ける（無理に終わらせない）。

### 段階 A（③ ch02〜05）— 約 25 回

| 回 | 内容 | 並行して読む① |
|---|---|---|
| 1 | ③ 環境構築（`setup/`）、ch01 を chat で読む | — |
| 2〜3 | ch02 2.1〜2.4（トークン化・ID・特殊トークン） | ① ch01 文字・バイトトークナイザ |
| 4 | ch02 2.5 BPE（tiktoken） | ① ch01 BPE |
| 5〜6 | ch02 2.6〜2.8（スライディングウィンドウ・埋め込み・位置埋め込み） | — |
| 7 | ch03 3.1〜3.3（重みなし self-attention） | ① ch02 `01_soft_dict`〜`02_attn_math` |
| 8 | ch03 3.4（学習する重み付き self-attention） | ① ch02 `03_attn_scaling` |
| 9 | ch03 3.5（因果マスク・dropout） | ① ch02 `06_attn_mask` |
| 10〜11 | ch03 3.6（マルチヘッド） | ① ch02 `08_multi_head` |
| 12 | ch04 4.1〜4.2（全体像・LayerNorm） | ① ch02 `09_norm_gelu` |
| 13 | ch04 4.3〜4.4（GELU の FFN・残差接続） | — |
| 14〜15 | ch04 4.5〜4.6（Transformer ブロック・GPT モデル） | ① ch02 `10_gpt2` |
| 16 | ch04 4.7（テキスト生成） | — |
| 17〜18 | ch05 5.1（損失・パープレキシティ） | ① ch03 `01_pretrain` |
| 19〜21 | ch05 5.2（学習ループ） | — |
| 22 | ch05 5.3（temperature・top-k） | ① ch03 `02_generate` |
| 23 | ch05 5.4（保存・読み込み） | — |
| 24〜25 | ch05 5.5（GPT-2 公式重みの読み込み）→ **段階 A 完了** | — |

### 段階 A'（④ nanoGPT）— 約 4 回
26. `model.py` を読み、③ ch04 との対応表を作る（Claude Code で並べる）
27. `train.py` を読み、③ ch05 との違い（勾配累積・学習率スケジュール・`torch.compile`）を書き出す
28〜29. shakespeare_char を小さい設定で学習 → `sample.py` で生成

### 段階 B（GPT-2 → Gemma 4）— 約 12 回
30. ① ch05 `01_rope`（RoPE）
31. ① ch05 `02_swiglu`、`03_rmsnorm`
32. ① ch05 `05_kvcache` ／ ③ `ch04/03_kv-cache`
33〜34. ③ `ch04/04_gqa`（GQA とメモリ見積もり）
35. ③ `ch04/06_swa`（スライディングウィンドウ注意）
36. ③ `ch04/10_kv-sharing`（KV の層間共有。Gemma 4 E2B/E4B のメモリ図あり）
37〜38. ③ `ch05/12_gemma3`（GPT-2 との差分を表に）
39〜41. ③ `ch05/17_gemma4`（per-layer embeddings、sliding / full の層の並び、`global_head_dim`）→ `notes/gpt2-vs-gemma4.md` を仕上げる

### 段階 C 以降
③ ch06・ch07・付録 E はそれぞれの節を 1 回の単位にする（ch07 は 7.1〜7.8 の節がある）。② の 4.4 / 5.2 / 5.3 は Colab ノート 1 本を 2〜3 回に分ける。

---

## 確認した事実（2026-10-24 時点の clone）

- **① deep-learning-from-scratch-6**：README は「`ch01`〜`ch09` 各章のコード」と書いているが、実際のフォルダは ch01〜ch07 と ch09 で、**ch08 のフォルダは無い**（ch08 にコードが無いだけか、理由は未確認）。ほかに `codebot` / `storybot` / `webbot`。章タイトルは README に無く、Web でも目次は見つからなかった（**未確認**）。中身はファイル名から推測：ch01 トークナイザ、ch02 Attention〜GPT-2、ch03 事前学習・SFT・GRPO、ch04 BPE の高速化、ch05 RoPE・SwiGLU・RMSNorm・KV キャッシュ、ch06 AdamW・学習率スケジューラ・混合精度・DPO・LLM-as-a-Judge、ch07 BPE、ch09 DDP 事前学習。Python ≥ 3.10、torch ≥ 2.5.0。
- **② book-llm-from-scratch**：全 6 章＋付録（1 歴史、2 Transformer、3 GPT、4 学習（大規模化・分散・LoRA）、5 アラインメント（SFT・DPO）、6 推論モデル、付録 NumPy/PyTorch）。ノートがあるのは 2〜5 章。Python 3.12 限定。
- **③ LLMs-from-scratch**：本編 7 章＋付録 A〜E。本編はノート PC で動く（README）。ボーナスに `ch05/17_gemma4`（Gemma 4 E2B/E4B、テキストのみ、dense）、`ch05/12_gemma3`、`ch04/04_gqa`・`06_swa`・`07_moe`・`10_kv-sharing` などがある。日本語版『つくりながら学ぶ！LLM自作入門』（マイナビ出版、2025 年）は原書の翻訳。日本語版の目次が原書と同じかは、冒頭 4 章以外**未確認**。また、日本語版は原書の刊行時点の内容なので、**Gemma 4 などのボーナスは GitHub だけにあり、書籍には載っていない**と考えられる（未確認）。
- **④ nanoGPT**：最後のコミットは 2025-11-12。README で deprecated と明記し、nanochat を案内。
