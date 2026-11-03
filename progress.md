# progress

## 完了事項
- 2026-10-24 学習プラン作成（study-plan.md）。４教材のリポで章数・必要環境を確認した
- 2026-10-24 自宅 GPU は Quadro P400 2GB → 自宅は CPU 運用、GPU が要るもの（Gemma 実重み・LoRA/SFT/DPO・②）は Colab に決定
- 2026-10-24 GitHub を `main` 1 本に整理（デフォルトブランチ = main、作業ブランチは削除済み）
- 2026-10-24 ローカルに clone 済み：`E:\study\llm-basics`
- 2026-10-31 段階 A 第 1 回の環境構築完了：教材 4 本を `refs/` に clone、VS Code＋Git Bash＋uv で `.venv` 作成、③ `ch02.ipynb` の 1 つ目のセルが実行できた（下の「環境メモ」）
- 2026-11-03 までに ② を読了（残りの数章＋付録）
- 2026-11-03 までに ③ ch01 を読了
- 2026-11-03 までに ③ 付録 A の A.1〜A.4（段階 A0-1〜A0-3）を読了

## 未完了事項
- 自宅 PC の CPU・メモリ容量を確認（③ ch05 の学習ループの速さに影響）
- Colab Pro の GPU とユニットを公式ページで確認（第三者情報は食い違いあり）
- ③ `ch05/17_gemma4` の必要メモリ（未確認。設定値からの概算では埋め込みだけで bf16 約 5.5GB）
- ①の章タイトル（未確認）、①に ch08 フォルダが無い理由（未確認）

## 次にやること（段階 A0：③ 付録 A で PyTorch を固める → 段階 A へ）
きっかけ：② 付録の PyTorch の章で理解不足に気づいた。①でも PyTorch／実装の所で詰まっていた。
③ README も「PyTorch が初めてなら付録 A を」と案内している（2026-11-03 確認）。

1. ③ 付録 A を節ごとに進める（区切りは study-plan.md「段階 A0」）。コードは `work/rasbt/appA/` に置く
   - **進め方（2026-11-03 変更）**：Claude がステップごとにコピペできるコードを出す → 自宅 PC（ローカル、`.venv`）で素早くなぞって動かす → 少し変えて試す、を何度も繰り返す
   - 教材ノート：`refs/LLMs-from-scratch/appendix-A/01_main-chapter-code/code-part1.ipynb`（A.1〜A.8）
   - `refs/` のノートは実行してもよいが**保存しない**
   - ①で詰まった所が付録 A のどの節に当たるかを、回ごとのメモに書く
   - **①で詰まったのは `nn.Module` の使い方** → A0-4（A.5）を重点にする
2. 次の回：**A0-4（A.5 `nn.Module` で多層ネット）**。その後 A0-5（A.6）、A0-6（A.7〜A.8）
3. 付録 A が終わったら：③ ch02 2.1〜2.4（元の第 2 回）

## 環境メモ（2026-10-31 構築・自宅 PC）
- エディタ：VS Code（フォルダ信頼済み）、ターミナル：Git Bash
- 仮想環境：`.venv`（uv で作成、Python 3.12.13。PC 本体の Python 3.14 とは別）
- 入れたもの：torch 2.15.0+cpu、tiktoken 0.14.0、jupyterlab、ipykernel、matplotlib、tqdm、numpy、pandas
- **未導入：tensorflow**（③ ch05 の `gpt_download.py` で GPT-2 重みを読むときに必要。ch05 で入れる）
- Copilot のインライン補完はこのフォルダだけ停止（`.vscode/settings.json`）。写経は自分で書く
- コードの書き方（2026-11-03 決定）：**vi で `.py` を書き、`# %%` の行でセルに区切る** → VS Code で開くと「Run Cell」でセルごとに実行できる（確認済み）。`.ipynb` は中身が JSON なので vi で作らない（VS Code で新規作成する）
- 毎回の始め方（Git Bash）：
  ```bash
  cd /e/study/llm-basics
  source .venv/Scripts/activate
  ```
- 作り直すときの手順（Git Bash）：
  ```bash
  uv venv --python 3.12
  source .venv/Scripts/activate
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install jupyterlab ipykernel tiktoken matplotlib tqdm numpy pandas
  ```

## 回ごとのメモ（分かった／詰まった／次）
<!-- 例: 2026-10-25 ③ch03 3.4 ／ 分かった: W_q, W_k, W_v の形 ／ 詰まった: transpose の次元 ／ 次: 3.5 -->
- 2026-10-24 ／ 計画づくりとリポ準備のみ ／ 次: 段階 A 第 1 回
- 2026-10-26 ／ 現状確認のみ（作業は進めていない。未完了事項・次にやることは 10-24 から変更なし） ／ 次: 段階 A 第 1 回（上の手順 1〜4）
- 2026-10-31 ／ 約 1 週間ぶりに再開。VS Code＋Git Bash で環境構築（手順 1〜3）完了。TF は ch05 まで保留。git：ブランチ整理の流れ（fetch --prune → branch -a → branch -r --merged main → push origin --delete）と、remotes/origin は GitHub の様子を写したメモであることを学んだ ／ 次: ③ ch01 を読んで報告 → 第 2 回 ch02 2.1〜2.4
- 2026-11-03 ／ 約 3 日ぶり。② 読了、③ ch01 読了。② 付録の PyTorch で理解不足に気づいた（①でも PyTorch／実装の所で詰まっていた）→ ③ 付録 A（PyTorch 入門）を読み始めた。計画に段階 A0（付録 A）を足した ／ 次: ③ 付録 A を節ごとに
- 2026-11-03 ③ ch01 ／ 分かった: LLM の概観は概ね理解。GPT-3 の事前学習データセットの種類・サイズは他の本に無かった情報 ／ 詰まった: 特になし。Transformer は概説レベルの理解なので、詳細は ③ ch03〜04 で ／ 次: 付録 A
- 2026-11-03 ③ 付録 A A.1〜A.4（A0-1〜A0-3）を読了 ／ ①で詰まったのは `nn.Module` の使い方 ／ 次: A0-4（A.5）
- 2026-11-03 ／ 進め方を変更：写経ではなく、Claude が出すステップごとのコードをコピペで素早くなぞる。A0〜A0-6 はローカル（Colab 不要。A.9 GPU だけは読むのみ） ／ 次: A0-4 をステップ 1〜6 で
- 2026-11-03 ／ 分かった: `.ipynb` の中身は JSON なので vi では作れない。`.py` に `# %%` を入れると VS Code でセルごとに実行できた → `a5.py` 方式で進める ／ 次: A0-4 ステップ 1〜6
- 2026-11-03 ／ 分かった: `Linear(50, 30)` の weight が (30, 50) なのは「1 行 = 出力ニューロン 1 個ぶんの重み」だから。計算は `X @ W.T + b`。ステップ 6 を直して試した（`self` に持たせれば登録される。forward が古い属性を見ていると、パラメータ数は合っても `model(X)` で落ちる） ／ 教材: `notes/linear-cheatsheet.html`（nn.Linear 早見表、claude.ai に Artifact としても公開・非公開設定）／ 次: A0-5（A.6 Dataset／DataLoader）
