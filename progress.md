# progress

## 完了事項
- 2026-10-24 学習プラン作成（study-plan.md）。４教材のリポで章数・必要環境を確認した
- 2026-10-24 自宅 GPU は Quadro P400 2GB → 自宅は CPU 運用、GPU が要るもの（Gemma 実重み・LoRA/SFT/DPO・②）は Colab に決定
- 2026-10-24 GitHub を `main` 1 本に整理（デフォルトブランチ = main、作業ブランチは削除済み）
- 2026-10-24 ローカルに clone 済み：`E:\study\llm-basics`
- 2026-10-31 段階 A 第 1 回の環境構築完了：教材 4 本を `refs/` に clone、VS Code＋Git Bash＋uv で `.venv` 作成、③ `ch02.ipynb` の 1 つ目のセルが実行できた（下の「環境メモ」）

## 未完了事項
- 自宅 PC の CPU・メモリ容量を確認（③ ch05 の学習ループの速さに影響）
- Colab Pro の GPU とユニットを公式ページで確認（第三者情報は食い違いあり）
- ③ `ch05/17_gemma4` の必要メモリ（未確認。設定値からの概算では埋め込みだけで bf16 約 5.5GB）
- ①の章タイトル（未確認）、①に ch08 フォルダが無い理由（未確認）

## 次にやること（段階 A 第 1 回の残り → 第 2 回）
1. ③ ch01（コードなし）を chat で読む → 下にメモ 3 行 → commit・push
2. 第 2 回：③ ch02 2.1〜2.4（トークン化・ID・特殊トークン）。写経は `work/rasbt/ch02/` に自分のファイルを作る
   - `refs/` のノートは実行してもよいが**保存しない**（教材を書き換えないため）

## 環境メモ（2026-10-31 構築・自宅 PC）
- エディタ：VS Code（フォルダ信頼済み）、ターミナル：Git Bash
- 仮想環境：`.venv`（uv で作成、Python 3.12.13。PC 本体の Python 3.14 とは別）
- 入れたもの：torch 2.15.0+cpu、tiktoken 0.14.0、jupyterlab、ipykernel、matplotlib、tqdm、numpy、pandas
- **未導入：tensorflow**（③ ch05 の `gpt_download.py` で GPT-2 重みを読むときに必要。ch05 で入れる）
- Copilot のインライン補完はこのフォルダだけ停止（`.vscode/settings.json`）。写経は自分で書く
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
- 2026-10-31 ／ 約 1 週間ぶりに再開。VS Code で環境構築（手順 1〜3）完了。TF は ch05 まで保留 ／ 次: ③ ch01 を chat で読む
