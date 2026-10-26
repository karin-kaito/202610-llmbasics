# progress

## 完了事項
- 2026-10-24 学習プラン作成（study-plan.md）。４教材のリポで章数・必要環境を確認した
- 2026-10-24 自宅 GPU は Quadro P400 2GB → 自宅は CPU 運用、GPU が要るもの（Gemma 実重み・LoRA/SFT/DPO・②）は Colab に決定
- 2026-10-24 GitHub を `main` 1 本に整理（デフォルトブランチ = main、作業ブランチは削除済み）
- 2026-10-24 ローカルに clone 済み：`E:\study\llm-basics`

## 未完了事項
- 自宅 PC の CPU・メモリ容量を確認（③ ch05 の学習ループの速さに影響）
- Colab Pro の GPU とユニットを公式ページで確認（第三者情報は食い違いあり）
- ③ `ch05/17_gemma4` の必要メモリ（未確認。設定値からの概算では埋め込みだけで bf16 約 5.5GB）
- ①の章タイトル（未確認）、①に ch08 フォルダが無い理由（未確認）

## 次にやること（段階 A 第 1 回・約 30 分）
1. 教材を `refs\` に clone（`refs\` は .gitignore 済み）
   ```powershell
   cd E:\study\llm-basics
   git clone https://github.com/rasbt/LLMs-from-scratch.git refs\LLMs-from-scratch
   git clone https://github.com/oreilly-japan/deep-learning-from-scratch-6.git refs\deep-learning-from-scratch-6
   git clone https://github.com/elith-co-jp/book-llm-from-scratch.git refs\book-llm-from-scratch
   git clone https://github.com/karpathy/nanoGPT.git refs\nanoGPT
   ```
2. ③用の環境（③の README は Windows なら `uv add` ではなく `uv pip` か pip を推奨）
   ```powershell
   pip install uv
   uv venv --python 3.12
   .venv\Scripts\activate
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu
   uv pip install -r refs\LLMs-from-scratch\requirements.txt
   ```
   エラーが出たら普通の `pip install -r ...` に切り替える。
3. 確認：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → `False` なら狙いどおり（CPU）。`jupyter lab` で `refs\LLMs-from-scratch\ch02\01_main-chapter-code\ch02.ipynb` が開けば完成
4. ③ ch01（コードなし）を chat で読む → 下にメモ 3 行 → commit・push

## 回ごとのメモ（分かった／詰まった／次）
<!-- 例: 2026-10-25 ③ch03 3.4 ／ 分かった: W_q, W_k, W_v の形 ／ 詰まった: transpose の次元 ／ 次: 3.5 -->
- 2026-10-24 ／ 計画づくりとリポ準備のみ ／ 次: 段階 A 第 1 回
- 2026-10-26 ／ 現状確認のみ（作業は進めていない。未完了事項・次にやることは 10-24 から変更なし） ／ 次: 段階 A 第 1 回（上の手順 1〜4）
