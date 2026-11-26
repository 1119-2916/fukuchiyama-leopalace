# 対象ベースモデル（選定済み 2026-11-26）

LoRA を作る対象として以下の 4 モデルを選定した。
いずれも「生成用」と「学習用」でチェックポイントが分かれるモデルがあり、
**LoRA 学習は原則として Base（未蒸留）側で行う**点が共通している。

| モデル | 学習に使うチェックポイント | 規模 / 構成 | 学習ツール（実績） | ライセンス上の注意 |
|---|---|---|---|---|
| Anima | Anima Base（circlestone-labs/Anima） | ~2B DiT（Cosmos-Predict2 由来）、Qwen3-0.6B TE、Qwen-Image VAE | kohya sd-scripts（`anima_train_network.py`） | 重みは CircleStone Labs Non-Commercial License（＋NVIDIA Open Model License の派生条件）。**商用不可**に注意 |
| Moss（MOSS-Image） | MOSS-Image-Base（OpenMOSS-Team） | ~5B 単一ストリーム DiT、Qwen3.5-9B TE、FLUX.2 VAE | kohya sd-scripts（`moss_image_train_network.py`）、DiffSynth-Studio、AI Toolkit 等 | Apache 2.0 |
| Z-Image | Z-Image Base（Tongyi-MAI/Z-Image）※ Turbo は不可 | ~6B S3-DiT、Qwen3 TE | Diffusers（DreamBooth LoRA 例あり）、各種コミュニティツール | Apache 2.0 |
| Flux-Klein | Klein の **Base**（蒸留版は不可） | 世代により異なる（下記） | AI Toolkit、Diffusers 等 | 世代により異なる（下記） |

## モデルごとの注意点

### Anima
- アニメ特化。プロンプトは Danbooru タグと自然文の併用が可能で、
  学習データもタグ形式キャプションと相性がよい
- 学習率の目安（コミュニティ実績）: rank 32 で 2e-5 程度、LLM adapter は学習しない

### Moss（MOSS-Image）
- **学習は Base、生成の出来栄え確認は SFT** という使い分けが公式の想定
- テキストエンコーダ（Qwen3.5-9B）は凍結するのが公式ルール
- キャプションは「長く逐語的」な形式で学習されている。公式の recaptioning は
  長文キャプション＋見える文字の正確な転記を求める形式で、
  **Danbooru タグ形式はこのモデルの流儀に合わない**
  → Moss 向けは `--style detailed` の自然文キャプションを使う（下記）

### Z-Image
- Turbo は蒸留済みで LoRA 学習には使わない。学習は未蒸留の Base で行い、
  Turbo 用 LoRA と Base 用 LoRA は互換性がない（混用すると出力が壊れる報告あり）
- 自然文（英・中）の理解が強いモデル。キャプションは自然文寄りが無難

### Flux-Klein ― 世代の確認が必要
「Flux-Klein」には世代が複数ある。どれを指すかでライセンスと手順が変わる:
- FLUX.2 [klein] Base 4B / 9B: Apache 2.0。学習用は Base（50 step 側）
- FLUX.3 Image Klein Base（5.6B）: FLUX Kommunity License（ゲート付き）。
  Base で学習した LoRA は Distilled 側にも載せられる設計
- どちらでも、学習は Base、生成の高速化は Distilled、という役割分担は同じ

## キャプション形式はモデルごとに変える

現在の生成フロー（docs/captioning.md）はモデル別にキャプション形式を
出し分けられる。学習データの置き方はモデルごとにフォルダを分ける想定:

- Anima 向け: `--style tags`（Danbooru タグ）
- Moss / Z-Image / Flux-Klein 向け: `--style detailed`（自然文・詳細）
  - 特に Moss は公式流儀に合わせ、見える文字の転記を含めるか検討する

`caption_batch.py --output-dir <dir>` で、画像フォルダとは別の場所に
モデル別キャプション一式を書き出せる（ファイル名は画像と同名 `.txt`）。
学習時は「画像＋そのモデル用キャプション」を組にして各ツールの
データセットフォルダへ配置する。
