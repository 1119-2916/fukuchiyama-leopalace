# 対象ベースモデル（選定済み 2026-11-26）

LoRA を作る対象として以下の 4 モデルを選定した。
いずれも「生成用」と「学習用」でチェックポイントが分かれるモデルがあり、
**LoRA 学習は原則として Base（未蒸留）側で行う**点が共通している。

| モデル | 学習に使うチェックポイント | 規模 / 構成 | 学習ツール（実績） | ライセンス上の注意 |
|---|---|---|---|---|
| Anima | Anima Base（circlestone-labs/Anima） | ~2B DiT（Cosmos-Predict2 由来）、Qwen3-0.6B TE、Qwen-Image VAE | kohya sd-scripts（`anima_train_network.py`） | 重みは CircleStone Labs Non-Commercial License（＋NVIDIA Open Model License の派生条件）。**商用不可**に注意 |
| Moss（MOSS-Image） | MOSS-Image-Base（OpenMOSS-Team） | ~5B 単一ストリーム DiT、Qwen3.5-9B TE、FLUX.2 VAE | kohya sd-scripts（`moss_image_train_network.py`）、DiffSynth-Studio、AI Toolkit 等 | Apache 2.0 |
| Z-Image | Z-Image Base（Tongyi-MAI/Z-Image）※ Turbo は不可 | ~6B S3-DiT、Qwen3 TE | Diffusers（DreamBooth LoRA 例あり）、各種コミュニティツール | Apache 2.0 |
| Flux-Klein | FLUX.3 Image Klein **Base**（`black-forest-labs/flux-3-image-klein-base`） | 5.6B、Qwen3-VL 系テキスト条件付け、編集・マルチ参照対応 | AI Toolkit（`arch: flux3`）、Diffusers | FLUX Kommunity License（ゲート付き、Apache 2.0 ではない） |

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

### Flux-Klein ― FLUX.3 Image Klein Base に確定（2026-11-26）
- 対象は FLUX.2 ではなく **FLUX.3 Image Klein Base（5.6B）** と確定した
- Hugging Face: `black-forest-labs/flux-3-image-klein-base`
- ライセンスは FLUX Kommunity License（ゲート付き）。FLUX.2 klein の
  Apache 2.0 とは異なるので、LoRA の公開条件はこのライセンスに従う
- 役割分担: 学習は Base（50 steps / CFG 3.5 目安）、高速生成は Distilled
  （16 steps / CFG 1）。Base で学習した LoRA は Distilled 側にも
  そのまま載せられる設計
- 学習ツール: AI Toolkit（`arch: flux3`）または Diffusers
- テキスト条件付けは Qwen3-VL 系。プロンプトは自然文が前提のため、
  キャプションは `--style detailed` を使う

## キャプション形式はモデルごとに変える

現在の生成フロー（docs/captioning.md）はモデル別にキャプション形式を
出し分けられる。学習データの置き方はモデルごとにフォルダを分ける想定:

- Anima 向け: `--style tags`（Danbooru タグ）
- Moss / Z-Image / Flux-Klein 向け: `--style detailed`（自然文・詳細）
  - 特に Moss は公式流儀に合わせ、見える文字の転記を含めるか検討する

スクリプト側の対応（実装済み）: `caption_batch.py --target-model
<anima|moss|z-image|flux-klein>` が、モデルごとの形式・プロンプト・
出力先（`dataset-captions/<model>`）をプリセットとして適用する。
Moss 向けには `long` 形式（5〜8文の逐語的キャプション＋見える文字の
正確な転記）と専用プロンプト `prompts/caption_moss.txt` を用意した。
ただし Moss 公式レシピは画風の記述も求めるのに対し、本プロジェクトでは
「画風はトリガーワードに結びつける」方針のため、その点だけ意図的に
外してある（プロンプト内に理由を明記）。
学習時は「画像＋そのモデル用キャプション」を組にして各ツールの
データセットフォルダへ配置する。
