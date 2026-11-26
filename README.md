# fukuchiyama-leopalace

手描きイラスト 30〜50 枚を訓練データセットとして、様々な画像生成 AI 向けの
LoRA ファインチューニング重みを作成するプロジェクト。

## 目的

- 手描きイラスト 30〜50 枚から、その画風・特徴を学習した LoRA 重みを作る
- 1 つのベースモデルに限定せず、複数の画像生成 AI / ベースモデル向けに
  それぞれ LoRA を学習・比較する
- 学習条件（データ、キャプション、パラメータ、環境）を記録し、再現できるようにする

## 前提環境

- GPU: NVIDIA RTX 6000 Blackwell Max-Q
- 学習は上記 GPU を搭載したマシン上で行う前提（このリポジトリ自体は
  コード・設定・ドキュメントの管理用）

## リポジトリ構成

```text
.
├── README.md          # このファイル
├── docs/
│   ├── dataset.md     # データセットの仕様・キャプション方針
│   ├── captioning.md  # Vision LLM によるキャプション生成フロー
│   ├── base-models.md # 対象ベースモデル 4 種の注意点と比較
│   └── training.md    # 学習環境・手順・記録の取り方
├── scripts/
│   ├── caption_batch.py  # キャプション一括生成（Python 標準ライブラリのみ）
│   └── caption_batch.sh  # そのラッパー
├── prompts/
│   └── caption_default.txt  # キャプション生成プロンプト
├── tests/
│   └── test_caption_batch.py  # mock プロバイダによる単体テスト
├── dataset/
│   └── README.md      # 画像とキャプションの置き方
└── configs/
    ├── README.md      # 学習設定ファイルの置き場（モデルごとに分ける）
    ├── anima/  ├── moss/  ├── z-image/  └── flux-klein/
```

## ワークフロー

1. データセットを `dataset/` に配置し、画像と同名の `.txt` でキャプションを付ける。
   キャプションは Vision LLM で一括生成する
   （`./scripts/caption_batch.sh`、詳細: [docs/captioning.md](docs/captioning.md)）
2. 対象ベースモデルごとに `configs/` へ学習設定を用意する
3. RTX 6000 Blackwell Max-Q 上で LoRA 学習を実行する
4. 学習条件と結果（サンプル生成、所感）を `docs/training.md` の記録形式で残す
5. 完成した LoRA 重みは Git ではなく **GitHub Releases** に添付して配布する
   （理由は下記）

## 重みファイルの扱い

- `.safetensors` 等の重みファイルは `.gitignore` で Git 管理から外している
  - GitHub は 100 MB を超えるファイルを Git に push できないため
  - LoRA 重みは Releases のアセット（1 ファイル 2 GB まで）として公開する想定
- Git LFS を使う方針に変える場合は、この README と `.gitignore` を合わせて更新する

## ライセンス

- LoRA 重み: CC BY-NC 4.0（商用利用は全面不可）→ [LICENSE-LORA.md](LICENSE-LORA.md)
- データセット画像: このリポジトリで公開する
- コード: 未定

## ステータス

- 2026-11-26: リポジトリ作成（公開）。初期構成のみ、データセット・学習は未着手

## 未決定事項（決まり次第ここを更新）

- [x] 対象ベースモデル: Anima / Moss（MOSS-Image）/ Z-Image / Flux-Klein
      （詳細: [docs/base-models.md](docs/base-models.md)）
- [x] Flux-Klein の世代: FLUX.3 Image Klein Base（5.6B、Kommunity License）
- [ ] トリガーワード（概念は [docs/dataset.md](docs/dataset.md) 参照）
- [x] データセット画像自体の公開: 公開する（2026-11-26 決定）
- [x] 学習済み重みのライセンス: CC BY-NC 4.0（商用利用は全面不可）
      → [LICENSE-LORA.md](LICENSE-LORA.md)
- [ ] コード（スクリプト類）のライセンス
