# 学習について

## 前提ハードウェア

- GPU: NVIDIA RTX 6000 Blackwell Max-Q
- ドライバ / CUDA / 学習ツールのバージョンは、実際に学習する環境で
  確定した時点でここに記録する
- キャプション生成も同じマシン上のローカル LLM で行う:
  vLLM 上の Qwen 3.8 27B FP8 を LiteLLM プロキシ
  （`http://localhost:4000`、OpenAI 互換 API）経由で使う。
  詳細は [captioning.md](captioning.md)

## 対象

- 様々な画像生成 AI / ベースモデル向けに LoRA を作成する
- 対象モデル（2026-11-26 選定）: Anima / Moss（MOSS-Image）/ Z-Image / Flux-Klein
- 各モデルの学習用チェックポイント・ツール・ライセンス・
  キャプション形式の違いは [base-models.md](base-models.md) に整理した。
  特に「学習は Base（未蒸留）側で行う」「キャプション形式はモデルごとに
  変える（Anima はタグ形式、それ以外は自然文 detailed）」が要点

## 学習用データセットの置き場所

学習ツールに渡すフォルダは、原画像の `dataset/` や生成キャプションの
保管場所 `dataset-captions/` とは分けて、`training-data/<モデル名>/`
に組み立てる（画像とキャプションが同名ペアで並ぶ形式）:

```bash
./scripts/assemble_training.sh dataset-captions/<実行日時>/<モデル名>
```

詳細と来歴マニフェスト（`training_manifest.json`）については
[captioning.md](captioning.md) の「キャプションのバリエーションと
学習用データの分離」を参照。学習条件の違いを比べるときは、
キャプションのバリエーション（元ラン）を変えたことも含めて
下記の学習記録に残すこと。

## 学習記録の形式

学習を 1 回行うごとに、再現できるよう最低限以下を記録する。

| 項目 | 内容 |
|---|---|
| 日付 | 学習日 |
| ベースモデル | モデル名とバージョン / ハッシュ |
| 学習ツール | ツール名とバージョン（例: kohya-ss/sd-scripts、ai-toolkit 等） |
| データセット | 枚数、解像度、繰り返し数（repeats） |
| 主なパラメータ | rank / alpha、学習率、バッチサイズ、エポック / ステップ数 |
| 設定ファイル | `configs/` 内の対応ファイルへのパス |
| 結果 | サンプル生成の所感、過学習 / 学習不足の有無 |
| 重み | 公開した Release へのリンク |

## メモ

- Blackwell 世代の GPU では、学習ツール側の対応バージョン
  （PyTorch / CUDA の組み合わせ）が結果を左右することがあるため、
  動いた組み合わせは必ず記録に残す
