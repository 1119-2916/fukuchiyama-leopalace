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

## モデル別ランブック（学習の実行方法）

クローン直後からの全行程は [quickstart.md](quickstart.md) を参照。
前提の充足は `python3 scripts/preflight.py` で機械的に確認できる。

学習コードと設定はリポジトリに用意済み。実行は RTX 6000 の
学習マシン側で行う（このリポジトリの開発環境には GPU がない）。
全モデル共通の前提: `training-data/<モデル名>/` を
`assemble_training.sh` で組み立てておくこと。出力は `outputs/<モデル名>/`。

| モデル | ツール | 設定 | 起動 |
|---|---|---|---|
| Anima | kohya sd-scripts `anima_train_network.py` | `configs/anima/dataset.toml` | `SD_SCRIPTS_DIR=... ANIMA_BASE_PATH=... ./scripts/train/train_anima.sh` |
| Moss | 自前スクリプト（Diffusers + PEFT、flow matching） | `scripts/train/train_moss_lora.py` 内の既定値 | `./scripts/train/train_moss.sh` |
| Z-Image | AI Toolkit（arch: zimage） | `configs/z-image/ai_toolkit.yaml` | `AI_TOOLKIT_DIR=... ./scripts/train/train_z_image.sh` |
| FLUX.3 Klein | AI Toolkit（arch: flux3） | `configs/flux-klein/ai_toolkit.yaml` | `AI_TOOLKIT_DIR=... ./scripts/train/train_flux_klein.sh`（要 HF_TOKEN） |

- 全起動スクリプトに `--dry-run` があり、実行コマンドだけを表示できる
- ハイパーパラメータは 30〜50 枚向けの初期値（rank 32、lr 1e-4 系）。
  FLUX.3 は BFL 公式例（4000 steps）に合わせた
- **MOSS のみ未検証の自前実装**: 既製の LoRA トレーナーが無いため
  Diffusers + PEFT で実装した。実機で最初に確認すべき点
  （パイプラインの属性名、潜在表現のパッキング方法、
  encode_prompt の有無、VAE のスケーリング、タイムステップ規約）は
  スクリプト冒頭に列挙してある。ここが通るまでは他3モデルの結果を
  先に確定させるのが安全
- Anima の kohya フラグ名も、実機の sd-scripts のバージョンで
  一度照合すること（`--dry-run` の出力と `--help` を突き合わせる）
- 設定ファイルの整合性（arch、トリガーワード、no-upscale、
  Base チェックポイント等）は `tests/test_train_configs.py` が
  機械的に守っている

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
