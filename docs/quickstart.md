# クイックスタート: クローンから学習まで

学習マシン（RTX 6000）での最短手順。**クローンして画像を置いただけ
では学習は始まらない**。キャプション生成、学習ツールの導入、
ベースモデルの入手が別途必要で、この文書がその全行程を一本に
まとめたもの。

迷ったらまず前提チェック:

```bash
python3 scripts/preflight.py --with-captioning   # 何が足りないか一覧表示
```

## 0. クローンと画像配置

```bash
git clone https://github.com/1119-2916/fukuchiyama-leopalace.git
cd fukuchiyama-leopalace
# 手描きイラスト 30〜50 枚を dataset/ に置く（連番命名推奨: 001.png ...）
pip install -r requirements.txt   # Pillow 等
```

## 1. キャプション生成（ローカル LLM）

前提（リポジトリ外）: 同じマシンで vLLM（Qwen 3.8 27B FP8）+
LiteLLM（http://localhost:4000）が動いていること。

```bash
cp .env.example .env
# .env の CAPTION_MODEL に LiteLLM のモデル alias を記入
./scripts/caption_batch.sh --target-model anima
./scripts/caption_batch.sh --target-model moss
./scripts/caption_batch.sh --target-model z-image
./scripts/caption_batch.sh --target-model flux-klein
# -> dataset-captions/<日時>/<モデル>/ にモデル別キャプションが残る
```

生成後、数枚を目視確認すること（詳細: [captioning.md](captioning.md)）。

## 2. （任意）正規化とモデル別前処理

学習ツール内蔵のバケッティングで足りる場合は省略可。寸法を外部で
確定させたい場合は挟む（詳細: [normalization.md](normalization.md) /
[preprocessing.md](preprocessing.md)）:

```bash
./scripts/normalize_images.sh
./scripts/preprocess_images.sh dataset-normalized/<日時>/generic --target-model moss
```

## 3. 学習用フォルダの組み立て（モデルごとに必須）

```bash
./scripts/assemble_training.sh dataset-captions/<日時>/anima
./scripts/assemble_training.sh dataset-captions/<日時>/moss
./scripts/assemble_training.sh dataset-captions/<日時>/z-image
./scripts/assemble_training.sh dataset-captions/<日時>/flux-klein
# -> training-data/<モデル>/ に画像+キャプションのペアができる
# 前処理を挟んだ場合は --images dataset-preprocessed/<日時>/<モデル> を付ける
```

## 4. 学習ツールの導入（リポジトリ外・モデルごと）

| モデル | 導入物 | 追加で要るもの |
|---|---|---|
| Anima | kohya-ss/sd-scripts を隣に clone し、その手順で環境構築 | Anima Base チェックポイントの入手、`ANIMA_BASE_PATH` と `SD_SCRIPTS_DIR` の設定 |
| Z-Image | ostris/ai-toolkit を隣に clone し、その手順で環境構築 | Z-Image Base（未蒸留）の重み。repo id は実機で要確認 |
| FLUX.3 Klein | 同上 | **ゲート付き**: Hugging Face でライセンス同意 + `HF_TOKEN` 設定 |
| Moss | なし（自前スクリプト） | `pip install -r requirements-train.txt`（Blackwell 対応 torch を先に入れること）。スクリプト冒頭の確認5点を先に潰す |

## 5. 学習の実行

```bash
python3 scripts/preflight.py            # READY になるまで 1〜4 を埋める
./scripts/train/train_anima.sh          # Anima (kohya)
./scripts/train/train_moss.sh           # Moss (自前)
./scripts/train/train_z_image.sh        # Z-Image (AI Toolkit)
./scripts/train/train_flux_klein.sh     # FLUX.3 Klein (AI Toolkit)
# 出力: outputs/<モデル>/  /  全て --dry-run でコマンド確認可
```

## 6. 学習後

- 結果を [training.md](training.md) の記録形式で残す
  （ベースモデルのハッシュ、使用したキャプションの元ラン、
  パラメータ、サンプル所感）
- 重み（`.safetensors`）は Git に入れず GitHub Releases で配布する。
  表記は [../LICENSE-LORA.md](../LICENSE-LORA.md) の方針
  （ベース準拠 + 商用全面不可）に従う
