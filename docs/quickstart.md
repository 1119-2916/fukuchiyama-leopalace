# 手順書: クローンから学習まで（AIエージェント実行用）

この文書は、実験環境の AI エージェントが上から順に実行するための
手順書である。人間向けの背景説明は各リンク先に譲り、ここには
実行コマンド・検証ゲート・停止条件だけを書く。

## 実行ルール（最初に必ず読むこと）

1. 手順は番号順に実行する。飛ばしてよい手順は明示したものだけ。
2. 全工程で共通の `RUN_ID` を使う（手順0で決める）。これにより
   全ての生成物のパスが一意に決まる。`<日時>` のようなプレース
   ホルダを推測で埋めてはいけない。
3. 各手順の「検証ゲート」を必ず実行し、失敗したら先に進まず、
   「停止条件」に従う。
4. 1つのモデルの失敗は、そのモデルだけを止める。他モデルは続行し、
   最後にまとめて報告する（共通手順 0〜2 の失敗は全体を止める）。
5. 推測で値を決めてはいけない対象: `CAPTION_MODEL`（LLM の alias）、
   ベースモデルの repo id / パス、ライセンス同意。これらが不明なら
   停止して報告する。

## 手順0: クローン・画像配置・共通 ID の決定

```bash
git clone https://github.com/1119-2916/fukuchiyama-leopalace.git
cd fukuchiyama-leopalace
# 運営者が dataset/ に手描きイラストを配置済みであること
pip install -r requirements.txt
export RUN_ID=$(date +%Y%m%d-%H%M%S)
echo "$RUN_ID" > .run_id
IMG_COUNT=$(find dataset -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.gif' \) | wc -l)
echo "images: $IMG_COUNT"
```

検証ゲート: `IMG_COUNT` が 1 以上（想定は 30〜50。範囲外なら
警告として記録し、続行する）。

## 手順1: キャプション生成

前提（運営者の担当。欠けていたら全体を停止して報告する）:

- 同じマシンで vLLM（Qwen 3.8 27B FP8）+ LiteLLM
  （http://localhost:4000）が稼働している
- `.env` があり、`CAPTION_MODEL` に LiteLLM のモデル alias が
  記入されている（空なら `.env.example` から作るが、alias の値は
  推測せず停止して報告する）

```bash
[ -f .env ] || cp .env.example .env
python3 scripts/preflight.py --with-captioning --model anima
# ↑ の captioning セクションが全て OK でなければ停止して報告
for M in anima moss z-image flux-klein; do
  ./scripts/caption_batch.sh --target-model "$M" --run-id "$RUN_ID"
done
```

検証ゲート（モデルごとに実行）:

```bash
for M in anima moss z-image flux-klein; do
  C=$(ls dataset-captions/$RUN_ID/$M/*.txt | wc -l)
  echo "$M captions: $C / $IMG_COUNT"
  test "$C" -eq "$IMG_COUNT" || echo "STOP: $M caption count mismatch"
  test -f dataset-captions/$RUN_ID/$M/caption_run.json || echo "STOP: $M manifest missing"
done
```

停止条件: いずれかのモデルでキャプション数が画像数と一致しない、
または caption_batch の summary に `failed` が 1 以上ある場合、
そのモデルを停止し、失敗した画像名を報告する。

## 手順2: 画像の前処理（正規化 → モデル別前処理）

元画像と手順1の生成キャプションを使い、学習用画像をモデル別に作る。

```bash
./scripts/normalize_images.sh --run-id "$RUN_ID"
# -> dataset-normalized/$RUN_ID/generic/
for M in anima moss z-image flux-klein; do
  ./scripts/preprocess_images.sh dataset-normalized/$RUN_ID/generic \
      --target-model "$M" --run-id "$RUN_ID"
done
# -> dataset-preprocessed/$RUN_ID/<モデル>/
```

検証ゲート: 各モデルの `dataset-preprocessed/$RUN_ID/<モデル>/` 内の
`.png` 数が `IMG_COUNT` と一致すること。一致しないモデルは停止。

## 手順3: 学習用フォルダの組み立て

```bash
for M in anima moss z-image flux-klein; do
  ./scripts/assemble_training.sh dataset-captions/$RUN_ID/$M \
      --images dataset-preprocessed/$RUN_ID/$M --overwrite
done
# -> training-data/<モデル>/（画像+キャプションのペア）
```

検証ゲート: 各モデルの `training-data/<モデル>/training_manifest.json`
の `pairs` が `IMG_COUNT` と一致すること。`captions_without_image` /
`images_without_caption` が空でないモデルは停止して内訳を報告する。

## 手順4: 学習ツールの導入（モデル別）

| モデル | エージェントが行ってよいこと | 運営者でなければできないこと |
|---|---|---|
| Anima | kohya-ss/sd-scripts を隣に clone し、その README の手順で環境構築 | Anima Base チェックポイントの入手先決定。`ANIMA_BASE_PATH` が未設定なら停止して報告 |
| Z-Image | ostris/ai-toolkit を隣に clone し環境構築 | Base（未蒸留）の正確な repo id の確定。曖昧なら停止して報告 |
| FLUX.3 Klein | ai-toolkit の環境構築（Z-Image と共通） | **ライセンス同意は法的な行為なので運営者のみ**。`HF_TOKEN` が無ければこのモデルだけ停止して報告 |
| Moss | `pip install -r requirements-train.txt` | なし。ただし手順5の事前確認が必須 |

## 手順5: 学習の実行

共通: 必ず先に `python3 scripts/preflight.py --model <モデル>` を
実行し、そのモデルのセクションが READY（WARN は可）であること。
`--dry-run` でコマンドを確認し、`outputs/<モデル>/train_command.txt`
に保存してから本実行する。学習は長時間かかるため、ログを
`outputs/<モデル>/train.log` に残し、完了は `.safetensors` の
生成で判定する。

```bash
# Anima
SD_SCRIPTS_DIR=../sd-scripts ANIMA_BASE_PATH=<運営者が指定> \
  ./scripts/train/train_anima.sh --dry-run
SD_SCRIPTS_DIR=../sd-scripts ANIMA_BASE_PATH=<運営者が指定> \
  ./scripts/train/train_anima.sh > outputs/anima/train.log 2>&1

# Z-Image / FLUX.3 Klein（FLUX.3 は HF_TOKEN 前提）
AI_TOOLKIT_DIR=../ai-toolkit ./scripts/train/train_z_image.sh --dry-run
AI_TOOLKIT_DIR=../ai-toolkit ./scripts/train/train_z_image.sh > outputs/z-image/train.log 2>&1
AI_TOOLKIT_DIR=../ai-toolkit ./scripts/train/train_flux_klein.sh --dry-run
AI_TOOLKIT_DIR=../ai-toolkit ./scripts/train/train_flux_klein.sh > outputs/flux-klein/train.log 2>&1
```

Moss のみ追加の事前確認: `scripts/train/train_moss_lora.py` 冒頭の
「Verify these points」を実機で確認してから実行する。1つでも
合わなければ Moss だけを停止し、実機で観測した事実
（属性名・形状など）を報告する。

```bash
./scripts/train/train_moss.sh --dry-run
./scripts/train/train_moss.sh > outputs/moss/train.log 2>&1
```

## 手順5.5: 評価生成（学習したモデルのみ）

学習が完了したモデルは、固定評価セットでサンプル生成を行う
（詳細: [generation.md](generation.md)）。これが手順6の学習記録の
「サンプル所感」の根拠になる。

```bash
./scripts/generate.sh --target-model <モデル> --eval \
    --lora outputs/<モデル>/<完成した重み>.safetensors \
    --run-id "$RUN_ID"
# -> generated/$RUN_ID/<モデル>/ に eval プロンプト6件分の画像
```

検証ゲート: `generated/$RUN_ID/<モデル>/generation_manifest.json`
が存在し、画像が6枚あること。生成に失敗したモデルは、エラー内容を
記録して手順6の報告に含める（学習自体の成否とは分けて扱う）。

## 手順6: 学習記録と最終報告

モデルごとに `outputs/<モデル>/TRAINING_RECORD.md` を作り、
[training.md](training.md) の記録形式の項目
（ベースモデルとハッシュ、使用したキャプションの元ラン
＝ `dataset-captions/$RUN_ID/<モデル>`、前処理ラン、パラメータ、
ステップ数、生成された重みファイル名、サンプル所感）を埋める。

最終報告の形式:

- モデルごとの成否（成功 / 停止した手順と理由）
- 使った `RUN_ID` と各生成物のパス
- 検証ゲートの実測値（画像数・キャプション数・ペア数）
- 重みの出力先（配布は GitHub Releases、表記は
  [../LICENSE-LORA.md](../LICENSE-LORA.md) に従う。リリース作業
  自体は運営者の指示があるまで行わない）
