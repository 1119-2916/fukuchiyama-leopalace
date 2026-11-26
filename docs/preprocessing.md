# モデル別 前処理（preprocess_images.py）

画像パイプラインの全体像:

```
dataset/（原画・正本）
  └─ normalize_images.py   … モデル共通の衛生処理（EXIF/RGB/透明/上限）
       dataset-normalized/<実行日時>/<モデル>/
  └─ preprocess_images.py  … モデル別の幾何処理（本書）
       dataset-preprocessed/<実行日時>/<モデル>/
  └─ assemble_training.py  … 採用キャプションと組み合わせて学習用へ
       training-data/<モデル>/
```

## なぜモデル別の前処理を別に用意するか

調査（[normalization.md](normalization.md)）の結論は「バケッティングは
学習ツール内蔵なので、外部の幾何処理は必須ではない」だった。
それでもモデル別の前処理コードを用意する理由は2つ:

1. **バケッティングを持たない経路がある**: 素の Diffusers 系ローダや
   MOSS-Image のスタックは、16 の倍数でない寸法でエラーになるか、
   固定リサイズ前提になる。モデルに合った寸法へ先に揃えておけば、
   どの経路でもそのまま使える
2. **再現性**: ツール内部の処理は設定とバージョンで変わり得る。
   外部で確定させれば「どの寸法・どの切り抜きで学習したか」が
   `preprocess_manifest.json` に残り、学習結果の比較が正確になる

kohya / AI Toolkit で学習する場合、こちらで前処理した画像を渡しても
二重処理にはならない（既にバケット寸法なので実質そのまま通る）。

## モデル別プリセット

| モデル | 目標面積 | グリッド（step） | 辺の範囲 | 根拠 |
|---|---|---|---|---|
| anima | 1024² | 32 | 256〜2048 | AI Toolkit が 32 倍数にスナップ。kohya 例は 16〜64 と幅があるため中間の 32 を採用 |
| moss | 1024² | 16 | 256〜2048 | 公式範囲 0.25〜4MP、16 の倍数必須 |
| z-image | 1024² | 16 | 256〜2048 | 公式の総面積範囲と AI Toolkit の 16 倍数スナップ |
| flux-klein | 1024² | 16 | 256〜2048 | BFL 公式学習例は約 1MP |

## 処理内容

- 画像ごとに、元のアスペクト比を保ったまま目標面積に最も近い
  バケット（両辺が step の倍数）を決める
- バケットを覆うようリサイズ（resize-to-cover）し、中央で切り抜く
- **アップスケールはしない**: 目標面積未満の小さい画像は、自分の
  サイズを step グリッドに切り下げた寸法がバケットになる
- 画像ごとに元サイズ・バケット・拡大縮小率・切り抜き範囲を
  `preprocess_manifest.json` に記録する
- 出力先は他工程と同型: `dataset-preprocessed/<実行日時>/<モデル>/`
  （再生成可能なため gitignore 対象）

## 使い方

```bash
pip install Pillow
# 正規化済み画像を入力にするのが標準の流れ
./scripts/preprocess_images.sh dataset-normalized/<日時>/moss --target-model moss

# 学習用フォルダの組み立てでは前処理済みを画像ソースに使う
./scripts/assemble_training.sh dataset-captions/<日時>/moss \
    --images dataset-preprocessed/<日時>/moss
```

`--target-area` / `--step` でプリセットの上書きも可能（実験用。
使った場合はマニフェストに上書き後の値が残る）。
