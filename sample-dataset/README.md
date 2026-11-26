# サンプルデータセット（AI生成のテストフィクスチャ）

ファインチューニングのパイプライン検証用に、AIで生成した
イラスト30枚を置いたテストデータ。**本番の手描きデータセット
（`dataset/`）とは別物**であり、本番データの代わりではない。

## なぜこの形か

LoRA が学習するのは「全画像に共通する要素」なので、テストデータも
本番と同じ構造にしてある：**画風は全枚で統一し、題材だけを変える**。
画風は特徴が強いもの（下記スペック）を選んだので、学習前後の生成を
比べれば「画風が再現できたか」を目で判定できる。

## 画風スペック（全枚の生成プロンプトに逐語で含めた文）

> Children's picture-book illustration in gouache: thick dark-umber
> outlines, flat color shapes with visible dry-brush gouache texture,
> limited palette of cream, mustard yellow, brick red, teal and
> charcoal, subtle paper grain, soft rounded shapes, matte finish,
> no gradients, no photorealism, no text, no watermark.

特定の既存作家・素材サイトの画風の模倣ではなく、このスペックで
定義したオリジナルのタッチ。

## 中身

- `sample_01.png` 〜 `sample_30.png`（全て 1920×1280、PNG）
- 題材: 人物（傘の少女、パン屋、宇宙飛行士の少年、魔女 他）、
  動物（猫、狐、クジラ、ペンギン、熊、象、パンダ 他）、
  風景・乗り物（蒸気機関車、灯台、気球、フェリー 他）を分散
- `contact_sheet.jpg`: 全30枚の一覧（画風の統一感の確認用）

## 使い方

手順書（docs/quickstart.md）の検証リハーサルでは、このフォルダの
画像を `dataset/` にコピーしてから手順0〜3を実行する。
本番の手描き画像が揃ったら、そちらに差し替える。

## 既知の限界

- 全枚が同じアスペクト比（3:2）。本番の手描きデータは比率が
  バラバラなはずなので、バケットの多様性の検証は本番データで
  改めて必要
- AI生成画像であり、手描き特有の線の癖やムラは再現していない。
  キャプション品質や学習の最終評価は本番データで行うこと

## 権利

このリポジトリのプロジェクトのために AI 生成した画像であり、
第三者の素材は含まない。コードと同じリポジトリで公開する
テストフィクスチャとして扱う。
