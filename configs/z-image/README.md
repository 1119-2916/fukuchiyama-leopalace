# configs/z-image/
Z-Image Base (未蒸留) 向け LoRA 学習設定。Turbo 用と混用しない。
キャプション形式: detailed (自然文)。詳細: ../../docs/base-models.md
画像前処理: preprocess_images.py プリセット z-image（面積 1024^2、グリッド 16）。詳細: ../../docs/preprocessing.md
学習設定: ai_toolkit.yaml（AI Toolkit, arch: zimage）。起動: ../../scripts/train/train_z_image.sh
画像生成: scripts/generate.py（生成も Base 側）。詳細: ../../docs/generation.md
