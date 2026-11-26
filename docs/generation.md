# 画像生成（generate.py）

学習した LoRA を使って画像を生成する。基盤は Diffusers で統一し、
モデル差分は `scripts/generate.py` 内のプリセットとアダプタに
閉じ込めてある。実行は学習マシン（GPU + 重みがある環境）。
依存は学習と共通（`requirements-train.txt`）。

## 使い方

```bash
# 通常生成（1プロンプト）
./scripts/generate.sh --target-model flux-klein \
    --lora outputs/flux-klein/okakikai-flux-klein.safetensors \
    --prompt "okakikai. A girl with long silver hair in a white dress, standing in a garden"

# 評価生成（固定プロンプト×固定シードでチェックポイント比較用）
./scripts/generate.sh --target-model anima --eval \
    --lora outputs/anima/okakikai-anima-step500.safetensors

# FLUX.3: 高速生成は Distilled 側に LoRA を載せる
./scripts/generate.sh --target-model flux-klein --distilled \
    --lora outputs/flux-klein/okakikai-flux-klein.safetensors --prompt "..."
```

- 出力: `generated/<実行日時>/<モデル>/` に画像と
  `generation_manifest.json`（ベースモデル、使用 LoRA のパスと
  SHA-256、プロンプト、シード、steps、CFG、サイズ）
- `--dry-run` で生成計画（解決済みパラメータ）だけを表示できる
- サイズは全モデル共通の 16px グリッドに自動で寄せられる
- `--lora` を省略するとベースモデル単体の生成（比較用）
- `--lora-scale` で LoRA の強さ、`--num` でシードをずらした複数枚

## 評価生成の約束

`prompts/eval_prompts.txt` が固定評価セット（Anima は [tags] 節、
他は [natural] 節、各6件、シードは 1000+連番で固定）。
チェックポイントやキャプションのバリエーションを比較するときは、
必ず同じ eval セットで生成して比べること。このファイルを編集すると
過去の評価と比較できなくなるので、変更は慎重に。

## モデル別の注意（実機確認ポイントを含む）

| モデル | 既定 | 注意 |
|---|---|---|
| Anima | 30 steps / CFG 4.5、タグ形式プロンプト | ベースは `--base-model` か `ANIMA_BASE_PATH` で指定。DiffusionPipeline 経由で読めるか実機確認 |
| Moss | 30 steps / CFG 4.0 | 公式想定では生成確認は SFT 側: `--sft` で SFT チェックポイントに切り替え可能（repo id は実機確認）。学習スクリプトが保存する PEFT 形式 LoRA が `load_lora_weights` で読めるかも実機確認（読めなければ safetensors への変換が要る） |
| Z-Image | 28 steps / CFG 4.0 | 生成も Base（未蒸留）で行う。Turbo に載せない |
| FLUX.3 Klein | Base: 50 steps / CFG 3.5、Distilled: 16 steps / CFG 1 | Distilled の repo id は実機確認。ゲート付きのため `HF_TOKEN` が必要 |

## Discord への投稿（持ち出し用）

生成画像を Discord の特定チャンネルに投稿する補助スクリプトがある。
実験環境から画像を持ち出す手間を省くためのもので、生成そのものとは
独立している。

```bash
./scripts/post_generated.sh --dir generated/<実行日時>/<モデル> \
    --webhook "$DISCORD_WEBHOOK_URL"
# 先に確認: --dry-run で送信内容（概要文とバッチ構成）だけ表示
```

- 引数: `--dir`（画像のあるディレクトリ）と `--webhook`
  （または環境変数 `DISCORD_WEBHOOK_URL`、`.env` の同名キー）
- ディレクトリに `generation_manifest.json` があれば、モデル・
  ベース・LoRA・steps/CFG・枚数の概要文を先に投稿し、続けて
  画像を10枚ずつ（Discord の添付上限）まとめて投稿する
- 25MB 超のファイルは警告してスキップ。429 は `retry_after` に
  従って再試行する
- **Webhook URL は認証情報**。引数・環境変数・`.env` のいずれでも
  渡せるが、コミットしないこと。スクリプトは URL を表示・記録
  しない

## 手順書との関係

[quickstart.md](quickstart.md) 手順5.5 として、学習直後に
評価生成を行う工程が組み込まれている。生成サンプルは学習記録
（`outputs/<モデル>/TRAINING_RECORD.md` のサンプル所感）の
根拠になるので、学習したら必ず eval 生成まで行うこと。
