# キャプション生成フロー

画像が読める LLM（Vision LLM）にデータセット画像を見せて、LoRA 学習用の
キャプションを一括生成する。

## 仕組み

```text
dataset/*.png 等 ──→ scripts/caption_batch.py ──→ dataset/*.txt（画像と同名）
                        │  画像 + プロンプトを Vision LLM の API へ送信
                        │  プロンプト: prompts/caption_default.txt
                        └─ トリガーワードは生成後にスクリプトが機械的に付与
```

- Python 標準ライブラリのみで動作する（pip インストール不要、Python 3.10+）
- 対応プロバイダ:
  - `openai-compatible`（既定）: OpenAI Chat Completions 互換 API。
    OpenAI 本体のほか、ローカルの vLLM / Ollama / LM Studio にも
    `--base-url` を向ければ使える（RTX 6000 上でローカル VLM を
    動かす場合はこちら。画像を外部に送らない構成にできる）
  - `anthropic`: Anthropic Messages API
  - `mock`: API を呼ばない動作確認用（プレースホルダを書く）
- 既に `.txt` がある画像はスキップするので、中断しても再実行で再開できる

## 本番構成（決定）

- 実行マシン: NVIDIA RTX 6000 Blackwell Max-Q 搭載マシン
- モデル: Qwen 3.8 27B FP8 を vLLM 上で実行（画像が読める構成であること）
- API: LiteLLM プロキシが `http://localhost:4000` で
  OpenAI API Compatible な口を開ける
- このスクリプトは**そのマシン上で**実行し、`localhost:4000` に投げる
  （`localhost` は実行マシンの自分自身を指す点に注意）

実行例（`.env` に `CAPTION_MODEL` 等を設定済みの前提）:

```bash
./scripts/caption_batch.sh --dry-run
./scripts/caption_batch.sh --limit 5    # 少量で質を確認
./scripts/caption_batch.sh              # 本番
```

`.env` の雛形は `.env.example`（本番構成の値が入っている）。
`CAPTION_MODEL` には LiteLLM の config で付けたモデル alias を書くこと。

## 使い方

### 0. 準備

`.env.example` を `.env` にコピーして値を埋める（スクリプトが自動で読む。
シェルで export 済みの変数が優先される）。キーはコミットしない。

### 1. まず dry-run で対象を確認

```bash
./scripts/caption_batch.sh --dry-run
```

### 2. 少量で試す（推奨）

いきなり全件回さず、数枚でキャプションの質を確認する。

```bash
./scripts/caption_batch.sh --limit 5 --trigger-word <トリガーワード>
```

生成された `dataset/*.txt` を目視確認し、必要なら
`prompts/caption_default.txt` や `--style` を調整する。

### 3. 本番バッチ

```bash
./scripts/caption_batch.sh --trigger-word <トリガーワード> \
    --log-file caption_run.jsonl
```

## 主なオプション

| オプション | 既定 | 説明 |
|---|---|---|
| `input_dir`（位置引数） | `dataset` | 画像のあるフォルダ |
| `--provider` | `openai-compatible` | `anthropic` / `mock` も可。環境変数 `CAPTION_PROVIDER` でも指定可 |
| `--model` | プロバイダ依存 | モデル名。環境変数 `CAPTION_MODEL` でも指定可 |
| `--base-url` | プロバイダ依存 | ローカルサーバ等への接続先。環境変数 `CAPTION_BASE_URL` |
| `--style` | `tags` | `tags`（カンマ区切りタグ）/ `natural`（一文）/ `detailed`（2〜3文） |
| `--language` | `en` | キャプション言語 `en` / `ja` |
| `--trigger-word` | なし | 全キャプションの先頭に付ける。環境変数 `CAPTION_TRIGGER_WORD` |
| `--overwrite` | off | 既存キャプションも再生成する |
| `--limit N` | なし | 処理件数の上限（試行用） |
| `--workers N` | 1 | 並列リクエスト数。API のレート制限に注意 |
| `--delay SEC` | 0 | workers=1 のときのリクエスト間隔 |
| `--prompt-file` | `prompts/caption_default.txt` | プロンプトテンプレートの差し替え |
| `--log-file` | なし | 1 画像 1 行の JSON Lines で結果を追記 |

## プロンプトの設計方針

- キャプションに書くのは「画像ごとに異なる要素」（被写体、ポーズ、服装、
  小物、背景）だけ
- 画風・画材など全画像共通の要素は書かない ― 共通要素こそ LoRA に
  学習させたい対象で、全キャプションに書くと学習が弱まるため
- トリガーワードはプロンプトでは付けさせず、スクリプトが後付けする
  （付け忘れ・表記揺れを防ぐため）

## テスト

API を使わない単体テスト（mock プロバイダでバッチ処理だけを検証）:

```bash
python3 -m unittest discover -s tests -v
```

## 注意

- このリポジトリは公開リポジトリだが、API 呼び出し自体は公開とは無関係。
  ただしクラウド API を使うと画像はそのプロバイダに送信される。
  画像を外部に出したくない場合は、ローカルの Vision LLM
  （vLLM / Ollama 等、`--base-url` 指定）を使うこと
- 生成キャプションは下書きとして扱い、学習前に一度目視確認する前提
