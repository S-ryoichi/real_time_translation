# Real-time Translation (日本語 → 英語)

日本語音声をリアルタイムで英語に翻訳し、ブラウザ上に表示するWebアプリケーションです。

## 主要機能

- 日本語音声 → 英語テキストのリアルタイム翻訳
- WebSocketによる双方向通信
- ブラウザ上での翻訳結果表示（大画面表示対応）
- 文字サイズ調整機能
- リアルタイム追記と自動スクロール

## 技術スタック

- **バックエンド**: FastAPI + OpenAI Whisper (small model)
- **フロントエンド**: HTML + JavaScript (Vanilla)
- **通信**: WebSocket
- **音声入力**: MediaRecorder API

---

## セットアップ手順（初学者向け）

### 前提条件

- Python 3.10以上がインストールされていること
- マイクが接続されていること
- インターネット接続（初回のみ、Whisperモデルのダウンロードに必要）

### 1. リポジトリのクローンまたはダウンロード

すでにこのフォルダにいる場合は、この手順をスキップしてください。

```bash
# Gitでクローンする場合
git clone <リポジトリURL>
cd real_time_translation

# またはZIPでダウンロードして解凍
```

### 2. Python仮想環境の作成（推奨）

仮想環境を使うことで、プロジェクト専用のパッケージ環境を作成できます。

#### Windowsの場合

```bash
# 仮想環境を作成
python -m venv venv

# 仮想環境を有効化
venv\Scripts\activate
```

#### macOS / Linuxの場合

```bash
# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate
```

仮想環境が有効化されると、ターミナルのプロンプトの先頭に `(venv)` が表示されます。

### 3. 環境変数の設定（オプション）

Whisperモデルをカスタマイズしたい場合は、`.env`ファイルを作成します。

```bash
# .env.exampleをコピーして.envファイルを作成
cp .env.example .env
```

`.env`ファイルを編集して、使用するモデルやログレベルを指定します:

```bash
# Whisperモデルの設定（例: smallモデルを使用する場合）
WHISPER_MODEL=small

# ログレベルの設定（例: DEBUGモードで詳細ログを表示）
LOG_LEVEL=INFO
```

**利用可能なモデル:**
- `tiny`: 最も高速だが精度は低め
- `base`: 高速で軽量
- `small`: バランス型（デフォルト、推奨）
- `medium`: 高精度だが重い
- `large`: 最高精度だが非常に重い
- `large-v2`, `large-v3`: largeの改良版

**利用可能なログレベル:**
- `DEBUG`: 詳細なデバッグ情報（音声処理の詳細ログ）
- `INFO`: 一般的な情報（デフォルト、推奨）
- `WARNING`: 警告メッセージのみ
- `ERROR`: エラーメッセージのみ
- `CRITICAL`: 致命的なエラーのみ

**注意**:
- `.env`ファイルがない場合や、無効な値が設定されている場合は、自動的にデフォルト値（`WHISPER_MODEL=small`, `LOG_LEVEL=INFO`）が使用されます。
- `LOG_LEVEL=DEBUG`は大量のログが出力されるため、問題のデバッグ時のみ使用してください。

### 4. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

**注意**: 初回インストール時は数分かかる場合があります。特にWhisperモデルのダウンロードに時間がかかります。

#### ffmpegのインストール（Linuxの場合）

Whisperは音声のデコードにffmpegを使用します。Linuxの場合は以下のコマンドでインストールしてください。

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

#### ffmpegのインストール（Windowsの場合）

1. [ffmpeg公式サイト](https://ffmpeg.org/download.html)からダウンロード
2. ダウンロードしたZIPを解凍
3. `bin`フォルダ内の`ffmpeg.exe`をシステムPATHに追加

#### ffmpegのインストール（macOSの場合）

Homebrewを使用している場合:

```bash
brew install ffmpeg
```

### 5. インストールの確認

以下のコマンドで、必要なパッケージがインストールされているか確認できます。

```bash
pip list | grep -E "fastapi|uvicorn|whisper|torch"
```

または

```bash
pip show fastapi uvicorn openai-whisper python-dotenv
```

---

## 実行方法

### 1. サーバーの起動

ターミナルで以下のコマンドを実行します。

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**各オプションの意味:**
- `main:app`: `main.py`内の`app`オブジェクトを起動
- `--host 0.0.0.0`: すべてのネットワークインターフェースでリッスン
- `--port 8000`: ポート8000で起動
- `--reload`: コード変更時に自動でリロード（開発時に便利）

**起動時のログ例:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

初回起動時は、Whisperモデル（デフォルトは`small`）がダウンロードされるため、数分かかる場合があります。

**起動時のログで使用モデルを確認:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
[INFO] Loading Whisper model: small
[INFO] Whisper model 'small' loaded successfully on cpu
INFO:     Application startup complete.
```

ログレベルを`DEBUG`に設定すると、音声処理の詳細ログが表示されます。

### 2. ブラウザでアクセス

サーバーが起動したら、ブラウザで以下のURLにアクセスします。

```
http://localhost:8000
```

### 3. 使い方

1. **マイクの許可**: ブラウザがマイクへのアクセス許可を求めてきたら「許可」をクリック
2. **録音開始**: 「Start Recording」ボタンをクリック
3. **日本語を話す**: マイクに向かって日本語を話す
4. **翻訳結果の表示**: 約10秒ごとに翻訳結果が画面に表示されます
5. **録音停止**: 「Stop Recording」ボタンをクリック
6. **クリア**: 「Clear」ボタンで翻訳結果をクリア
7. **文字サイズ調整**: 「Font Size」ドロップダウンでサイズを変更

---

## エンドポイント一覧

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/` | GET | index.htmlを提供（ブラウザUI） |
| `/translate` | POST | 音声ファイルを翻訳（単発、API用） |
| `/ws` | WebSocket | リアルタイム翻訳（ブラウザUI用） |
| `/health` | GET | ヘルスチェック |

### API使用例（curlでテスト）

音声ファイルを直接送信して翻訳結果を取得:

```bash
curl -X POST -F "file=@sample.wav" http://localhost:8000/translate
```

レスポンス例:

```json
{"translation": "Hello, this is a translation test."}
```

---

## トラブルシューティング

### Q1. サーバーが起動しない

**エラー**: `ModuleNotFoundError: No module named 'fastapi'`

**解決方法**: 仮想環境が有効化されているか確認し、依存パッケージを再インストール

```bash
pip install -r requirements.txt
```

### Q2. ポート8000が使用中

**エラー**: `Error: [Errno 48] Address already in use`

**解決方法**: 別のポートを使用

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

その場合、ブラウザで `http://localhost:8001` にアクセスしてください。

### Q3. マイクへのアクセスが拒否される

**エラー**: ブラウザで「マイクへのアクセスが拒否されました」

**解決方法**:
1. ブラウザの設定でマイクの許可を確認
2. HTTPSではなくHTTPでアクセスしている場合、`localhost`でのみ動作します
3. Chrome/Edgeの場合: `chrome://settings/content/microphone` で許可を確認

### Q4. WebSocketが接続できない

**エラー**: `WebSocket error` または `WebSocket closed`

**解決方法**:
1. サーバーが起動しているか確認
2. ファイアウォールがポート8000をブロックしていないか確認
3. ブラウザのコンソール（F12）でエラーログを確認

### Q5. 翻訳結果が表示されない

**考えられる原因**:
1. マイクに音声が入力されていない → マイクの音量を確認
2. 音声が短すぎる → 10秒以上話してみる
3. 日本語以外の言語を話している → 日本語で話してみる

### Q6. ffmpegが見つからない

**エラー**: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`

**解決方法**: 上記「3. 依存パッケージのインストール」のffmpegセクションを参照

---

## 開発メモ

### ファイル構成

```
real_time_translation/
├── main.py              # FastAPIバックエンド
├── index.html           # フロントエンドUI
├── requirements.txt     # Python依存パッケージ
├── .env.example         # 環境変数のサンプル
├── .env                 # 環境変数設定（ユーザーが作成）
├── README.md            # このファイル
└── 要件定義.md          # プロジェクト仕様書
```

### 設定のカスタマイズ

#### 音声チャンク送信間隔の変更

[index.html:176](index.html#L176)の`SEGMENT_MS`を変更:

```javascript
const SEGMENT_MS = 10000; // 10秒 → 好きな値に変更（ミリ秒）
```

短くすると翻訳の応答が早くなりますが、精度が下がる可能性があります。

#### Whisperモデルの変更

`.env`ファイルで`WHISPER_MODEL`を設定:

```bash
# .envファイルを編集
WHISPER_MODEL=small  # tiny, base, small, medium, large, large-v2, large-v3 から選択
```

- `tiny`, `base`: 高速だが精度低め
- `small`: バランス型（デフォルト）
- `medium`, `large`: 高精度だが処理が重い

設定後、サーバーを再起動して反映させてください。

#### ログレベルの変更

`.env`ファイルで`LOG_LEVEL`を設定:

```bash
# .envファイルを編集
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL から選択
```

- `DEBUG`: 音声処理の詳細ログを表示（デバッグ時に有用）
- `INFO`: 一般的な情報のみ表示（推奨）
- `WARNING`以上: エラー関連のみ表示

設定後、サーバーを再起動して反映させてください。

---

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

## サポート

問題が発生した場合は、以下を確認してください:

1. Python 3.10以上がインストールされているか
2. 仮想環境が有効化されているか
3. 依存パッケージがすべてインストールされているか
4. ffmpegがインストールされているか（Linuxの場合）
5. マイクが正しく接続されているか

それでも解決しない場合は、GitHubのIssueを作成してください。
