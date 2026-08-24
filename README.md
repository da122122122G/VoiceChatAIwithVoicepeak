# Gemini × Whisper.cpp × VOICEPEAK 音声チャット

マイクへ話しかけると、Whisperが文字起こしし、Geminiの返答をVOICEPEAKが読み上げるWindows向け音声チャットです。

キー操作は不要です。Whisper・Gemini・VOICEPEAKの処理中もマイク入力を続け、次の発話をキューへ保存します。

## 主な機能

- 常時リスニングとRMSベースの自動発話検出
- 起動直後の環境ノイズを基準にした動的な発話閾値
- 0.3秒のプリロールと末尾無音のトリミング
- Whisper Serverの自動起動とHTTP接続再利用
- Geminiとの会話履歴保存・起動時読み込み
- PCの現在日時とタイムゾーンをGeminiへ通知
- 効果音表記やWhisperの代表的な誤認識を除外
- VOICEPEAK Runtime Bridgeへのキュー送信
- 音声処理中も次の発話を取り込むバックグラウンド録音

## 必要なもの

- Windows 10 / 11（x64）
- Python 3.12以降
- Git for Windows
- CMake
- Visual Studio Build Tools 2022
  - 「C++によるデスクトップ開発」ワークロード
- .NET SDK
- .NET Framework 4.8 Developer Pack
- VOICEPEAK本体
- Gemini APIキー
- NVIDIA GPU版を使う場合のみCUDA Toolkit

VOICEPEAK、Gemini APIキー、OS向け開発ツールは利用者が用意してください。それ以外のPythonパッケージ、whisper.cpp、Whisperモデル、VoicepeakProxyCore、VOICEPEAK Bridgeは`setup.ps1`が準備します。

## 最短セットアップ

### 1. リポジトリをclone

```powershell
git clone https://github.com/da122122122G/VoiceChatAIwithVoicepeak.git
cd VoiceChatAIwithVoicepeak
```

配置場所は任意です。`C:\voice_ai`へ置く必要はありません。

### 2. Gemini APIキーを設定

```powershell
setx GEMINI_API_KEY "ここにAPIキー"
```

`setx`を実行した後は、新しいPowerShellまたはコマンドプロンプトを開いてください。

確認:

```powershell
echo $env:GEMINI_API_KEY
```

### 3. 自動セットアップを実行

VOICEPEAKとこの音声チャットを実行中の場合は、先に終了してください。BridgeのDLLがVOICEPEAKへ読み込まれている間は、Windowsのファイルロックにより再ビルドできません。

CPU版:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

NVIDIA GPU／CUDA版:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -WhisperBackend cuda
```

CUDAアーキテクチャを指定する例:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 `
  -WhisperBackend cuda `
  -CudaArchitectures "86"
```

`setup.ps1`は次を自動実行します。

1. `.venv`を作成
2. `requirements.txt`からPythonパッケージをインストール
3. 公式`ggml-org/whisper.cpp`の動作確認済みリビジョンをclone
4. `whisper-server.exe`をビルド
5. `ggml-small.bin`をダウンロードしてSHA-1を検証
6. 公式`rotensyo/VoicepeakProxy` Release v1.2.1から`VoicepeakProxyCore`を取得
7. C#のVOICEPEAK Bridgeをビルド
8. Bridgeの実行先へVoicepeakProxyCore一式を配置
9. `app_config.json`がなければ設定例から作成

セットアップは再実行できます。既に取得済みのファイルは基本的に再利用します。WhisperモデルとVoicepeakProxyCoreを再取得する場合は`-ForceDownload`を付けます。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -ForceDownload
```

### 4. VOICEPEAKの場所を設定

セットアップで作成された`app_config.json`を開きます。

```json
{
  "voicepeak_exe": "C:\\Path\\To\\VOICEPEAK\\voicepeak.exe"
}
```

`voicepeak_exe`だけ、実際にインストールされている`voicepeak.exe`の絶対パスへ変更してください。

例:

```json
{
  "voicepeak_exe": "D:\\Apps\\VOICEPEAK\\voicepeak.exe"
}
```

個人環境のパスを含む`app_config.json`は`.gitignore`対象です。配布用の初期値は`app_config.example.json`にあります。

### 5. 起動

`start.bat`をダブルクリックするか、ターミナルから実行します。

```cmd
start.bat
```

終了するときはコンソールで`Ctrl+C`を押します。

## パスの扱い

VOICEPEAK本体を除くすべてのファイルは、cloneしたリポジトリを基準に自動解決します。フォルダーごと別の場所へ移動しても、Pythonファイルのパスを書き換える必要はありません。

主な配置先:

```text
VoiceChatAIwithVoicepeak\
├─ voice_chat_gemini.py
├─ app_config.json                 # 利用者が設定、Git管理外
├─ gemini_config.json
├─ system_instruction.txt
├─ requirements.txt
├─ setup.ps1
├─ start.bat
├─ .venv\                          # setup.ps1が作成、Git管理外
├─ whisper.cpp\                    # setup.ps1が取得、Git管理外
│  ├─ build-voice-chat\
│  └─ models\ggml-small.bin
├─ external\VoicepeakProxyCore\    # setup.ps1が取得、Git管理外
└─ voicepeak_proxy_test\
   └─ bin\Release\net48\          # setup.ps1がビルド、Git管理外
```

## 会話方法

起動後、最初の0.5秒間は環境ノイズを測定します。この間は声を出さないでください。

```text
発話待機中...
    ↓
100ms連続で声を検出
    ↓
● 発話検出
    ↓
話し終えて0.55秒無音になる
    ↓
■ 発話終了
    ↓
Whisper → Gemini → VOICEPEAK
```

発話開始前0.3秒も録音へ含まれます。1回の発話は最大20秒で、0.35秒未満の短い入力は誤検出として破棄されます。

Whisper、Gemini、VOICEPEAKの処理中もマイクは有効です。続けて話した音声はキューへ追加され、録音された順番に処理されます。

## 設定ファイル

### app_config.json

利用者の環境に依存する設定です。

| 設定 | 内容 |
| --- | --- |
| `voicepeak_exe` | `voicepeak.exe`の絶対パス |

### gemini_config.json

```json
{
  "model": "gemini-3.5-flash-lite",
  "max_output_tokens": 512,
  "thinking_level": "minimal",
  "history_max_turns": 30
}
```

| 設定 | 内容 |
| --- | --- |
| `model` | 使用するGeminiモデル |
| `max_output_tokens` | Gemini応答の最大トークン数 |
| `thinking_level` | Geminiの思考レベル |
| `history_max_turns` | 起動時に読み込む会話の往復数。`0`で無効 |

必要なら`temperature`も追加できます。省略時はモデルのデフォルト値を使用します。

### system_instruction.txt

Geminiの人格、話し方、返答ルールを記述します。変更は次回起動時から反映されます。

### 録音設定

録音関係は`voice_chat_gemini.py`冒頭で調整できます。

| 設定 | 初期値 | 内容 |
| --- | ---: | --- |
| `PRE_ROLL_SECONDS` | `0.3` | 発話開始前に含める秒数 |
| `SPEECH_START_SECONDS` | `0.10` | 発話開始に必要な連続時間 |
| `SILENCE_END_SECONDS` | `0.55` | 発話終了と判定する無音時間 |
| `MIN_SPEECH_SECONDS` | `0.35` | これ未満の発話を破棄 |
| `MAX_RECORD_SECONDS` | `20.0` | 1発話の最大録音時間 |
| `TRAILING_SILENCE_KEEP_SECONDS` | `0.15` | WAVへ残す末尾無音 |
| `NOISE_CALIBRATION_SECONDS` | `0.50` | 起動時のノイズ測定時間 |
| `MIN_SPEECH_RMS` | `150.0` | 発話閾値の最低値 |
| `NOISE_THRESHOLD_MULTIPLIER` | `2.5` | 環境ノイズへ掛ける倍率 |
| `CAPTURE_QUEUE_SECONDS` | `5.0` | 入力取りこぼし対策のバッファ |

## 保存されるファイル

すべてリポジトリ直下へ保存され、GitHubにはアップロードされません。

- `input.wav`: 直近の発話。PCM16／16 kHz／mono
- `conversation_history.jsonl`: ユーザー発話とGemini返答の会話履歴
- `whisper_server.log`: Whisper Serverのログ

会話履歴からは、ユーザー発話とアシスタント返答が揃った直近の会話だけを起動時に読み込みます。

## ノイズの扱い

Whisperが次のような効果音表記だけを返した場合はGeminiへ送りません。

```text
(音楽)
（パッ）
[拍手]
【雑音】
```

本文と効果音が混ざった場合は効果音部分だけを削除します。「ご視聴ありがとうございました」など、無音時に出やすい代表的な誤認識も無視します。

## VOICEPEAK読み上げ中の注意

マイクはVOICEPEAKの読み上げ中も有効です。スピーカーの音をマイクが拾うと自己応答を繰り返す可能性があります。

- ヘッドホンまたはイヤホンを使う
- Windowsや音声デバイスのエコー抑制を有効にする
- マイクをスピーカーから離す

VOICEPEAKは1プロセスだけ起動し、ウィンドウを最小化しないで使用してください。

## トラブルシューティング

### setup.ps1で前提ツールが見つからない

エラーに表示されたツールをインストールし、新しいPowerShellを開いて再実行してください。Visual Studio Build Toolsでは「C++によるデスクトップ開発」を有効にします。

### whisper-server.exeのビルドに失敗する

- CPU版では`setup.ps1`を引数なしで再実行
- CUDA版ではCUDA Toolkitと対応GPUを確認
- Visual Studio Build Tools 2022のC++ワークロードを確認
- 古い途中生成物と分けるため、自動セットアップは`build-voice-chat`を使用します

### VOICEPEAKが見つからない

`app_config.json`のJSON構文と`voicepeak_exe`を確認します。Windowsのパス区切りはJSON内で`\\`と書きます。

### マイク入力の取りこぼし

「マイク入力の取りこぼしを検出しました」と表示されても監視は継続します。頻発する場合はCPU負荷、入力デバイス、マイク拡張機能を確認し、必要なら`CAPTURE_QUEUE_SECONDS`を増やします。

### 発話を検出しない／雑音を拾いすぎる

- 検出しない場合: `MIN_SPEECH_RMS`または`NOISE_THRESHOLD_MULTIPLIER`を下げる
- 雑音を拾う場合: これらを上げるか`SPEECH_START_SECONDS`を長くする

### Geminiから本文が返らない

一度だけなら同じ内容を話し直してください。頻発する場合は`max_output_tokens`を増やす、`temperature`を外す、`thinking_level`を`minimal`にする方法があります。

### VOICEPEAKが読み上げない

- VOICEPEAKが1プロセスだけ起動しているか確認
- VOICEPEAKを最小化していないか確認
- `app_config.json`のパスを確認
- `setup.ps1`を再実行してBridgeを再ビルド

## 外部プロジェクト

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [VoicepeakProxy](https://github.com/rotensyo/VoicepeakProxy)

各プロジェクトとVOICEPEAKの利用条件を確認して使用してください。
