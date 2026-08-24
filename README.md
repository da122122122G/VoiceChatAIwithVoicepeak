# Gemini × Whisper.cpp × VOICEPEAK 音声チャット

マイクへ話しかけると、Whisperが文字起こしし、Geminiの返答をVOICEPEAKが読み上げるWindows向け音声チャットです。

現在はキー操作不要の常時リスニング方式です。Whisper・Gemini・VOICEPEAKの処理中もマイク入力を続け、次の発話をキューへ保存します。

## 必要なもの

- Windows 10 / 11
- Python 3.12
- Gemini APIキー
- VOICEPEAK
- VoicepeakProxyCore
- whisper.cppとWhisperの`small`モデル
- Git、CMake、Ninja、Visual Studio Build Tools 2022
- .NET SDK／.NET Framework 4.8
- NVIDIA GPUを使用する場合はCUDA Toolkit

以下では、作業フォルダを`C:\voice_ai`として説明します。

## 初回セットアップ

### 1. Pythonとパッケージ

Python 3.12をインストールし、PowerShellで確認します。

```powershell
python --version
python -m pip install numpy sounddevice soundfile requests google-genai
```

### 2. Gemini APIキー

Gemini APIキーをWindowsの環境変数`GEMINI_API_KEY`へ設定します。

```powershell
setx GEMINI_API_KEY "ここにAPIキー"
```

新しいPowerShellを開き直し、設定を確認します。

```powershell
echo $env:GEMINI_API_KEY
```

### 3. VOICEPEAK

VOICEPEAKをインストールし、Pythonファイルのパスを実際の場所に合わせます。

```python
VOICEPEAK_EXE = r"H:\動画\VOICEPEAK\voicepeak.exe"
```

VOICEPEAK側では、ナレーター、速度、ピッチ、感情、音声出力先などを設定してください。VOICEPEAKは複数同時に起動せず、ウィンドウを最小化しないで使用します。

### 4. whisper.cpp

Visual Studio Build Toolsでは「C++によるデスクトップ開発」を有効にしてください。

```powershell
cd C:\
mkdir voice_ai
cd C:\voice_ai
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
```

CPU版:

```powershell
cmake -B build -G Ninja
cmake --build build -j 8
```

NVIDIA GPU／CUDA版:

```powershell
cmake -B build -G Ninja -DGGML_CUDA=ON
cmake --build build -j 8
```

CUDAアーキテクチャを指定する例:

```powershell
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build -j 8
```

ビルド後、次のファイルが必要です。

```text
C:\voice_ai\whisper.cpp\build\bin\whisper-server.exe
```

### 5. Whisperモデル

whisper.cpp付属のダウンロードスクリプトなどで`small`モデルを取得し、次の場所へ配置します。

```text
C:\voice_ai\whisper.cpp\models\ggml-small.bin
```

### 6. VoicepeakProxyCore

VoicepeakProxyのReleaseから`VoicepeakProxyCore.zip`を取得し、展開します。

```text
C:\voice_ai\VoicepeakProxyCore-1.2.1\
```

主な必要ファイル:

```text
VoicepeakProxyCore.dll
VoicepeakProxyCore.deps\
EasyHook32.dll
EasyHook64.dll
EasyLoad32.dll
EasyLoad64.dll
Interop.UIAutomationClient.dll
NAudio.Core.dll
NAudio.Wasapi.dll
```

配置場所を変更する場合は、`VoicepeakProxyTest.csproj`の`HintPath`も合わせて変更します。

### 7. VOICEPEAK Bridge

次の2ファイルを配置します。

```text
C:\voice_ai\voicepeak_proxy_test\Program.cs
C:\voice_ai\voicepeak_proxy_test\VoicepeakProxyTest.csproj
```

Bridgeをビルドします。

```powershell
cd C:\voice_ai\voicepeak_proxy_test
dotnet build -c Release
```

生成先:

```text
C:\voice_ai\voicepeak_proxy_test\bin\Release\net48\VoicepeakProxyTest.exe
```

VoicepeakProxyCore一式を出力先へコピーします。

```powershell
Copy-Item `
  "C:\voice_ai\VoicepeakProxyCore-1.2.1\*" `
  "C:\voice_ai\voicepeak_proxy_test\bin\Release\net48\" `
  -Recurse -Force
```

必要に応じてDLLのブロックを解除します。

```powershell
Get-ChildItem `
  "C:\voice_ai\voicepeak_proxy_test\bin\Release\net48" `
  -Recurse |
  Unblock-File
```

### 8. ファイル配置とパス

次の3ファイルを配置します。

```text
C:\voice_ai\voice_chat_gemini.py
C:\voice_ai\gemini_config.json
C:\voice_ai\system_instruction.txt
```

標準パス:

```text
Whisper Server:
C:\voice_ai\whisper.cpp\build\bin\whisper-server.exe

Whisper model:
C:\voice_ai\whisper.cpp\models\ggml-small.bin

VOICEPEAK Bridge:
C:\voice_ai\voicepeak_proxy_test\bin\Release\net48\VoicepeakProxyTest.exe
```

異なる場所を使う場合は、Pythonファイル冒頭の設定を変更してください。

```python
VOICEPEAK_EXE = r"..."
VOICEPEAK_BRIDGE = r"..."
WHISPER_SERVER_EXE = r"..."
WHISPER_MODEL = r"..."
INPUT_WAV = r"..."
CONVERSATION_LOG = r"..."
GEMINI_CONFIG_FILE = r"..."
SYSTEM_INSTRUCTION_FILE = r"..."
```

## 起動

PowerShellで次を実行します。

```powershell
cd C:\voice_ai
python .\voice_chat_gemini.py
```

起動時に次の処理が自動実行されます。

- Gemini設定とシステム指示の読み込み
- 過去の会話履歴の読み込み
- Whisper Serverの起動または既存Serverへの接続
- VOICEPEAK本体の起動確認
- VOICEPEAK Bridgeの起動
- マイク入力の開始
- 0.5秒間の環境ノイズ測定

表示例:

```text
================================
 Gemini Voice Chat
================================

Whisper Server準備完了
VOICEPEAK確認OK
VOICEPEAK Bridge準備完了

常時リスニングを開始します。
最初の0.50秒間は環境ノイズを測定します。
Ctrl+Cで終了。

ノイズレベル: 42.0 / 発話閾値: 150.0
発話待機中...
```

起動直後の0.5秒間は声を出さないでください。声が環境ノイズとして測定されると、発話を検出しにくくなることがあります。

## 会話方法

SpaceやF8は使用しません。`発話待機中...`と表示されたら、そのままマイクへ話します。

```text
話し始める
    ↓
100ms連続で声を検出
    ↓
● 発話検出
    ↓
話し終えて0.55秒無音になる
    ↓
■ 発話終了
    ↓
Whisperで文字起こし
    ↓
Geminiが返答
    ↓
VOICEPEAKが読み上げ
```

発話開始前0.3秒も録音へ含まれるため、語頭が切れにくくなっています。

1回の発話は最大20秒です。0.35秒未満の短い入力は、誤検出として破棄されます。

## 処理中に続けて話す場合

マイクは次の処理中も有効です。

- Whisperによる文字起こし
- Geminiの返答生成
- VOICEPEAKへの送信
- VOICEPEAKの読み上げ

処理中に話した音声はキューへ追加され、先に録音された発話から順番に処理されます。

```text
■ 発話終了
音声処理キューへ追加 (待機数: 1)
発話待機中...
```

## 終了

`Ctrl+C`を押します。

Python自身が起動したWhisper ServerとVOICEPEAK Bridgeも終了します。実行前から起動していたWhisper Serverは終了しません。

## 保存されるファイル

### input.wav

直近の発話を次の形式で保存します。

- PCM16
- 16 kHz
- mono

```text
C:\voice_ai\input.wav
```

発話終了判定には0.55秒の無音を使いますが、WAVへ残す末尾無音は0.15秒です。

`input.wav`は発話ごとに上書きされます。過去の音声は保存されません。

### conversation_history.jsonl

認識されたユーザー発話とGeminiの返答を、日時付きで追記します。

```text
C:\voice_ai\conversation_history.jsonl
```

記録例:

```json
{"timestamp": "2026-08-24T10:30:12+09:00", "role": "user", "text": "今何時？"}
{"timestamp": "2026-08-24T10:30:13+09:00", "role": "assistant", "text": "今は10時30分くらいだよ。"}
```

起動時にこのファイルから、ユーザー発話とアシスタント返答が揃った会話を読み込みます。本文のないGemini応答などでユーザー行だけが残った記録は、履歴の構造を壊さないよう読み飛ばします。

読み込む件数は`gemini_config.json`の`history_max_turns`で指定します。初期値は直近30往復です。`0`にすると過去ログを読み込みません。

### whisper_server.log

Whisper Serverのログです。

```text
C:\voice_ai\whisper_server.log
```

## 現在日時

Geminiにはユーザー発話と一緒に、PCの現在日時とタイムゾーンを渡しています。

そのため、次のような質問ができます。

```text
今何時？
今日は何日？
今日の予定を考えたい
```

日時はPCの時計を基準にします。

## ノイズの扱い

録音開始には、動的に計算した発話閾値を100ms連続で超える必要があります。短いキーボード音などでは開始しにくい設定です。

Whisperが次のような効果音表記を返した場合は、Geminiへ送りません。

```text
(音楽)
（パッ）
[拍手]
【雑音】
```

本文と効果音が混ざった場合は、効果音部分だけを削除します。

```text
認識結果: （音楽）こんにちは
Geminiへ送る内容: こんにちは
```

無音時に誤認識されやすい次の文も無視します。

```text
ご視聴ありがとうございました
ご清聴ありがとうございました
チャンネル登録お願いします
```

## VOICEPEAK読み上げ中の注意

マイクはVOICEPEAKの読み上げ中も有効です。

スピーカーの音をマイクが拾うと、VOICEPEAK自身の声をユーザー発話として認識し、自己応答を繰り返す可能性があります。

次のいずれかを推奨します。

- ヘッドホンまたはイヤホンを使う
- Windowsや音声デバイスのエコー抑制を有効にする
- マイクをスピーカーから離す

Bridgeは`QUEUED|job_id`を受け取った時点で送信完了とします。VOICEPEAKの実際の再生終了は待ちません。

## 主な設定

### Gemini設定

モデルや履歴の読込件数は、次のファイルで設定します。

```text
C:\voice_ai\gemini_config.json
```

初期設定:

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
| `history_max_turns` | 起動時に読み込む会話の往復数 |

必要な場合は`temperature`もJSONへ追加できます。省略時はモデルのデフォルト値を使用します。

### システム指示

人格、話し方、返答ルールは次のテキストファイルへ記述します。

```text
C:\voice_ai\system_instruction.txt
```

変更内容は次回起動時から反映されます。Pythonファイルへ長いプロンプトを直接記述する必要はありません。

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

発話を検出しない場合は`MIN_SPEECH_RMS`または`NOISE_THRESHOLD_MULTIPLIER`を少し下げます。雑音を拾いすぎる場合は上げます。

## トラブルシューティング

### マイク入力の取りこぼし

次の表示が出ても、入力監視は自動的に継続します。

```text
マイク入力の取りこぼしを検出しました。
入力監視は自動的に継続します。
```

頻繁に発生する場合:

- CPU負荷の高いアプリを終了する
- 仮想オーディオデバイスやマイク拡張機能を見直す
- `CAPTURE_QUEUE_SECONDS`を増やす
- `INPUT_LATENCY = "high"`になっているか確認する
- マイクが16 kHz入力へ対応しているか確認する

### 発話を検出しない

- 起動直後の0.5秒間に話していないか確認する
- Windowsの入力デバイスとマイク音量を確認する
- `MIN_SPEECH_RMS`を少し下げる
- `NOISE_THRESHOLD_MULTIPLIER`を少し下げる

### 雑音を発話として検出する

- `MIN_SPEECH_RMS`を少し上げる
- `NOISE_THRESHOLD_MULTIPLIER`を少し上げる
- `SPEECH_START_SECONDS`を少し長くする

### Geminiから本文が返らない

```text
Geminiから本文のないレスポンスが返されました。
finish_reason: FinishReason.STOP
```

`STOP`は正常終了を表しますが、テキスト部分のない応答が返る場合があります。一度だけなら同じ内容をもう一度話してください。

頻発する場合は、Gemini設定の`max_output_tokens`を512以上へ増やす、`temperature`指定を外す、thinking levelを`minimal`にする、といった調整候補があります。

### 音声を認識しなくなった

- マイクがほかのアプリに占有されていないか確認する
- Windowsの既定の入力デバイスを確認する
- プログラムをCtrl+Cで終了して再起動する
- `whisper_server.log`を確認する

### VOICEPEAKが読み上げない

- VOICEPEAKが1プロセスだけ起動しているか確認する
- VOICEPEAKを最小化していないか確認する
- BridgeとVoicepeakProxyCoreのDLL一式を確認する
- Bridgeを再ビルドする場合は、実行中のBridgeを先に終了する
