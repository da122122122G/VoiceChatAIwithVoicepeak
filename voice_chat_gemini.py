import base64
import io
import json
import math
import os
import queue
import re
import socket
import subprocess
import threading
import time
import wave

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import sounddevice as sd
import requests

from google import genai
from google.genai import types

WHISPER_SESSION = requests.Session()
WHISPER_REQUEST_DATA = {
    "response_format": "json",
    "temperature": "0.0",
    "temperature_inc": "0.2",
}

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def repository_path(*parts):
    return os.path.join(BASE_DIR, *parts)


# ============================================================
# 設定
# ============================================================

# ------------------------------------------------------------
# VOICEPEAK
# ------------------------------------------------------------

APP_CONFIG_FILE = repository_path(
    "app_config.json"
)

# 利用者ごとに異なるため、app_config.jsonから起動時に読み込む。
VOICEPEAK_EXE = None

VOICEPEAK_BRIDGE = repository_path(
    "voicepeak_proxy_test",
    "bin",
    "Release",
    "net48",
    "VoicepeakProxyTest.exe",
)

# VOICEPEAKのプロセス起動後、操作対象ウィンドウと内部初期化を待つ。
VOICEPEAK_WINDOW_TIMEOUT = 30.0
VOICEPEAK_GUI_SETTLE_SECONDS = 5.0

# VOICEPEAKの内部初期化が遅い場合にBridge起動を再試行する。
VOICEPEAK_BRIDGE_START_RETRY_SECONDS = (
    3.0,
    5.0,
    8.0,
)


# ------------------------------------------------------------
# Whisper Server
# ------------------------------------------------------------

WHISPER_SERVER_CANDIDATES = [
    repository_path(
        "whisper.cpp",
        "build-voice-chat",
        "bin",
        "Release",
        "whisper-server.exe",
    ),
    repository_path(
        "whisper.cpp",
        "build-voice-chat",
        "bin",
        "whisper-server.exe",
    ),
    repository_path(
        "whisper.cpp",
        "build",
        "bin",
        "whisper-server.exe",
    ),
    repository_path(
        "whisper.cpp",
        "build",
        "bin",
        "Release",
        "whisper-server.exe",
    ),
]

WHISPER_SERVER_EXE = next(
    (
        path
        for path in WHISPER_SERVER_CANDIDATES
        if os.path.isfile(path)
    ),
    WHISPER_SERVER_CANDIDATES[0],
)

WHISPER_MODEL = repository_path(
    "whisper.cpp",
    "models",
    "ggml-small.bin",
)

WHISPER_HOST = "127.0.0.1"
WHISPER_PORT = 8080

WHISPER_URL = (
    f"http://{WHISPER_HOST}:"
    f"{WHISPER_PORT}/inference"
)

WHISPER_LOG = repository_path(
    "whisper_server.log"
)

# Whisper設定
WHISPER_THREADS = 8
WHISPER_BEST_OF = 1
WHISPER_BEAM_SIZE = 1

# Server起動待ち時間
WHISPER_START_TIMEOUT = 30.0

# CUDAの初回推論で発生する初期化コストを会話前に消化する。
# この環境では1回目だけでなく2回目も遅くなる場合があるため2回実行する。
WHISPER_WARMUP_REQUESTS = 2
WHISPER_WARMUP_AUDIO_SECONDS = 0.25
WHISPER_WARMUP_TIMEOUT = 30


# ------------------------------------------------------------
# 録音
# ------------------------------------------------------------

INPUT_WAV = repository_path("input.wav")
CONVERSATION_LOG = repository_path(
    "conversation_history.jsonl"
)
GEMINI_CONFIG_FILE = repository_path(
    "gemini_config.json"
)
SYSTEM_INSTRUCTION_FILE = repository_path(
    "system_instruction.txt"
)

SAMPLE_RATE = 16000
CHANNELS = 1

# 語頭を切らないための録音前バッファ
PRE_ROLL_SECONDS = 0.3

# 16kHzで10ms
AUDIO_BLOCK_SIZE = 160
INPUT_LATENCY = "high"
CAPTURE_QUEUE_SECONDS = 5.0

SPEECH_START_SECONDS = 0.10
SILENCE_END_SECONDS = 0.55
MIN_SPEECH_SECONDS = 0.35
MAX_RECORD_SECONDS = 20.0
TRAILING_SILENCE_KEEP_SECONDS = 0.15

# 起動直後に環境ノイズを測定し、発話判定の基準にする
NOISE_CALIBRATION_SECONDS = 0.50
MIN_SPEECH_RMS = 150.0
NOISE_THRESHOLD_MULTIPLIER = 2.5

# Whisperが雑音に対して返すことがある効果音表記
NOISE_LABELS = {
    "音楽",
    "bgm",
    "拍手",
    "笑い",
    "笑い声",
    "雑音",
    "ノイズ",
    "無音",
    "環境音",
    "効果音",
    "咳",
    "咳払い",
    "ため息",
    "息",
    "呼吸音",
    "物音",
    "パッ",
    "パン",
    "ポン",
    "カチッ",
    "ガサガサ",
    "ブツッ",
}

NOISE_ONLY_TEXTS = {
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "チャンネル登録お願いします",
}

BRACKETED_TEXT_PATTERN = re.compile(
    r"[\(（\[［【<＜]"
    r"\s*([^\)）\]］】>＞]{1,20}?)\s*"
    r"[\)）\]］】>＞]",
    re.IGNORECASE,
)

KATAKANA_SOUND_PATTERN = re.compile(
    r"^[ァ-ヶー・ッ]{1,8}$"
)


# ============================================================
# アプリ設定
# ============================================================

def load_app_settings():

    try:
        with open(
            APP_CONFIG_FILE,
            "r",
            encoding="utf-8",
        ) as config_file:
            settings = json.load(config_file)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            "app_config.jsonが見つかりません。\n"
            "app_config.example.jsonをコピーし、"
            "voicepeak_exeを設定してください:\n"
            f"{APP_CONFIG_FILE}"
        ) from e

    except (OSError, ValueError) as e:
        raise RuntimeError(
            "アプリ設定ファイルを読み込めません: "
            f"{APP_CONFIG_FILE}"
        ) from e

    if not isinstance(settings, dict):
        raise ValueError(
            "app_config.jsonのルートは"
            "JSONオブジェクトにしてください。"
        )

    voicepeak_exe = str(
        settings.get("voicepeak_exe", "")
    ).strip()

    if not voicepeak_exe:
        raise ValueError(
            "app_config.jsonのvoicepeak_exeへ"
            "VOICEPEAKの実行ファイルを指定してください。"
        )

    voicepeak_exe = os.path.expandvars(
        os.path.expanduser(voicepeak_exe)
    )

    if not os.path.isabs(voicepeak_exe):
        voicepeak_exe = repository_path(
            voicepeak_exe
        )

    voicepeak_exe = os.path.abspath(
        voicepeak_exe
    )

    if not os.path.isfile(voicepeak_exe):
        raise FileNotFoundError(
            "VOICEPEAKが見つかりません。"
            "app_config.jsonのvoicepeak_exeを"
            "確認してください:\n"
            f"{voicepeak_exe}"
        )

    return {
        "voicepeak_exe": voicepeak_exe,
    }


def initialize_app_settings():

    global VOICEPEAK_EXE

    settings = load_app_settings()
    VOICEPEAK_EXE = settings[
        "voicepeak_exe"
    ]

    print(
        "アプリ設定: "
        f"{APP_CONFIG_FILE}"
    )


# ============================================================
# Whisper Server
# ============================================================

class WhisperServer:

    def __init__(self):
        self.process = None
        self.log_file = None
        self.owns_process = False

        self.start()
        self._warm_up()


    def _is_running(self):
        """
        Whisper Serverのポートが開いているか確認。
        """

        try:
            with socket.create_connection(
                (WHISPER_HOST, WHISPER_PORT),
                timeout=0.2
            ):
                return True

        except OSError:
            return False


    def _close_log(self):

        if self.log_file is not None:

            try:
                self.log_file.close()

            except Exception:
                pass

            self.log_file = None


    def _warm_up(self):
        """短い無音を推論し、CUDAの遅延初期化を会話前に済ませる。"""

        sample_count = max(
            1,
            round(
                SAMPLE_RATE
                * WHISPER_WARMUP_AUDIO_SECONDS
            ),
        )
        audio = np.zeros(sample_count, dtype=np.int16)
        wav_data = encode_pcm16_wav(audio)

        print("Whisperをウォームアップ中...")
        start = time.perf_counter()

        try:
            for _ in range(WHISPER_WARMUP_REQUESTS):
                response = WHISPER_SESSION.post(
                    WHISPER_URL,
                    files={
                        "file": (
                            "warmup.wav",
                            wav_data,
                            "audio/wav",
                        )
                    },
                    data=WHISPER_REQUEST_DATA,
                    timeout=WHISPER_WARMUP_TIMEOUT,
                )
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            # ウォームアップ失敗だけでアプリ全体は停止させない。
            print(f"Whisperウォームアップ警告: {e}")
            return

        elapsed = time.perf_counter() - start
        print(
            "Whisperウォームアップ完了: "
            f"{elapsed:.2f} 秒"
        )


    def start(self):

    # すでに起動している場合はそのまま利用
        if self._is_running():

            print()
            print("Whisper Server確認OK")

            self.owns_process = False
            return


        print()
        print("Whisper Serverを起動中...")


        if not os.path.isfile(WHISPER_SERVER_EXE):
            raise FileNotFoundError(
                "whisper-server.exeが見つかりません:\n"
                f"{WHISPER_SERVER_EXE}"
            )


        if not os.path.isfile(WHISPER_MODEL):
            raise FileNotFoundError(
                "Whisperモデルが見つかりません:\n"
                f"{WHISPER_MODEL}"
            )


        self.log_file = open(
            WHISPER_LOG,
            "a",
            encoding="utf-8",
            buffering=1
        )


        command = [
            WHISPER_SERVER_EXE,

            "-m", WHISPER_MODEL,
            "-l", "ja",
            "-t", str(WHISPER_THREADS),

            "-bo", str(WHISPER_BEST_OF),
            "-bs", str(WHISPER_BEAM_SIZE),

            "--host", WHISPER_HOST,
            "--port", str(WHISPER_PORT),
        ]


        self.process = subprocess.Popen(
            command,

            stdout=self.log_file,
            stderr=subprocess.STDOUT,

            cwd=os.path.dirname(
                WHISPER_SERVER_EXE
            ),

            creationflags=subprocess.CREATE_NO_WINDOW,
        )   


        self.owns_process = True


        deadline = (
            time.perf_counter()
            + WHISPER_START_TIMEOUT
        )


        while time.perf_counter() < deadline:

            if self.process.poll() is not None:

                return_code = self.process.returncode

                self._close_log()

                raise RuntimeError(
                    "Whisper Serverの起動に失敗しました。\n"
                    f"終了コード: {return_code}\n"
                    f"ログ: {WHISPER_LOG}"
                )


            if self._is_running():

                print("Whisper Server準備完了")
                return


            time.sleep(0.1)


        self.close()

        raise RuntimeError(
            "Whisper Serverの起動がタイムアウトしました。\n"
            f"ログ: {WHISPER_LOG}"
        )


    def close(self):

        # 自分で起動したServerだけ終了する。
        # 手動起動済みのServerは終了させない。
        if (
            self.owns_process
            and self.process is not None
            and self.process.poll() is None
        ):

            print()
            print(
                "Whisper Serverを終了中..."
            )

            try:

                self.process.terminate()

                self.process.wait(
                    timeout=3
                )

            except Exception:

                try:
                    self.process.kill()

                except Exception:
                    pass


        self.process = None

        self._close_log()


# ============================================================
# VOICEPEAK起動確認
# ============================================================

def get_voicepeak_processes():

    # 日本語パスが文字化けしないよう、
    # PowerShell側でUTF-8を明示する。
    ps_script = r"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Get-Process voicepeak -ErrorAction SilentlyContinue |
    ForEach-Object {
        [PSCustomObject]@{
            path = $_.Path
            has_main_window = (
                $_.MainWindowHandle -ne 0 -and
                $_.MainWindowTitle -match '^VOICEPEAK(?:\s|$)'
            )
        }
    } |
    ConvertTo-Json -Compress
"""


    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            ps_script
        ],

        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )


    stdout = (result.stdout or "").strip()

    if not stdout:
        return []

    try:
        processes = json.loads(stdout)
    except ValueError as e:
        raise RuntimeError(
            "VOICEPEAKのプロセス情報を解析できません。"
        ) from e

    if isinstance(processes, dict):
        processes = [processes]

    return [
        process
        for process in processes
        if process.get("path")
    ]


def normalize_paths(paths):

    return [
        os.path.normcase(
            os.path.abspath(path)
        )
        for path in paths
    ]


def wait_for_voicepeak_window(
    correct_path,
    timeout=VOICEPEAK_WINDOW_TIMEOUT,
):

    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:

        processes = get_voicepeak_processes()
        process_paths = [
            process["path"]
            for process in processes
        ]
        correct_process_count = normalize_paths(
            process_paths
        ).count(correct_path)

        if correct_process_count > 1:
            raise RuntimeError(
                "VOICEPEAKが複数起動しています。"
                "すべて終了してから、もう一度"
                "起動してください。"
            )

        window_paths = [
            process["path"]
            for process in processes
            if process.get("has_main_window")
        ]

        if correct_path in normalize_paths(
            window_paths
        ):

            print("VOICEPEAKウィンドウ確認OK")

            # MainWindowHandle生成後の内部UI初期化を待つ。
            time.sleep(
                VOICEPEAK_GUI_SETTLE_SECONDS
            )

            return True

        time.sleep(0.5)

    return False


def ensure_voicepeak_running(
    auto_start=True
):

    processes = get_voicepeak_processes()
    paths = [
        process["path"]
        for process in processes
    ]


    correct_path = os.path.normcase(
        os.path.abspath(
            VOICEPEAK_EXE
        )
    )


    normalized_paths = normalize_paths(paths)


    # app_config.jsonで指定したVOICEPEAKが起動中
    if correct_path in normalized_paths:

        print()
        print("VOICEPEAKプロセス確認OK")

        if not wait_for_voicepeak_window(
            correct_path
        ):
            raise RuntimeError(
                "VOICEPEAKのウィンドウ準備が"
                "タイムアウトしました。"
            )

        print("VOICEPEAK準備完了")

        return True


    # 別インストールのVOICEPEAKが起動中
    if paths:

        print()
        print(
            "別のVOICEPEAKが起動しています:"
        )

        for path in paths:
            print(" ", path)


        raise RuntimeError(
            "app_config.jsonで指定したVOICEPEAKを"
            "起動してください:\n"
            f"{VOICEPEAK_EXE}"
        )


    if not auto_start:
        return False


    # VOICEPEAKが起動していないので自動起動
    print()
    print(
        "VOICEPEAKを起動します..."
    )


    subprocess.Popen(
        [VOICEPEAK_EXE],
        cwd=os.path.dirname(
            VOICEPEAK_EXE
        )
    )


    if wait_for_voicepeak_window(correct_path):

        print(
            "VOICEPEAK起動完了"
        )

        return True


    raise RuntimeError(
        "VOICEPEAKを起動できませんでした:\n"
        f"{VOICEPEAK_EXE}"
    )


# ============================================================
# Gemini
# ============================================================

client = None
chat = None

DEFAULT_GEMINI_SETTINGS = {
    "model": "gemini-3.1-flash-lite",
    "max_output_tokens": 512,
    "thinking_level": "minimal",
    "history_max_turns": 30,
    "request_timeout_seconds": 15.0,
}


def load_gemini_settings():

    settings = DEFAULT_GEMINI_SETTINGS.copy()

    try:
        with open(
            GEMINI_CONFIG_FILE,
            "r",
            encoding="utf-8",
        ) as config_file:
            loaded = json.load(config_file)

    except FileNotFoundError:
        print(
            "Gemini設定ファイルが見つからないため"
            "初期値を使用します:"
        )
        print(GEMINI_CONFIG_FILE)
        return settings

    except (OSError, ValueError) as e:
        raise RuntimeError(
            "Gemini設定ファイルを読み込めません: "
            f"{GEMINI_CONFIG_FILE}"
        ) from e

    if not isinstance(loaded, dict):
        raise ValueError(
            "Gemini設定ファイルのルートは"
            "JSONオブジェクトにしてください。"
        )

    settings.update(loaded)
    return settings


def load_system_instruction():

    try:
        with open(
            SYSTEM_INSTRUCTION_FILE,
            "r",
            encoding="utf-8",
        ) as instruction_file:
            instruction = instruction_file.read().strip()

    except OSError as e:
        raise RuntimeError(
            "システム指示ファイルを読み込めません: "
            f"{SYSTEM_INSTRUCTION_FILE}"
        ) from e

    if not instruction:
        raise ValueError(
            "システム指示ファイルが空です: "
            f"{SYSTEM_INSTRUCTION_FILE}"
        )

    return instruction


def load_conversation_history(max_turns):

    if max_turns <= 0:
        return []

    if not os.path.isfile(CONVERSATION_LOG):
        print("会話履歴: まだ記録がありません。")
        return []

    # 長期間使ってログが大きくなっても、必要な直近分だけを保持する。
    completed_turns = deque(maxlen=max_turns)
    pending_user_text = None
    skipped_lines = 0

    try:
        with open(
            CONVERSATION_LOG,
            "r",
            encoding="utf-8",
        ) as log_file:
            for line in log_file:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except ValueError:
                    skipped_lines += 1
                    continue

                role = record.get("role")
                text = str(record.get("text", "")).strip()

                if not text:
                    continue

                if role == "user":
                    pending_user_text = text

                elif (
                    role == "assistant"
                    and pending_user_text is not None
                ):
                    completed_turns.append(
                        (
                            pending_user_text,
                            text,
                        )
                    )
                    pending_user_text = None

    except OSError as e:
        raise RuntimeError(
            "会話履歴を読み込めません: "
            f"{CONVERSATION_LOG}"
        ) from e

    history = []

    for user_text, assistant_text in completed_turns:
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=user_text
                    )
                ],
            )
        )
        history.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=assistant_text
                    )
                ],
            )
        )

    print(
        "会話履歴を読み込み: "
        f"{len(completed_turns)} 往復"
    )

    if skipped_lines:
        print(
            "読み飛ばした不正なログ行: "
            f"{skipped_lines}"
        )

    return history


def initialize_gemini_chat():

    global client

    settings = load_gemini_settings()
    system_instruction = load_system_instruction()

    model = str(settings["model"])
    max_output_tokens = int(
        settings["max_output_tokens"]
    )
    history_max_turns = max(
        0,
        int(settings["history_max_turns"]),
    )
    request_timeout_seconds = max(
        1.0,
        float(settings["request_timeout_seconds"]),
    )

    config_kwargs = {
        "system_instruction": system_instruction,
        "max_output_tokens": max_output_tokens,
    }

    temperature = settings.get("temperature")

    if temperature is not None:
        config_kwargs["temperature"] = float(
            temperature
        )

    thinking_level = settings.get(
        "thinking_level"
    )

    if thinking_level:
        config_kwargs["thinking_config"] = (
            types.ThinkingConfig(
                thinking_level=str(
                    thinking_level
                )
            )
        )

    history = load_conversation_history(
        history_max_turns
    )

    print(f"Gemini model: {model}")
    print(
        "Gemini設定: "
        f"{GEMINI_CONFIG_FILE}"
    )
    print(
        "システム指示: "
        f"{SYSTEM_INSTRUCTION_FILE}"
    )
    print(
        "Geminiタイムアウト: "
        f"{request_timeout_seconds:.1f} 秒"
    )

    client = genai.Client(
        http_options=types.HttpOptions(
            timeout=round(
                request_timeout_seconds * 1000
            ),
            retry_options=types.HttpRetryOptions(
                attempts=1
            ),
        )
    )

    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            **config_kwargs
        ),
        history=history,
    )


# ============================================================
# 録音
# ============================================================


@dataclass(slots=True)
class Recording:
    frames: list
    speech_duration: float
    ended_by_silence: bool


@dataclass(slots=True)
class DetectorState:
    threshold: float | None = None
    above_count: int = 0
    silence_count: int = 0
    record_blocks: int = 0
    speech_span_blocks: int = 0
    last_voice_span_blocks: int = 0
    recording: bool = False
    frames: list = field(default_factory=list)
    stream_status_count: int = 0
    capture_drop_count: int = 0
    detector_error: Exception | None = None


def calculate_rms(chunk):
    """PCM16ブロックのRMSを一時配列1個で計算する。"""

    samples = chunk.reshape(-1).astype(np.float32)
    mean_square = np.dot(samples, samples) / samples.size
    return math.sqrt(float(mean_square))


def encode_pcm16_wav(audio):
    """PCM16音声をWhisperへそのまま渡せるWAVバイト列にする。"""

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())

    return wav_buffer.getvalue()


def append_conversation_log(role, text):

    record = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "role": role,
        "text": text,
    }

    try:
        with open(
            CONVERSATION_LOG,
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    except OSError as e:
        print(f"会話ログ保存エラー: {e}")


def process_recording(recording, bridge):

    block_seconds = AUDIO_BLOCK_SIZE / SAMPLE_RATE
    frames = recording.frames

    if recording.ended_by_silence:
        trim_seconds = max(
            0.0,
            SILENCE_END_SECONDS - TRAILING_SILENCE_KEEP_SECONDS,
        )
        trim_blocks = round(trim_seconds / block_seconds)

        if trim_blocks > 0 and len(frames) > trim_blocks:
            frames = frames[:-trim_blocks]

    if not frames:
        print("録音データがないためスキップします。")
        return

    audio = np.concatenate(frames, axis=0).reshape(-1)
    duration = audio.size / SAMPLE_RATE
    wav_data = encode_pcm16_wav(audio)

    with open(INPUT_WAV, "wb") as audio_file:
        audio_file.write(wav_data)

    print()
    print(f"発話時間: {recording.speech_duration:.2f} 秒")
    print(f"録音時間: {duration:.2f} 秒")
    print(f"保存先: {INPUT_WAV}")

    text = transcribe(wav_data)

    if not text:
        print("音声を認識できませんでした。")
        return

    append_conversation_log("user", text)

    response = ask_gemini_and_speak(
        text,
        bridge,
    )

    if not response:
        print(
            "Geminiから返答が"
            "ありませんでした。"
        )
        return

    append_conversation_log("assistant", response)

    print()
    print("--------------------------------")
    print("マイクは引き続き発話待機中です。")
    print("--------------------------------")


def run_continuous_conversation(bridge):

    print()
    print("常時リスニングを開始します。")
    print(
        f"最初の{NOISE_CALIBRATION_SECONDS:.2f}秒間は"
        "環境ノイズを測定します。"
    )
    print("Ctrl+Cで終了。")
    print()

    block_seconds = AUDIO_BLOCK_SIZE / SAMPLE_RATE
    pre_roll_blocks = max(1, round(PRE_ROLL_SECONDS / block_seconds))
    calibration_blocks = max(1, round(NOISE_CALIBRATION_SECONDS / block_seconds))
    start_blocks = max(1, round(SPEECH_START_SECONDS / block_seconds))
    silence_blocks = max(1, round(SILENCE_END_SECONDS / block_seconds))
    max_record_blocks = max(1, round(MAX_RECORD_SECONDS / block_seconds))
    capture_queue_blocks = max(
        1,
        round(CAPTURE_QUEUE_SECONDS / block_seconds),
    )

    audio_queue = queue.Queue()
    capture_queue = queue.Queue(maxsize=capture_queue_blocks)
    stop_event = threading.Event()
    pre_buffer = deque(maxlen=pre_roll_blocks)
    calibration_rms = []
    state = DetectorState()

    def reset_recording(pre_roll_source):
        pre_buffer.clear()
        pre_buffer.extend(pre_roll_source[-pre_roll_blocks:])

        state.above_count = 0
        state.silence_count = 0
        state.record_blocks = 0
        state.speech_span_blocks = 0
        state.last_voice_span_blocks = 0
        state.recording = False
        state.frames = []

    def process_audio_chunk(chunk):
        rms = calculate_rms(chunk)

        if state.threshold is None:
            calibration_rms.append(rms)
            pre_buffer.append(chunk)

            if len(calibration_rms) >= calibration_blocks:
                noise_floor = float(np.median(calibration_rms))
                state.threshold = max(
                    MIN_SPEECH_RMS,
                    noise_floor * NOISE_THRESHOLD_MULTIPLIER,
                )
                print(
                    f"ノイズレベル: {noise_floor:.1f} / "
                    f"発話閾値: {state.threshold:.1f}"
                )
                print("発話待機中...")
            return

        is_voice = rms >= state.threshold

        if not state.recording:
            pre_buffer.append(chunk)
            state.above_count = (
                state.above_count + 1 if is_voice else 0
            )

            if state.above_count >= start_blocks:
                state.recording = True
                state.speech_span_blocks = state.above_count
                state.last_voice_span_blocks = state.above_count
                state.frames.extend(pre_buffer)
                state.record_blocks = len(pre_buffer)
                pre_buffer.clear()
                print("● 発話検出")
            return

        state.frames.append(chunk)
        state.record_blocks += 1
        state.speech_span_blocks += 1

        if is_voice:
            state.silence_count = 0
            state.last_voice_span_blocks = state.speech_span_blocks
        else:
            state.silence_count += 1

        ended_by_silence = state.silence_count >= silence_blocks
        reached_maximum = state.record_blocks >= max_record_blocks

        if not ended_by_silence and not reached_maximum:
            return

        print("■ 発話終了")

        speech_duration = (
            state.last_voice_span_blocks
            * block_seconds
        )
        completed_frames = state.frames
        pre_roll_source = completed_frames[-pre_roll_blocks:]

        if speech_duration < MIN_SPEECH_SECONDS:
            print(
                "発話が短すぎるため誤検出として破棄します。"
                f"({speech_duration:.2f} 秒)"
            )
        else:
            audio_queue.put(
                Recording(
                    frames=completed_frames,
                    speech_duration=speech_duration,
                    ended_by_silence=ended_by_silence,
                )
            )
            print(
                "音声処理キューへ追加 "
                f"(待機数: {audio_queue.qsize()})"
            )

        reset_recording(pre_roll_source)
        print("発話待機中...")

    def detector_loop():
        last_status_print = 0.0
        reported_status_count = 0
        reported_drop_count = 0

        try:
            while not stop_event.is_set():
                try:
                    chunk = capture_queue.get(timeout=0.1)

                except queue.Empty:
                    continue

                try:
                    process_audio_chunk(chunk)

                finally:
                    capture_queue.task_done()

                status_count = state.stream_status_count
                drop_count = state.capture_drop_count
                now = time.perf_counter()

                if (
                    (
                        status_count != reported_status_count
                        or drop_count != reported_drop_count
                    )
                    and now - last_status_print >= 2.0
                ):
                    print(
                        "マイク入力の取りこぼしを検出しました。"
                        "入力監視は自動的に継続します。"
                    )
                    reported_status_count = status_count
                    reported_drop_count = drop_count
                    last_status_print = now

        except Exception as e:
            state.detector_error = e
            stop_event.set()

    def capture_callback(indata, frames_count, time_info, status):
        if status:
            state.stream_status_count += 1

        try:
            capture_queue.put_nowait(indata.copy())

        except queue.Full:
            state.capture_drop_count += 1

    detector_thread = threading.Thread(
        target=detector_loop,
        name="voice-detector",
        daemon=True,
    )
    detector_thread.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=AUDIO_BLOCK_SIZE,
            latency=INPUT_LATENCY,
            callback=capture_callback,
        ):
            while True:
                detector_error = state.detector_error

                if detector_error is not None:
                    raise RuntimeError(
                        "音声判定スレッドが停止しました。"
                    ) from detector_error

                try:
                    recording = audio_queue.get(timeout=0.1)

                except queue.Empty:
                    continue

                try:
                    process_recording(
                        recording,
                        bridge,
                    )

                finally:
                    audio_queue.task_done()

    finally:
        stop_event.set()
        detector_thread.join(timeout=2.0)


# ============================================================
# Whisper
# ============================================================

def clean_transcription(text):

    def replace_bracketed_text(match):
        label = re.sub(
            r"[\s　・。.!！?？]",
            "",
            match.group(1),
        ).lower()

        if (
            label in NOISE_LABELS
            or KATAKANA_SOUND_PATTERN.fullmatch(label)
        ):
            return " "

        return match.group(0)

    cleaned = BRACKETED_TEXT_PATTERN.sub(
        replace_bracketed_text,
        text,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    normalized = re.sub(
        r"[\s　、。,.!！?？…・ー〜～\-]",
        "",
        cleaned,
    ).lower()

    if (
        not normalized
        or normalized in NOISE_ONLY_TEXTS
    ):
        return ""

    return cleaned


def transcribe(wav_data=None):

    print()
    print(
        "Whisperで認識中..."
    )


    start = time.perf_counter()


    response = None

    try:
        if wav_data is None:
            with open(INPUT_WAV, "rb") as audio_file:
                wav_data = audio_file.read()

        response = WHISPER_SESSION.post(
            WHISPER_URL,
            files={
                "file": (
                    "input.wav",
                    wav_data,
                    "audio/wav",
                )
            },
            data=WHISPER_REQUEST_DATA,
            timeout=60,
        )


        response.raise_for_status()


        result = response.json()


        raw_text = (
            result.get(
                "text",
                ""
            ).strip()
        )

        text = clean_transcription(raw_text)

        if raw_text and not text:
            print()
            print(
                "ノイズ・効果音として無視: "
                f"{raw_text}"
            )
            return ""


    except requests.exceptions.ConnectionError:

        print(
            "Whisper Serverに"
            "接続できません。"
        )

        return ""


    except requests.exceptions.Timeout:

        print(
            "Whisper Serverが"
            "タイムアウトしました。"
        )

        return ""


    except requests.exceptions.RequestException as e:

        print(
            "Whisper Server通信エラー: "
            f"{e}"
        )

        return ""


    except ValueError:

        print(
            "Whisper Serverから"
            "不正なJSONが返されました。"
        )

        if response is not None:
            print(response.text)

        return ""


    elapsed = (
        time.perf_counter()
        - start
    )


    print()
    print("--- 認識結果 ---")
    print(text)
    print()


    print(
        f"Whisper処理時間: "
        f"{elapsed:.2f} 秒"
    )


    return text


# ============================================================
# Gemini
# ============================================================

def extract_gemini_text(response):
    """SDKのresponse.textが空でも候補パーツから本文を回収する。"""

    raw_text = getattr(response, "text", None)

    if raw_text:
        return raw_text.strip()

    text_parts = []

    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)

        if content is None:
            continue

        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)

            if part_text:
                text_parts.append(part_text)

    return "".join(text_parts).strip()


def ask_gemini_and_speak(text, bridge):

    print()
    print("Geminiで考え中...")

    start = time.perf_counter()

    try:
        current_datetime = datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
        message = (
            f"【システム補足: 現在日時 {current_datetime}】\n"
            f"【ユーザー発話】{text}"
        )
        response = chat.send_message(message)
        answer = extract_gemini_text(response)


        elapsed = (
            time.perf_counter()
            - start
        )


        # それでも本文がない場合
        if not answer:

            print()
            print(
                "Geminiから本文のない"
                "レスポンスが返されました。"
            )

            # 原因調査用
            for candidate in (
                response.candidates or []
            ):

                finish_reason = getattr(
                    candidate,
                    "finish_reason",
                    None
                )

                print(
                    "finish_reason:",
                    finish_reason
                )

            print(
                f"Gemini処理時間: "
                f"{elapsed:.2f} 秒"
            )

            return ""


        print()
        print("AI:")
        print(answer)
        print()

        print(
            f"Gemini処理時間: "
            f"{elapsed:.2f} 秒"
        )


        # 分割によるジョブ切替を避け、全文を1回で合成する。
        bridge.speak(answer)


        return answer


    except Exception as e:

        print()
        print(
            f"Geminiエラー: {e}"
        )

        return ""


# ============================================================
# VOICEPEAK Bridge
# ============================================================

class VoicepeakBridge:

    def __init__(self):

        self.process = None

        self.start()


    def start(self):

        last_error = None
        retry_delays = (
            0.0,
            *VOICEPEAK_BRIDGE_START_RETRY_SECONDS,
        )

        for attempt, wait_seconds in enumerate(
            retry_delays,
            start=1,
        ):

            if wait_seconds > 0:

                print()
                print(
                    "VOICEPEAK Bridgeの起動を"
                    "再試行します "
                    f"({attempt}/{len(retry_delays)})"
                )

                time.sleep(wait_seconds)

            try:
                self._start_once()
                return

            except RuntimeError as e:
                last_error = e
                self._force_stop()

                if attempt >= len(retry_delays):
                    break

                print(
                    "VOICEPEAK Bridge起動待機: "
                    f"{e}"
                )

        raise RuntimeError(
            "VOICEPEAK Bridgeを起動できませんでした。"
            f"最後のエラー: {last_error}"
        ) from last_error


    def _start_once(self):

        print()
        print("VOICEPEAK Bridgeを起動中...")


        if not os.path.isfile(VOICEPEAK_BRIDGE):

            raise FileNotFoundError(
                "VOICEPEAK Bridgeが見つかりません:\n"
                f"{VOICEPEAK_BRIDGE}"
            )


        self.process = subprocess.Popen(
            [VOICEPEAK_BRIDGE],

            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,

            text=True,
            encoding="utf-8",
            errors="replace",

            bufsize=1,

            cwd=os.path.dirname(
                VOICEPEAK_BRIDGE
            )
        )
    


        # READYより前にログが出ても待つ
        while True:

            line = self.process.stdout.readline()

            if line == "":

                try:
                    return_code = self.process.wait(
                        timeout=1
                    )

                except subprocess.TimeoutExpired:
                    return_code = self.process.poll()

                self._force_stop()

                raise RuntimeError(
                    "VOICEPEAK BridgeがREADYを返す前に"
                    "終了しました。"
                    f" 終了コード: {return_code}"
                )


            line = line.strip()


            if not line:
                continue


            if line == "READY":

                print("VOICEPEAK Bridge準備完了")
                return


            print(f"[Bridge] {line}")


    def _force_stop(self):

        if self.process is None:
            return


        if self.process.poll() is None:

            try:

                self.process.terminate()

                self.process.wait(
                    timeout=2
                )

            except Exception:

                try:
                    self.process.kill()

                except Exception:
                    pass


        self.process = None


    def restart(
        self,
        wait_seconds=2.0
    ):

        print()
        print(
            "VOICEPEAK Bridgeを"
            "再接続します..."
        )


        self._force_stop()


        # VOICEPEAK本体も確認
        ensure_voicepeak_running()


        time.sleep(
            wait_seconds
        )


        self.start()


    def _send_speak(
        self,
        text
    ):

        if (
            self.process is None
            or self.process.poll() is not None
        ):

            raise RuntimeError(
                "VOICEPEAK Bridgeが"
                "終了しています。"
            )


        # 日本語を安全にstdinへ渡すため
        # Base64に変換
        encoded = (
            base64
            .b64encode(
                text.encode("utf-8")
            )
            .decode("ascii")
        )


        start = time.perf_counter()


        self.process.stdin.write(
            f"SPEAK {encoded}\n"
        )

        self.process.stdin.flush()


        while True:

            line = (
                self.process
                .stdout
                .readline()
            )


            if line == "":

                raise ConnectionError(
                    "VOICEPEAK Bridgeとの"
                    "接続が切れました。"
                )


            line = line.strip()


            # VoicepeakRuntime版Bridgeでは
            # キュー投入成功時に
            #
            # QUEUED|JobId
            #
            # が返る
            if line.startswith("QUEUED|"):

                parts = line.split(
                    "|",
                    1
                )


                job_id = (
                    parts[1]
                    if len(parts) >= 2
                    else "unknown"
                )


                elapsed = (
                    time.perf_counter()
                    - start
                )


                print(
                    "VOICEPEAKキュー投入: "
                    f"{elapsed:.2f} 秒 "
                    f"(job={job_id})"
                )


                return


            if line.startswith("ERROR|"):

                raise RuntimeError(
                    line
                )


    def speak(
        self,
        text
    ):

        print()
        print(
            "VOICEPEAKへ送信中..."
        )


        # 通常送信 + 接続異常時の再接続
        max_attempts = 4


        reconnect_waits = [
            0.0,
            2.0,
            4.0,
            6.0,
        ]


        last_error = None


        for attempt in range(
            max_attempts
        ):

            if attempt > 0:

                wait_seconds = (
                    reconnect_waits[
                        attempt
                    ]
                )


                print()
                print(
                    "VOICEPEAK再接続を"
                    "試します "
                    f"({attempt}/"
                    f"{max_attempts - 1})"
                )


                self.restart(
                    wait_seconds
                )


            try:

                if attempt > 0:

                    print(
                        "VOICEPEAKへ"
                        "再送信中..."
                    )


                self._send_speak(
                    text
                )


                return


            except ConnectionError as e:

                last_error = e


                print(
                    "Bridge接続エラー: "
                    f"{e}"
                )


                continue


            except RuntimeError as e:

                error_text = str(e)

                last_error = e


                reconnect_errors = (
                    "ProcessLost",
                    "PrepareFailed",
                    "Exception",
                    "modifier_guard",
                    "target_pid_zero",
                )


                reconnect_needed = any(
                    keyword in error_text

                    for keyword
                    in reconnect_errors
                )


                if reconnect_needed:

                    print(
                        "VOICEPEAK接続異常を検出: "
                        f"{error_text}"
                    )

                    continue


                # その他のエラーはそのまま返す
                raise


        raise RuntimeError(
            "VOICEPEAKへの再接続に"
            "失敗しました。"
            f"最後のエラー: {last_error}"
        )


    def close(self):

        if self.process is None:
            return


        if self.process.poll() is None:

            try:

                self.process.stdin.write(
                    "QUIT\n"
                )

                self.process.stdin.flush()


                self.process.wait(
                    timeout=3
                )


            except Exception:

                self._force_stop()


        self.process = None


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":

    whisper_server = None
    bridge = None


    try:

        print(
            "================================"
        )

        print(
            " Gemini Voice Chat"
        )

        print(
            "================================"
        )

        # ----------------------------------------------------
        # アプリ設定
        # ----------------------------------------------------

        initialize_app_settings()


        # ----------------------------------------------------
        # Gemini設定・会話履歴
        # ----------------------------------------------------

        chat = initialize_gemini_chat()


        # ----------------------------------------------------
        # Whisper Server
        # ----------------------------------------------------

        whisper_server = (
            WhisperServer()
        )


        # ----------------------------------------------------
        # VOICEPEAK本体
        # ----------------------------------------------------

        ensure_voicepeak_running()


        # ----------------------------------------------------
        # VOICEPEAK Bridge
        # ----------------------------------------------------

        bridge = VoicepeakBridge()


        # ----------------------------------------------------
        # 会話ループ
        # ----------------------------------------------------

        run_continuous_conversation(bridge)


    except KeyboardInterrupt:

        print()
        print(
            "終了しました。"
        )


    except Exception as e:

        print()
        print("エラー:")
        print(e)


    finally:

        # Bridge終了
        if bridge is not None:

            bridge.close()


        # Python自身が起動した
        # Whisper Serverだけ終了
        if whisper_server is not None:

            whisper_server.close()
