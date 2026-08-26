import json
import io
import math
import tempfile
import unittest
import wave

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import voice_chat_gemini as app


class AudioUtilityTests(unittest.TestCase):
    def test_calculate_rms(self):
        chunk = np.array([[3], [4]], dtype=np.int16)

        self.assertAlmostEqual(
            app.calculate_rms(chunk),
            math.sqrt(12.5),
            places=5,
        )

    def test_encode_pcm16_wav(self):
        audio = np.array([0, 1000, -1000], dtype=np.int16)
        wav_data = app.encode_pcm16_wav(audio)

        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), app.CHANNELS)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), app.SAMPLE_RATE)
            self.assertEqual(wav_file.getnframes(), audio.size)
            self.assertEqual(wav_file.readframes(audio.size), audio.tobytes())


class WhisperServerTests(unittest.TestCase):
    def test_warmup_runs_configured_number_of_requests(self):
        response = SimpleNamespace(raise_for_status=lambda: None)

        with (
            patch.object(app.WhisperServer, "start"),
            patch.object(
                app.WHISPER_SESSION,
                "post",
                return_value=response,
            ) as post,
        ):
            app.WhisperServer()

        self.assertEqual(
            post.call_count,
            app.WHISPER_WARMUP_REQUESTS,
        )

        wav_data = post.call_args.kwargs["files"]["file"][1]

        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            self.assertEqual(
                wav_file.getnframes(),
                round(
                    app.SAMPLE_RATE
                    * app.WHISPER_WARMUP_AUDIO_SECONDS
                ),
            )


class TextUtilityTests(unittest.TestCase):
    def test_clean_transcription_removes_noise_only_text(self):
        self.assertEqual(app.clean_transcription("（音楽）"), "")
        self.assertEqual(app.clean_transcription("ご視聴ありがとうございました。"), "")

    def test_clean_transcription_keeps_spoken_text(self):
        self.assertEqual(
            app.clean_transcription("今日は晴れ（雑音）です"),
            "今日は晴れ です",
        )

    def test_extract_gemini_text_falls_back_to_parts(self):
        response = SimpleNamespace(
            text=None,
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="前半"),
                            SimpleNamespace(text="後半"),
                        ]
                    )
                )
            ],
        )

        self.assertEqual(app.extract_gemini_text(response), "前半後半")

    def test_sentence_buffer_flushes_completed_and_remaining_text(self):
        completed, remainder = app.take_voicepeak_sentences(
            "一文目です。『二文目です！』最後です",
            flush=True,
        )

        self.assertEqual(
            completed,
            [
                "一文目です。",
                "『二文目です！』",
                "最後です",
            ],
        )
        self.assertEqual(remainder, "")

    def test_sentence_buffer_waits_for_possible_closing_quote(self):
        completed, remainder = app.take_voicepeak_sentences(
            "最初です。",
            flush=False,
        )

        self.assertEqual(completed, [])
        self.assertEqual(remainder, "最初です。")

        completed, remainder = app.take_voicepeak_sentences(
            remainder + "」次です",
            flush=False,
        )

        self.assertEqual(completed, ["最初です。」"])
        self.assertEqual(remainder, "次です")

    def test_gemini_stream_queues_first_sentence_before_completion(self):
        spoken = []
        observed_during_stream = []

        def response_stream(message):
            yield SimpleNamespace(
                text="最初です。次",
                candidates=[],
            )
            observed_during_stream.append(list(spoken))
            yield SimpleNamespace(
                text="です。",
                candidates=[],
            )

        fake_chat = SimpleNamespace(
            send_message_stream=response_stream
        )
        bridge = SimpleNamespace(speak=spoken.append)

        with patch.object(app, "chat", fake_chat):
            answer = app.ask_gemini_and_speak(
                "テスト",
                bridge,
            )

        self.assertEqual(
            observed_during_stream,
            [["最初です。"]],
        )
        self.assertEqual(
            spoken,
            ["最初です。", "次です。"],
        )
        self.assertEqual(answer, "最初です。次です。")


class ConversationHistoryTests(unittest.TestCase):
    def test_only_requested_recent_turns_are_loaded(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            history_path = Path(temp_directory) / "history.jsonl"

            with history_path.open("w", encoding="utf-8") as history_file:
                for turn in range(5):
                    for role, text in (
                        ("user", f"質問{turn}"),
                        ("assistant", f"回答{turn}"),
                    ):
                        history_file.write(
                            json.dumps(
                                {"role": role, "text": text},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            with patch.object(app, "CONVERSATION_LOG", str(history_path)):
                history = app.load_conversation_history(2)

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].parts[0].text, "質問3")
        self.assertEqual(history[-1].parts[0].text, "回答4")


class GeminiClientTests(unittest.TestCase):
    def test_client_uses_configured_timeout_without_retries(self):
        settings = app.DEFAULT_GEMINI_SETTINGS | {
            "request_timeout_seconds": 12.5,
        }
        fake_chats = SimpleNamespace(
            create=lambda **kwargs: kwargs
        )
        fake_client = SimpleNamespace(chats=fake_chats)

        with (
            patch.object(
                app,
                "load_gemini_settings",
                return_value=settings,
            ),
            patch.object(
                app,
                "load_system_instruction",
                return_value="テスト指示",
            ),
            patch.object(
                app,
                "load_conversation_history",
                return_value=[],
            ),
            patch.object(
                app.genai,
                "Client",
                return_value=fake_client,
            ) as client_factory,
        ):
            chat_config = app.initialize_gemini_chat()

        http_options = client_factory.call_args.kwargs[
            "http_options"
        ]

        self.assertEqual(http_options.timeout, 12500)
        self.assertEqual(
            http_options.retry_options.attempts,
            1,
        )
        self.assertEqual(
            chat_config["model"],
            "gemini-3.1-flash-lite",
        )


if __name__ == "__main__":
    unittest.main()
