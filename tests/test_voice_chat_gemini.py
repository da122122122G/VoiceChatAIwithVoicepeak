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


if __name__ == "__main__":
    unittest.main()
