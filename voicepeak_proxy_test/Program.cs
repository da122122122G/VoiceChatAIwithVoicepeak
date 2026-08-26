using System;
using System.Text;
using VoicepeakProxyCore;

internal static class Program
{
    private const string SpeakPrefix = "SPEAK ";
    private const string QuitCommand = "QUIT";


    private static string Clean(string text)
    {
        return (text ?? string.Empty)
            .Replace("\r", " ")
            .Replace("\n", " ")
            .Replace("|", "/");
    }


    private static void WriteResponse(string response)
    {
        Console.WriteLine(response);
        Console.Out.Flush();
    }


    private static void EnqueueSpeech(VoicepeakRuntime runtime, string encoded)
    {
        try
        {
            byte[] bytes = Convert.FromBase64String(encoded);
            string text = Encoding.UTF8.GetString(bytes);

            if (string.IsNullOrWhiteSpace(text))
            {
                WriteResponse("ERROR|EmptyText");
                return;
            }

            EnqueueResult result = runtime.Enqueue(
                new SpeakRequest
                {
                    Text = text,
                    Mode = EnqueueMode.Queue,
                    Interrupt = false
                }
            );

            if (result.Succeeded)
            {
                WriteResponse($"QUEUED|{result.JobId}");
                return;
            }

            WriteResponse(
                $"ERROR|{result.Status}|{Clean(result.ErrorMessage)}"
            );
        }
        catch (Exception ex)
        {
            WriteResponse($"ERROR|Exception|{Clean(ex.Message)}");
        }
    }


    private static void Main()
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        Console.InputEncoding = Encoding.UTF8;

        var config = new AppConfig();

        // 音声会話向けの低遅延設定。
        // 入力直後の再生失敗を避ける最低限の待機は残す。
        config.InputTiming.PostTypeWaitMinMs = 100;
        config.InputTiming.PostTypeWaitPerCharMs = 3;
        config.Ui.DelayBeforePlayShortcutMs = 5;
        config.Audio.PollIntervalMs = 20;
        config.Audio.StopConfirmMs = 120;

        // これは固定待機ではなく、再生開始を検出できない場合の上限。
        config.Audio.StartConfirmTimeoutMs = 5000;

        using (var runtime = VoicepeakRuntime.Start(config))
        {
            WriteResponse("READY");

            while (true)
            {
                string line = Console.ReadLine();

                if (line == null || string.Equals(
                    line,
                    QuitCommand,
                    StringComparison.Ordinal
                ))
                {
                    break;
                }

                if (!line.StartsWith(
                    SpeakPrefix,
                    StringComparison.Ordinal
                ))
                {
                    WriteResponse("ERROR|InvalidCommand");
                    continue;
                }

                EnqueueSpeech(runtime, line.Substring(SpeakPrefix.Length));
            }

            runtime.Stop();
        }
    }
}
