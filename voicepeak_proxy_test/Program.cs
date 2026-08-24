using System;
using System.Text;
using VoicepeakProxyCore;

class Program
{
    static string Clean(string text)
    {
        if (text == null)
        {
            return "";
        }

        return text
            .Replace("\r", " ")
            .Replace("\n", " ")
            .Replace("|", "/");
    }


    static void Main()
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        Console.InputEncoding = Encoding.UTF8;

        var config = new AppConfig();

        // 発話開始検知には少し余裕を持たせる
        config.Audio.StartConfirmTimeoutMs = 5000;

        // ====================================================
        // VoicepeakProxyの常駐ランタイムを一度だけ起動
        // ====================================================

        using (var runtime = VoicepeakRuntime.Start(config))
        {
            Console.WriteLine("READY");
            Console.Out.Flush();

            while (true)
            {
                string line = Console.ReadLine();

                if (line == null)
                {
                    break;
                }

                if (line == "QUIT")
                {
                    break;
                }

                if (!line.StartsWith("SPEAK "))
                {
                    Console.WriteLine("ERROR|InvalidCommand");
                    Console.Out.Flush();
                    continue;
                }

                try
                {
                    string encoded = line.Substring(6);

                    byte[] bytes =
                        Convert.FromBase64String(encoded);

                    string text =
                        Encoding.UTF8.GetString(bytes);

                    if (string.IsNullOrWhiteSpace(text))
                    {
                        Console.WriteLine("ERROR|EmptyText");
                        Console.Out.Flush();
                        continue;
                    }

                    // ============================================
                    // 常駐Runtimeのキューに発話要求を追加
                    // ============================================

                    EnqueueResult result =
                        runtime.Enqueue(
                            new SpeakRequest
                            {
                                Text = text,

                                // 今喋っているものの後ろに追加
                                Mode = EnqueueMode.Queue,

                                // Queueでは割り込みなし
                                Interrupt = false
                            }
                        );

                    if (result.Succeeded)
                    {
                        Console.WriteLine(
                            $"QUEUED|{result.JobId}"
                        );
                    }
                    else
                    {
                        Console.WriteLine(
                            $"ERROR|" +
                            $"{result.Status}|" +
                            $"{Clean(result.ErrorMessage)}"
                        );
                    }

                    Console.Out.Flush();
                }
                catch (Exception ex)
                {
                    Console.WriteLine(
                        $"ERROR|Exception|{Clean(ex.Message)}"
                    );

                    Console.Out.Flush();
                }
            }

            runtime.Stop();
        }
    }
}