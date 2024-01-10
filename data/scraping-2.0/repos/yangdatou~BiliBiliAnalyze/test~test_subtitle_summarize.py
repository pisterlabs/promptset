import sys, os
from bilibili import BiliBiliVideo
from openai_api import SummarizeSubtitle

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", None)
BILIBILI_COOKIE = os.environ.get("BILIBILI_COOKIE", None)

def test_subtitle_summarize(bvid):
    bv = BiliBiliVideo(bvid)

    # The subtitles must be accessible with login
    bv.bilibili_cookie = BILIBILI_COOKIE

    bv.dump_info()
    subtitle_list = bv.dump_subtitle()

    sr = SummarizeSubtitle()
    sr.api_key    = OPENAI_API_KEY
    max_tokens    = 1000
    sr.kwargs = {
        "temperature": 0.5,
        "max_tokens": max_tokens,
    }

    print("\n### Subtitle Summarize")

    s = "".join(subtitle_list)
    batch_size = len(s) // (len(s) // max_tokens - 1)
    print("`len(s) = %d; " %(len(s)))
    print("batch_size = %d`" %(batch_size))

    s = ""
    for isubtitle, subtitle in enumerate(subtitle_list):
        s += ", " + subtitle if len(s) > 0 else subtitle

        if len(s) > batch_size or isubtitle == len(subtitle_list) - 1:
            c = ["以下是这个视频的字幕的一部分：", s]
            m = sr.run(c)
            print("%s" %(m["choices"][0]["message"]["content"]))
            s = ""

if __name__ == "__main__":
    test_subtitle_summarize("BV1ij411g7hN")
    