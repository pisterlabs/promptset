from openai import OpenAI
import pickle

client = OpenAI()
# defaults to getting the key using os.environ.get("OPENAI_API_KEY")
# if you saved the key under a different environment variable name, you can do something like:
# client = OpenAI(
#   api_key=os.environ.get("CUSTOM_ENV_NAME"),
# )
client = OpenAI()


def get_metadata(filename_list):
    filenames = ','.join(filename_list)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={ "type": "json_object" },
        # messages=[
        #     {"role": "system", "content": "You are a helpful assistant, skilled in extracting metadata from video filename and return json data."},
        #     {
        #         "role": "user",
        #         "content": "extract [[Sakurato] Kage no Jitsuryokusha ni Naritakute! S2 [01][HEVC-10bit 1080P AAC][CHS&CHT].mk,]"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": '''[
        #   "[Sakurato] Kage no Jitsuryokusha ni Naritakute! S2 [01][HEVC-10bit 1080P AAC][CHS&CHT].mkv": {
        #       "title": "Kage no Jitsuryokusha ni Naritakute!",
        #       "season": 2,
        #       "episode": 01
        #       "video_codec": "HEVC-10bit",
        #       "screen_size": "1080p",
        #       "audio_codec": "AAC",
        #       "release_group": "Sakurato",
        #       "container": "mkv",
        #       "type": "episode"
        #   }
        # ]'''
        #     },
        #     {
        #         "role": "user",
        #         "content": f"extract metadata filename list [{filenames}]"
        #     }
        # ]
        messages=[
            {
                "role": "user",
                "content": f"I give you a filename list, you extract video metadata from filename and return json including these keys: title, season, episode, video_codec, screen_size, audio_codec, release_group, container, type(episode or movie). This is the filename list: [{filenames}]"
            }
        ]
    )

    # print(completion)
    with open("completion.pkl", "wb") as f:
        pickle.dump(completion, f)

    finish_reason = completion.choices[0].finish_reason
    if finish_reason != "stop":
        print(f"WARNING: {finish_reason}")
    print(completion.choices[0].message)

L = [
    "[Sakurato] Kage no Jitsuryokusha ni Naritakute! S2 [01][HEVC-10bit 1080P AAC][CHS&CHT].mkv",
    "[Sakurato] Spy x Family Season 2 [04][HEVC-10bit 1080p AAC][CHS&CHT].mkv",
    "[BeanSub&FZSD][Kimetsu_no_Yaiba][49][GB][1080P][x264_AAC].mp4",
    "[Nekomoe kissaten&VCB-Studio] takt op.Destiny [CM][Ma10p_1080p][x265_flac].mkv",
    "[Neon Genesis Evangelion][15][BDRIP][1440x1080][H264_FLACx2].mkv",
    "[Neon Genesis Evangelion][Vol.09][SP04][NCED][BDRIP][1440x1080][H264_FLAC].mkv",
]
get_metadata(L)

