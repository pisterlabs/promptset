import os
import json
import asyncio
import subprocess
from typing import Tuple
from pathlib import Path

import openai

os.environ["PATH"] = f'{os.environ["PATH"]}:/opt/homebrew/Caskroom/miniforge/base/bin'


async def exec_command(command) -> Tuple[str, str, int]:
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode


async def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    if Path(audio_path).exists():
        Path(audio_path).unlink()

    command = f'ffmpeg -i "{video_path}" -vn "{audio_path}"'

    stdout, stderr, return_code = await exec_command(command)

    if return_code != 0:
        raise Exception(f"Failed to extract audio: {stderr}")


async def get_bilibili_video_title(url: str) -> str:
    command = f'you-get --json "{url}"'

    stdout, stderr, return_code = await exec_command(command)

    if stderr or return_code != 0:
        raise Exception(f"Failed to get bilibili video title: {stderr}")

    return json.loads(stdout)["title"]


async def download_bilibili_video(url: str, output_dir: str) -> None:
    video_title = await get_bilibili_video_title(url)

    command = f'you-get --output-dir {output_dir} "{url}"'

    stdout, stderr, return_code = await exec_command(command)

    if return_code != 0:
        raise Exception(f"Failed to download video: {stderr}")


async def convert_audio_to_text(audio_file_path: str) -> str:
    # url = "https://api.openai.com/v1/audio/transcriptions"
    #
    # headers = {
    #     'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}',
    #     'Content-Type': 'multipart/form-data'
    # }
    #
    # data = {
    #     'model': 'whisper-1',
    # }
    #
    # with open(audio_file_path, 'rb') as f:
    #     files = [
    #         ('file', (f.name, f, "application/octet-stream"))
    #     ]
    #
    #     async with httpx.AsyncClient() as client:
    #         response = await client.post(
    #             url,
    #             headers=headers,
    #             params=data,
    #             files=files,
    #             timeout=500
    #         )
    #         return response.json()["text"]

    audio_file = open(audio_file_path, "rb")
    transcript = await openai.Audio.atranscribe("whisper-1", audio_file)
    return transcript.text


async def convert_video_audio_to_text(url: str, output_dir: str, debug: bool = False) -> None:
    if debug:
        print("step 1 - get video title")
    title = await get_bilibili_video_title(url)

    print(f"generating video copywriting from {title}")

    if debug:
        print("step 2 - download video into local")
    await download_bilibili_video(url, output_dir)

    if debug:
        print("step 3 - extract audio from video")
    await extract_audio_from_video(
        str(Path(output_dir) / f"{title}.mp4"),
        str(Path(output_dir) / f"{title}.mp3"),
    )

    if debug:
        print("step 4 - convert audio to text")
    text = await convert_audio_to_text(str(Path(output_dir) / f"{title}.mp3"))
    print(text)


async def main():
    # url = "https://www.bilibili.com/video/BV1Mh4y127bX/?spm_id_from=444.41.list.card_archive.click&vd_source=e64200d4ea932bdbc9eb93c54976d3cf"
    url = "https://www.bilibili.com/video/BV1n14y1m7F2/?spm_id_from=333.337.search-card.all.click&vd_source=e64200d4ea932bdbc9eb93c54976d3cf"
    output_dir = "/Users/luominzhi/Scratch/audio/"

    await asyncio.gather(
        convert_video_audio_to_text(url, output_dir),
        # extract_audio_from_video(video_path, audio_path),
    )


if __name__ == "__main__":
    asyncio.run(main())
