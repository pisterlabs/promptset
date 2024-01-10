import json
from DataPusher.IFTTTPush import IFTTTPush
from DataReader.DropboxReader import DropboxReader
from DataWriter.OpenAIDataWriter import OpenAIDataWriter
from constant import PATH
import click
import time

api = {}
with open(f"{PATH}/key.json") as f:
    api = json.load(f)


whisper = OpenAIDataWriter(api["openai"])


@click.command()
@click.option("--filename", "-f", help="Filename to read")
@click.option(
    "--markdown", "-m", is_flag=True, help="Whether to output markdown", default=False
)
def main(filename, markdown=False):
    whisper.write_data(
        filename,
        "/Users/tianchenzhong/Downloads/mp3/语言转文字/",
        "whisper",
    )
    data = ""
    newfilename = ".".join(filename.split(".")[0:-1]) + ".txt"
    with open(newfilename) as f:
        data = f.read()
    new_data = whisper.improve_data(data)
    time.sleep(5)
    newfilename2 = ".".join(filename.split(".")[0:-1]) + "_edit.txt"
    with open(newfilename2, "w") as f:
        f.write(new_data)
    if not markdown:
        return
    new_data2 = whisper.markdown_data(new_data)
    newfilename3 = ".".join(filename.split(".")[0:-1]) + "_markdown.txt"
    with open(newfilename3, "w") as f:
        f.write(new_data2)


if __name__ == "__main__":
    main()
