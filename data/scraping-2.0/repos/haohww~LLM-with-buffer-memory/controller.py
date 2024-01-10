from buffer import *
from langchain.embeddings import OpenAIEmbeddings
from config import *

class Controller:
    def __init__(self, encoder='dpr'):
        self.buffer = Buffer()
        config = load_config()
        print(config)





if __name__ == "__main__":
    controller = Controller()
    # controller.add_sample('hellow world!')
    # controller.add_sample('how are you')
    # print(str(controller.buffer))

    # # Add sample key-value pairs to the buffer
    # for i in range(1, buffer_size + 3):
    #     sample = Sample(key=f"key_{i}", value=i)
    #     buffer.add_sample(sample)

    # # Display buffer contents
    # print("Buffer contents:")
    # for key, value in buffer.get_samples().items():
    #     print(f"Key: {key}, Value: {value}")

    # # Display buffer size
    # print(f"Buffer size: {len(buffer)}")
