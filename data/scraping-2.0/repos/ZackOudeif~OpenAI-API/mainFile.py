import openai
import math
import wget
import pathlib 
from base64 import b64decode
import io
from PIL import Image

openai.api_key = 'your key goes here'

iPrefix = "\n\nReview:\n"
iSuffix = "\n"
oPrefix = "\nEmail:\n"

review1 = "Nice socks, great colors, just enough support for wearing with a good pair of sneakers."
email1 = "Dear Customer, Thank you for buying socks form our store. Our socks come in a wide range of colors and types. We also sell sneakers on our platform, make sure to check out the huge collection online. Regards, Time Store"
review2 = "Love Deborah Harness's Trilogy! Didn't want the story to end and hope they turn this trilogy into a movie. I would love it if she wrote more books to continue this story!!!"
email2 = "Dear Customer, Thank you for purchasing the book from our store. We have many other books from Deborah Harness. We also have many movies with a similar theme. Make sure to check them out! Regards, Time Store"
review3 = "SO much quieter than other compressors. VERY quick as well. You will not regret this purchase."
email3 = "Dear Customer, Thank you for buying from our platform. Our compressors are among the best in the market. Make sure to check out our wide range of products. Regards, Time Store"
