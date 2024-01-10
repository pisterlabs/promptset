{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNa7+47HyRF4YtMDhZr2WdB",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/aditya-1442/Interview_AI_Assistant_LLM/blob/main/AI_Interview_Assistant.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "! pip install streamlit -q"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tRG8-J3huLSX",
        "outputId": "bf3a42d9-52e8-4e7a-be1c-a5ccf5b6a593"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.4/8.4 MB\u001b[0m \u001b[31m29.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m190.6/190.6 kB\u001b[0m \u001b[31m10.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.8/4.8 MB\u001b[0m \u001b[31m41.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m82.1/82.1 kB\u001b[0m \u001b[31m4.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m6.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install langchain\n",
        "!pip install OpenAI"
      ],
      "metadata": {
        "id": "jtLYtW5vvUZ5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain import PromptTemplate\n",
        "from langchain.chat_models import ChatOpenAI\n",
        "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
        "from langchain.chains.summarize import load_summarize_chain\n",
        "from langchain.prompts import PromptTemplate\n",
        "\n",
        "# Streamlit\n",
        "import streamlit as st\n",
        "import requests\n",
        "from bs4 import BeautifulSoup\n",
        "!pip install markdownify\n",
        "from markdownify import markdownify as md\n"
      ],
      "metadata": {
        "id": "n-RmLriLv3Zg"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install youtube-transcript-api"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n5GLpgRb27Wj",
        "outputId": "b5244269-7e18-4094-d663-5dddd9a1cfa4"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting youtube-transcript-api\n",
            "  Downloading youtube_transcript_api-0.6.1-py3-none-any.whl (24 kB)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from youtube-transcript-api) (2.31.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->youtube-transcript-api) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->youtube-transcript-api) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->youtube-transcript-api) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->youtube-transcript-api) (2023.11.17)\n",
            "Installing collected packages: youtube-transcript-api\n",
            "Successfully installed youtube-transcript-api-0.6.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain.document_loaders import YoutubeLoader"
      ],
      "metadata": {
        "id": "8CiZgT2n3B9u"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os"
      ],
      "metadata": {
        "id": "lOeqdBc83G67"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install python-dotenv\n",
        "from dotenv import load_dotenv\n",
        "load_dotenv()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PuE1RvQJ3ed8",
        "outputId": "180497f2-cf3b-401c-9075-c24ba9f49285"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting python-dotenv\n",
            "  Downloading python_dotenv-1.0.0-py3-none-any.whl (19 kB)\n",
            "Installing collected packages: python-dotenv\n",
            "Successfully installed python-dotenv-1.0.0\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "False"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import userdata"
      ],
      "metadata": {
        "id": "_DpKh8zp4s6k"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Setting up the LLM\n"
      ],
      "metadata": {
        "id": "WoaJiQ3V4lzk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "OPENAI_API_KEY = userdata.get('open_api_key')\n",
        "def load_LLM(openai_api_key):\n",
        "    # Make sure your openai_api_key is set as an environment variable\n",
        "    llm = ChatOpenAI(temperature=.7, openai_api_key=OPENAI_API_KEY, max_tokens=2000, model_name='gpt-3.5-turbo')\n",
        "    return llm"
      ],
      "metadata": {
        "id": "xijFMmUc4om_"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### A function that will be called only if the environment's openai_api_key isn't set"
      ],
      "metadata": {
        "id": "sZx8Qmna5YmH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def get_openai_api_key():\n",
        "    input_text = st.text_input(label=\"OpenAI API Key (or set it as .env variable)\",  placeholder=\"Ex: sk-2twmA8tfCb8un4...\", key=\"openai_api_key_input\")\n",
        "    return input_text"
      ],
      "metadata": {
        "id": "Ivkvf7Mq5XtE"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Here we'll pull data from a website and return it's text"
      ],
      "metadata": {
        "id": "UsIrsnU55v4N"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def pull_from_website(url):\n",
        "    st.write(\"Getting webpages...\")\n",
        "    # Doing a try in case it doesn't work\n",
        "    try:\n",
        "        response = requests.get(url)\n",
        "    except:\n",
        "        # In case it doesn't work\n",
        "        print (\"Whoops, error\")\n",
        "        return\n",
        "\n",
        "    # Put your response in a beautiful soup\n",
        "    soup = BeautifulSoup(response.text, 'html.parser')\n",
        "\n",
        "    # Get your text\n",
        "    text = soup.get_text()\n",
        "\n",
        "    # Convert your html to markdown. This reduces tokens and noise\n",
        "    text = md(text)\n",
        "\n",
        "    return text"
      ],
      "metadata": {
        "id": "JllE2Ltr5pOc"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Pulling data from YouTube in text form"
      ],
      "metadata": {
        "id": "GEqs4pZs6CGG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def get_video_transcripts(url):\n",
        "    st.write(\"Getting YouTube Videos...\")\n",
        "    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)\n",
        "    documents = loader.load()\n",
        "    transcript = ' '.join([doc.page_content for doc in documents])\n",
        "    return transcript"
      ],
      "metadata": {
        "id": "pDAuXha_53LJ"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Function to change our long text about a person into documents"
      ],
      "metadata": {
        "id": "bUkDu4Ao7d0f"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def split_text(user_information):\n",
        "    # First we make our text splitter\n",
        "    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)\n",
        "\n",
        "    # Then we split our user information into different documents\n",
        "    docs = text_splitter.create_documents([user_information])\n",
        "\n",
        "    return docs"
      ],
      "metadata": {
        "id": "rfLBu7Go6Hb0"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "response_types = {\n",
        "    'Interview Questions' : \"\"\"\n",
        "        Your goal is to generate interview questions that we can ask them\n",
        "        Please respond with list of a few interview questions based on the topics above\n",
        "    \"\"\",\n",
        "    '1-Page Summary' : \"\"\"\n",
        "        Your goal is to generate a 1 page summary about them\n",
        "        Please respond with a few short paragraphs that would prepare someone to talk to this person\n",
        "    \"\"\"\n",
        "}\n",
        "\n",
        "map_prompt = \"\"\"You are a helpful AI bot that aids a user in research.\n",
        "Below is information about a person named {persons_name}.\n",
        "Information will include tweets, interview transcripts, and blog posts about {persons_name}\n",
        "Use specifics from the research when possible\n",
        "\n",
        "{response_type}\n",
        "\n",
        "% START OF INFORMATION ABOUT {persons_name}:\n",
        "{text}\n",
        "% END OF INFORMATION ABOUT {persons_name}:\n",
        "\n",
        "YOUR RESPONSE:\"\"\"\n",
        "map_prompt_template = PromptTemplate(template=map_prompt, input_variables=[\"text\", \"persons_name\", \"response_type\"])\n",
        "\n",
        "combine_prompt = \"\"\"\n",
        "You are a helpful AI bot that aids a user in research.\n",
        "You will be given information about {persons_name}.\n",
        "Do not make anything up, only use information which is in the person's context\n",
        "\n",
        "{response_type}\n",
        "% PERSON CONTEXT\n",
        "{text}\n",
        "\n",
        "% YOUR RESPONSE:\n",
        "\"\"\"\n",
        "combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=[\"text\", \"persons_name\", \"response_type\"])\n",
        "\n",
        "# Start Of Streamlit page\n",
        "st.set_page_config(page_title=\"LLM Assisted Interview Prep\", page_icon=\":robot:\")\n",
        "\n",
        "# Start Top Information\n",
        "st.header(\"LLM Assisted Interview Prep\")\n",
        "\n",
        "col1, col2 = st.columns(2)\n",
        "\n",
        "with col1:\n",
        "    st.markdown(\"Have an interview coming up? I bet they are  YouTube or the web. This tool is meant to help you generate \\\n",
        "                interview questions based off of topics they've recently tweeted or talked about.\\\n",
        "                \\n\\nThis tool is powered by [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/#) [markdownify](https://pypi.org/project/markdownify/) [Tweepy](https://docs.tweepy.org/en/stable/api.html), [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \\\n",
        "                [@JbsAditya](https://twitter.com/JbsAditya). \\n\\n \")\n",
        "\n",
        "\n",
        "\n",
        "st.markdown(\"## :older_man: Larry The LLM Researcher\")\n",
        "# Output type selection by the user\n",
        "output_type = st.radio(\n",
        "    \"Output Type:\",\n",
        "    ('Interview Questions', '1-Page Summary'))"
      ],
      "metadata": {
        "id": "x7932GfPBPkJ"
      },
      "execution_count": 69,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "person_name = st.text_input(label=\"Person's Name\",  placeholder=\"Ex: Elad Gil\", key=\"persons_name\")\n",
        "youtube_videos = st.text_input(label=\"YouTube URLs (Use , to seperate videos)\",  placeholder=\"Ex: https://www.youtube.com/watch?v=c_hO_fjmMnk, https://www.youtube.com/watch?v=c_hO_fjmMnk\", key=\"youtube_user_input\")\n",
        "webpages = st.text_input(label=\"Web Page URLs (Use , to seperate urls. Must include https://)\",  placeholder=\"https://eladgil.com/\", key=\"webpage_user_input\")\n"
      ],
      "metadata": {
        "id": "YggSe0KF8IZv"
      },
      "execution_count": 56,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "st.markdown(f\"### {output_type}:\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4agcAucG8cze",
        "outputId": "989c76da-877c-4d80-b178-b763f5112555"
      },
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Get URLs from a string\n",
        "def parse_urls(urls_string):\n",
        "    \"\"\"Split the string by comma and strip leading/trailing whitespaces from each URL.\"\"\"\n",
        "    return [url.strip() for url in urls_string.split(',')]"
      ],
      "metadata": {
        "id": "WDblyHWM8gf8"
      },
      "execution_count": 58,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def get_content_from_urls(urls, content_extractor):\n",
        "    \"\"\"Get contents from multiple urls using the provided content extractor function.\"\"\"\n",
        "    return \"\\n\".join(content_extractor(url) for url in urls)\n"
      ],
      "metadata": {
        "id": "Xoc8E17Q8khM"
      },
      "execution_count": 59,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "button_ind = st.button(\"*Generate Output*\", type='secondary', help=\"Click to generate output based on information\")"
      ],
      "metadata": {
        "id": "Fjhy9rOp8o4H"
      },
      "execution_count": 60,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "if button_ind:\n",
        "    if not (youtube_videos or webpages):\n",
        "        st.warning('Please provide links to parse', icon=\"⚠️\")\n",
        "        st.stop()\n",
        "\n",
        "    if not OPENAI_API_KEY:\n",
        "        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon=\"⚠️\")\n",
        "        st.stop()\n",
        "\n",
        "    if OPENAI_API_KEY == 'YourAPIKeyIfNotSet':\n",
        "        # If the openai key isn't set in the env, put a text box out there\n",
        "        OPENAI_API_KEY = get_openai_api_key()"
      ],
      "metadata": {
        "id": "lGQnKWmv8s7f"
      },
      "execution_count": 61,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        " video_text = get_content_from_urls(parse_urls(youtube_videos), get_video_transcripts) if youtube_videos else \"\"\n",
        " website_data = get_content_from_urls(parse_urls(webpages), pull_from_website) if webpages else \"\"\n",
        "\n",
        " user_information = \"\\n\".join([video_text, website_data])"
      ],
      "metadata": {
        "id": "s61O3hHK859F"
      },
      "execution_count": 62,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "_ZUOEpI_9oJG"
      },
      "execution_count": 62,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        " user_information_docs = split_text(user_information)"
      ],
      "metadata": {
        "id": "8XuBcs_99Sbx"
      },
      "execution_count": 63,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        " llm = load_LLM(openai_api_key=OPENAI_API_KEY)"
      ],
      "metadata": {
        "id": "tNV_wSNy9tiI"
      },
      "execution_count": 64,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        " chain = load_summarize_chain(llm,\n",
        "                                 chain_type=\"map_reduce\",\n",
        "                                 map_prompt=map_prompt_template,\n",
        "                                 combine_prompt=combine_prompt_template,\n",
        "                                 verbose=True\n",
        "                                 )\n",
        "\n",
        " st.write(\"Sending to LLM...\")"
      ],
      "metadata": {
        "id": "TnLhDOen9zcP"
      },
      "execution_count": 65,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install tiktoken"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Eeh9xarR-EfJ",
        "outputId": "c2b67a1c-355c-45a0-96f0-912901bd8ce7"
      },
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tiktoken in /usr/local/lib/python3.10/dist-packages (0.5.2)\n",
            "Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2023.6.3)\n",
            "Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from tiktoken) (2.31.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->tiktoken) (2023.11.17)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "output = chain({\"input_documents\": user_information_docs,\n",
        "                    \"persons_name\": person_name,\n",
        "                    \"response_type\" : response_types[output_type]\n",
        "                    })\n",
        "st.markdown(f\"#### Output:\")\n",
        "st.write(output['output_text'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UGJC7HK398R2",
        "outputId": "9803f7b6-45fd-4bfe-d58b-6efe5105b27e"
      },
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "\n",
            "\u001b[1m> Entering new MapReduceDocumentsChain chain...\u001b[0m\n",
            "\n",
            "\n",
            "\u001b[1m> Entering new LLMChain chain...\u001b[0m\n",
            "\n",
            "\u001b[1m> Finished chain.\u001b[0m\n",
            "\n",
            "\n",
            "\u001b[1m> Entering new LLMChain chain...\u001b[0m\n",
            "Prompt after formatting:\n",
            "\u001b[32;1m\u001b[1;3m\n",
            "You are a helpful AI bot that aids a user in research.\n",
            "You will be given information about .\n",
            "Do not make anything up, only use information which is in the person's context\n",
            "\n",
            "\n",
            "        Your goal is to generate interview questions that we can ask them\n",
            "        Please respond with list of a few interview questions based on the topics above\n",
            "    \n",
            "% PERSON CONTEXT\n",
            "\n",
            "\n",
            "% YOUR RESPONSE:\n",
            "\u001b[0m\n",
            "\n",
            "\u001b[1m> Finished chain.\u001b[0m\n",
            "\n",
            "\u001b[1m> Finished chain.\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(output['output_text'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "85TTV5p6_ifd",
        "outputId": "ef8b86b0-ecd3-4dec-bd0a-58f9f80a2260"
      },
      "execution_count": 68,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1. Can you explain the main objectives and purpose of your research project?\n",
            "2. What specific research methodologies or techniques did you use in your study?\n",
            "3. What were the key findings or results of your research? How do these findings contribute to the existing knowledge in the field?\n",
            "4. How did you ensure the validity and reliability of your research data and findings?\n",
            "5. Were there any unexpected challenges or obstacles that you encountered during your research? How did you overcome them?\n",
            "6. Can you discuss any limitations or potential biases in your research design or data collection process?\n",
            "7. Did your research project involve any ethical considerations? If so, how did you address them?\n",
            "8. How do you plan to disseminate or publish your research findings? Do you have any specific plans for future research in this area?\n",
            "9. Can you elaborate on the significance or potential impact of your research in practical or theoretical terms?\n",
            "10. What are the potential applications or implications of your research findings?\n"
          ]
        }
      ]
    }
  ]
}