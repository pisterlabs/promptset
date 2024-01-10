import openai

sys = ""

u1 = "Can we talk about Reverse Prompt Engineering? By Reverse Prompt Engineering I mean creating a prompt from a given text."

a1 = """Yes, of course! Reverse Prompt Engineering is an interesting and useful technique that involves generating a prompt that can be used to generate text similar to a given input text. This technique is becoming increasingly popular in natural language processing and machine learning.

The idea behind Reverse Prompt Engineering is to create a prompt that provides sufficient information to the language model to generate text that is consistent with the input text. The prompt should be designed in such a way that it includes relevant information about the context, style, and structure of the input text.

Reverse Prompt Engineering can be used in a variety of applications, such as generating summaries, paraphrasing text, or generating responses to a given prompt. It can also be used to improve the performance of language models by providing them with more specific and relevant information about the task at hand."""

u2 = "Great. Can you give me a simple example of Reverse Prompt Engineering?"

a2 = """Sure, here's a simple example of Reverse Prompt Engineering:

Let's say we have the following input text: "The quick brown fox jumps over the lazy dog."

To generate a prompt that can be used to generate text similar to this input text, we can try the following:

Prompt: "Write a sentence about a quick animal jumping over a lazy animal."

This prompt captures the key elements of the input text: a quick animal (the fox) jumping over a lazy animal (the dog). It also provides some guidance on the style and structure of the text (i.e., a sentence).

Using this prompt, a language model can generate text that is similar in style and structure to the input text. For example, it might generate the following:

Output: "A swift gazelle leaped over a sleeping lion."

In this output, we see a similar structure and style to the input text, with a quick animal (the gazelle) jumping over a lazy animal (the sleeping lion).

Of course, the quality and relevance of the output will depend on the specific language model and the prompt used, but this example should give you a basic idea of how Reverse Prompt Engineering works."""

u3 = "Good. Write a short explanation on how dog training works, and Reverse Prompt Engineer the explanation."

a3 = """Sure, here's a short explanation of how dog training works:

Dog training is the process of teaching a dog to behave in a certain way in response to specific commands or stimuli. It typically involves using positive reinforcement or negative reinforcement techniques to encourage desirable behaviors and discourage undesirable ones. Positive reinforcement involves rewarding a dog for exhibiting the desired behavior, while negative reinforcement involves removing a negative stimulus when the desired behavior is exhibited. Consistency, patience, and repetition are key elements of successful dog training.

Now, let's Reverse Prompt Engineer this explanation to generate a prompt that can be used to generate text similar to this explanation:

Prompt: "Explain the process of teaching a dog to behave in a certain way and the techniques involved in it."

This prompt captures the key elements of the explanation: the process of teaching a dog to behave in a certain way, the use of positive and negative reinforcement techniques, and the importance of consistency, patience, and repetition. It also provides some guidance on the style and structure of the text (i.e., an explanation).

Using this prompt, a language model can generate text that is similar in style and structure to the explanation. For example, it might generate the following:

Output: "Teaching a dog to behave in a certain way involves using positive and negative reinforcement techniques. Positive reinforcement involves rewarding the dog for exhibiting desirable behaviors, while negative reinforcement involves removing a negative stimulus when the desired behavior is exhibited. Consistency, patience, and repetition are important elements of successful dog training."

In this output, we see a similar structure and style to the original explanation, with a clear explanation of the process and techniques involved in dog training."""

u4 = "Great. Can you create a very technical reverse prompt engineering template?"

a4 = 'Prompt: "Using [insert technical term], explain [insert concept or task] and the underlying principles that govern it."'

u5 = """Can you reverse engineer the following {product description), capture the writing style and the length of the text:
product description = Personalized Spatial Audio with dynamic head tracking places sound all around you.Note : If the size of the earbud tips does not match the size of your ear canals or the headset is not worn properly in your ears, you may not obtain the correct sound qualities or call performance. Change the earbud tips to ones that fit more snugly in your ear
Single fit
Force sensor lets you control your entertainment and answer or end calls
Sweat and water resistant for AirPods and charging case
Lightning Charging Case or MagSafe Charging Case
Up to 6 hours of listening time
Up to 30 hours total listening time"""

a5 = 'Prompt: "Write a product description for a set of earbuds that offer personalized spatial audio with dynamic head tracking, including information on the size of the earbud tips, the force sensor, sweat and water resistance, and the charging case. The description should be concise and informative, with a focus on the technical features and specifications of the product."'

# u6 = "Rewrite the reverse prompt into a prompt that can be used for any product description with a user input feature in the prompt that uses {}."

# u6 = """Reverse Prompt Engineer the following {text}, capture the TONE and WRITING STYLE of the {text} to include in the prompt:
# text = OBAMA SPEECH
# I stand here today humbled by the task before us, grateful for the trust you have bestowed, mindful of the sacrifices borne by our ancestors. thank President Bush for his service to our nation, as well as the generosity and cooperation he has shown throughout this transition. Forty-four American shave now taken the presidential oath. The words have been spoken during rising tides of prosperity and the still waters of peace. Yet, every so often the oath is taken amidst gathering clouds and raging storms. At these moments,America has carried on not simply because of the skill or vision of those in high office, but because We the People have remained faithful to the ideals of our forebears, and true to our founding documents. So it has been. So it must be with this generation of Americans. That we are in the midst of crisis is now well understood. Our nation is at war, against a far-reaching network of violence and hatred. Our economy is badly weakened, a consequence of greed and irresponsibility on the part of some, but also our collective failure to make hard choices and prepare the nation for a new age. Homes have been lost; jobs shed; businesses shuttered. Our health care is too costly; our schools fail too many: and each day brings further evidence that the ways we use energy strengthen our adversaries and threaten our planet."""


u6 = """都说ChatGPT是AI的“iPhone时刻”，现在属于它的“应用商店”来了。

OpenAI刚刚又甩出一个王炸——

宣布推出**插件功能**，赋予ChatGPT使用工具、联网、运行计算的能力。

例如在官方演示中，ChatGPT一旦接入数学知识引擎Wolfram Alpha，就再也不用担心数值计算不精准的问题。

并且还开放了OpenAI自己的两个插件：**网络浏览器和代码解释器**。

以前因为不联网，用户只能查询到2021年9月之前的消息；但现在不仅能直接检索到最新新闻，就连数理计算也一并解决了！

（再也不用担心ChatGPT胡说八道了！）

另外也支持个人或企业把私有数据（文档、笔记、邮件等）投喂给ChatGPT，成为人的“第二大脑”或企业的智能助理。

一闪而过的3页第三方插件列表中，还包括：

- 让ChatGPT查询世界各国语言词汇、短语意思
- 让ChatGPT查询航班、酒店信息，帮你规划差旅
- 让ChatGPT访问各大电商数据，帮你比价甚至直接下单
- 让ChatGPT接入自动化平台Zapier，与几乎所有办公软件连接，创建专属自己的智能工作流
  ……

而这只是一个开始，制作插件的权限和方法，同样会向开发者开放。

一位资深程序员感叹：

> 我为许多软件开发过插件系统，ChatGPT插件接口可能是我一生中见过的最疯狂、最令人印象深刻的。

刚刚回归OpenAI不久的特斯拉前AI主管Andrej Karpathy则表示：

> GPT类模型是一种运行在文本上的新型计算机架构，它不仅可以与我们人类交谈，也可以与现有的软件基础设施“交谈”，API是第一步，插件是第二步。

## ChatGPT有了眼睛和耳朵

按照官方介绍，这个插件功能主要是以**安全**为核心原则的语言模型设计工具

（嗯，换言之，就是不会让你那么快使用。）

OpenAI这波还是得先让用户加入候补名单。

不过他们也说，虽然现在处于Alpha阶段，优先考虑一些少数开发人员和ChatGPT Plus用户，但他们也打包票：

> 更大规模的开放，只是一个时间问题。

在目前已实现的“初步功能”来看，ChatGPT已经可以具备了三项能力：

- 访问最新信息，比如体育比分、股票价格、最新消息；
- 检索知识库信息，比如公司文件、个人笔记、运行计算；
- 代表用户执行操作，比如订机票、订餐。

目前已支持的第三方插件，可以看到有餐厅推荐、旅行计划、购物、AI语言老师、网上店铺，以及学术界知识应用Wolfram、应用库（与5000多个应用程序交互，包括Google表格）……

这样看起来，几乎已经涵盖了我们生活中的衣食住行、工作与学习。

除此之外，OpenAI自己也有两种插件：**网络浏览器和代码解释器**，并开源了一个知识库检索插件的代码。

前面提到，OpenAI此次非常强调安全原则，对此他们实施了多项保障措施，且都是一对一针对功能的。

**1、第三方插件**

首先是针对第三方插件，任何开发人员都可以自行参与构建，OpenAI甚至还给出了一整套构建流程：“如何在ChatGPT构建你的插件”，并在Github上已开源。

在ChatGPT上面的显示是这样的，用户可以自行选择插件并下载。

随后就可以在聊天界面上使用了。

**2、联网。**

> 你能告诉我，哪个人、哪部电影获得了这些类别的奥斯卡？

结果可以看到，ChatGPT在聊天界面就多出来“浏览”结果，直接得到了最新2023年的结果。

OpenAI所做的安全措施包括：用户使用这一插件的范围只限定在检索信息，并不包括“事务性操作”，比如表单提交；使用的是必应检索API继承了微软在信息来源上面的可靠性和真实性；在独立服务器中运行；显示来源等。

**3、代码解释器**：可以使用Python处理上传和下载的实验性ChatGPT模型。

简单来说，就是提供了一个在沙盒、防火墙执行环境中工作的 Python 解释器，以及一些临时磁盘空间。

由解释器插件运行的代码在一个持久会话中进行评估，并且后续调用可以相互构建。目前该功能支持将文件上传到当前对话工作区并下载工作结果。

不过OpenAI这边表示，这一代码解释器主要还是提供计算机的一些最基本的功能，让一些初级程序员和爱好者提供更顺滑的接口。

目前推荐的功能包括：

- 解决定量和定性的数学问题
- 进行数据分析和可视化
- 在格式之间转换文件

## One More Thing

对于ChatGPT新推出插件功能，有开发者表示这基本提供了苹果应用商店式的服务。

但也有人提出，目前还没有看到开发者的获利模式。

批评者则认为，OpenAI刚刚杀死了一个创业公司LangChain，如果像这样下去反而会阻碍了创新。

LangChain提供了利用语言模型API快速构建应用程序的Python框架，刚刚获得1000万美元种子轮融资。

你认为OpenAI会成为AI时代的苹果吗？谁又会是AI时代的安卓？"""

model="gpt-3.5-turbo"
temperature=0.3

output = openai.ChatCompletion.create(
    model=model,
    messages=[
            # {"role": "system", "content": eval(prompt)},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u2},
            {"role": "assistant", "content": a2},
            {"role": "user", "content": u3},
            {"role": "assistant", "content": a3},
            {"role": "user", "content": u4},
            {"role": "assistant", "content": a4},
            {"role": "user", "content": u5},
            {"role": "assistant", "content": a5},
            {"role": "user", "content": u6},
        ],
        temperature=temperature
    )
print(output.choices[0]['message']['content'])