import openai

sys = ""

u1 = "我们可以谈谈逆向提示工程吗？逆向提示工程是指从给定的文本中创建提示。"

a1 = """反向提示工程是一种有趣且有用的技术，它涉及生成一个提示，该提示可用于生成类似于给定输入文本的文本。这种技术在自然语言处理和机器学习中越来越受欢迎。

反向提示工程的想法是创建一个提示，为语言模型提供足够的信息，以生成与输入文本一致的文本。提示应该设计成包括有关输入文本的上下文、风格和结构的相关信息。

反向提示工程可以用于各种应用程序，例如生成摘要、改写文本或生成对给定提示的响应。它还可以通过提供更具体和相关的任务信息来提高语言模型的性能。"""

u2 = "很好。你能给我一个反向提示工程的简单例子吗？"

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

a3 = """当然，这里简单解释一下训狗的原理：

犬只训练是教导犬只以某种方式对特定命令或刺激做出反应的过程。它通常涉及使用正强化或负强化技术来鼓励理想的行为和阻止不理想的行为。正强化包括奖励狗表现出所需的行为，而负强化包括在表现出所需行为时消除负面刺激。一致性、耐心和重复性是成功训练狗的关键因素。

现在，让我们对这一解释进行反向提示工程，以生成一个提示，可用于生成与这一解释类似的文本：

提示： "解释一下教狗以某种方式行事的过程以及其中涉及的技巧。"

这个提示抓住了解释的关键因素：教狗以某种方式行事的过程，正负强化技术的使用，以及一致性、耐心和重复的重要性。它还对文本（即解释）的风格和结构提供了一些指导。

使用这个提示，语言模型可以生成在风格和结构上与解释相似的文本。例如，它可以生成以下内容：

输出： "教狗以某种方式行事包括使用积极和消极的强化技术。正强化包括奖励狗表现出理想的行为，而负强化包括在表现出理想的行为时消除负面刺激。一致性、耐心和重复是成功训练狗的重要因素"。

在这个输出中，我们看到了与原来的解释类似的结构和风格，对训犬过程和技术进行了清晰的解释。"""

u4 = "好的。你可以创建一个非常技术性的反向提示工程模板吗？"

a4 = '提示："使用[插入技术术语]，解释[插入概念或任务]和支配它的基本原则。"'

u5 = """你能不能对以下{产品描述）进行反向提示工程，抓住写作风格和文字长度：
产品描述 = 个性化空间音频与动态头部追踪将声音放在您的周围。注意：如果耳塞头的尺寸与您的耳道尺寸不匹配，或者耳机没有正确佩戴在您的耳朵上，您可能无法获得正确的声音质量或通话性能。请将耳塞头换成更适合您耳朵的耳塞头。
单一适合
力感应器让您可以控制您的娱乐和接听或结束通话
AirPods和充电盒具有防汗和防水功能
闪电充电盒或MagSafe充电盒
长达6小时的聆听时间
总聆听时间可达30小时"""

a5 = '提示："为一套提供动态头部追踪的个性化空间音频的耳塞写一份产品描述，包括关于耳塞头的尺寸、力传感器、防汗和防水以及充电盒的信息。描述应简明扼要，内容丰富，重点介绍产品的技术特点和规格。"'

# u6 = "Rewrite the reverse prompt into a prompt that can be used for any product description with a user input feature in the prompt that uses {}."

# u6 = """Reverse Prompt Engineer the following {text}, capture the TONE and WRITING STYLE of the {text} to include in the prompt:
# text = OBAMA SPEECH
# I stand here today humbled by the task before us, grateful for the trust you have bestowed, mindful of the sacrifices borne by our ancestors. thank President Bush for his service to our nation, as well as the generosity and cooperation he has shown throughout this transition. Forty-four American shave now taken the presidential oath. The words have been spoken during rising tides of prosperity and the still waters of peace. Yet, every so often the oath is taken amidst gathering clouds and raging storms. At these moments,America has carried on not simply because of the skill or vision of those in high office, but because We the People have remained faithful to the ideals of our forebears, and true to our founding documents. So it has been. So it must be with this generation of Americans. That we are in the midst of crisis is now well understood. Our nation is at war, against a far-reaching network of violence and hatred. Our economy is badly weakened, a consequence of greed and irresponsibility on the part of some, but also our collective failure to make hard choices and prepare the nation for a new age. Homes have been lost; jobs shed; businesses shuttered. Our health care is too costly; our schools fail too many: and each day brings further evidence that the ways we use energy strengthen our adversaries and threaten our planet."""


u6 = """对以下营销号文章进行反向提示工程，抓住写作风格、语气和文字长度。允许使用外部知识。文章如下：

都说ChatGPT是AI的“iPhone时刻”，现在属于它的“应用商店”来了。

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