# Applications


def json_print(data):
    print(json.dumps(data, indent=2))


import weaviate, os, json
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file


"""
In this lesson, you'll explore the flexibility of
a multilingual model with a vector database, which
allows you to load and query data in multiple languages. We'll
also introduce the concept of retrieval augmented
generation and explore how you can implement this multi-step
process of retrieval, reasoning and generation
in one simple query. Alright, let's build something cool
or like we say at home, zbudujmycę się fajnego.

So, let's have a quick look at what is multilingual
search and then how it works and what's the
idea behind it. It's very similar to how semantic search
works where we can compare like you know
dog to a puppy. And then, still be able to find a very
similar match. But in case of multilingual search, you can have
the same text but in different languages which will
also generate very similar embeddings if
not identical and through that we can use the
same methods to search across content in any languages
we need. We'll explore this in detail in a code.

So now, let's dive very quickly into what RUG
is and how it works. So, basically the idea that it allows us
to use vector database as an external knowledge, but
also it allows us to retrieve relevant information
and provide it back to the LLN, but
also it synergizes with the data that sits
within a vector database or another benefit of
using RUG is that we can use it in our applications, on our data, without
having to retrain or fine-tune our LLMs. And in a way you can think
of RUG is like you going and visiting a library.
So, if somebody asks you a question without any resources, you may
just make up something. But if you go to library, you can
read a book and then provide a response and that's basically what
RUG does with a vector database. So, some key advantages
of RUG is it allows you to reduce
hallucination, you can also enable your LLMs to
cite sources, but but also it can solve knowledge-intensive tasks, especially for
kind of information that is very rare
to find out in the wild. So, here's a very quick illustration
of how a full RAC query works.
First, we start with a query to a vector database,
which obtains all the relevant source objects. Then,
we can combine these together with a prompt and then
send it to the LLM, which then will go
and generate the response that we are interested in. And here's
a very quick example of a code to perform RAC. You
can see it's very similar to what we've done in
the past with the queries, except this time we are adding the part
with generate. But I'm not gonna spend too much time on it
because I'd like to do it in code. Let's prepare our JSON print function
and this time for our demo we'll be
using an already deployed instance of Weaviate in the cloud, which utilizes Cohere's
multilingual models. And in here, you can
see that we are providing two types of API keys, a Cohere
API key which we use for multilingual search and an OpenAI API key which
we use for generative search. Let's very quickly
see how many objects we are dealing with in our
database. So, we have about 4.3 million Wikipedia articles to
play with which is great. So now, let's have some
fun with this big data set and then run some queries.

In this case, what we want to do is search for
vacation spots in California. Let's run this query and
return five objects. And you can straightaway see that we got few objects
that are returned in English. There's a
one result in French, one in Spanish, and actually
two in Spanish. And another thing that you may have
noticed that we run this query across 4.3 million objects
in a blink of an eye. And to make our lives easier, we'll add a so
that the next results that we get back, they're all in English, and
also, we'll just get three objects back. But a query is still multilingual,
right? So, what else can we do with
this? How about we try to send a query in a different language?

So, in this case, we are asking for vacation spots in California,
but in Polish. And if I run this query, we're
still getting the same results as we had before. And
you may think that searching in Polish, which
uses pretty much the same alphabet is not that impressive. How
about we try something with a completely different
alphabet so we can run the same query
but in Arabic. And that's the performs really well as soon it
returns to objects that talks about vacation spots in
California. Now, let's do some RUG example. So, the starting point
is very similar as we had before. It's a straight up
semantic query and now we can add a prompt.
So, let's call it prompt and the text will be
something like write me a Facebook ad about and I want to
grab the title from my query and then so we do it with the curly brackets
and then we could say using information inside
and then we can provide text and by
doing that we basically constructed the prompt based
on the results we get from the query and. To
exit the actual query, we need to add with generate, and
we'll pass in a single prompt, which will take our prompt that we
constructed just above. And by doing this, we're basically asking
the vector database to pass in the same
prompt for every single object. So, we should get
three different generations as a response, and in here
we can see the results of the generation. So, this is
brand new content generated by GPT. So in this
case, we have you know looking for some fun in the sun
then look no further and this is the text is what it was constructed on, and
the same thing happened for the other
two responses.
Another type of a RAC query is a group task which basically
takes a prompt and then also runs a query and then sends
all the results as a single query into GPT and then we
can run it and expect only one generation
and like in this case we want to summarize what these posts
are about into paragraphs. And as a result, we get a summary
of all the three posts that were returned
from the original query. And that concludes this lesson in
which you learn how to use multilingual search where
you were able to search across content written in any language,
but also by providing queries in whatever language you needed.
And we also went over a few examples of rack query,
in which we use single prominent group task to be able to
generate responses based on individual objects or based on
collective response.

"""

auth_config = weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

client = weaviate.Client(
    url=os.getenv("WEAVIATE_API_URL"),
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.getenv("COHERE_API_KEY"),
        "X-Cohere-BaseURL": os.getenv("CO_API_URL"),
    },
)

client.is_ready()  # check if True

### 2. How many vectors are stored in this database

print(json.dumps(client.query.aggregate("Wikipedia").with_meta_count().do(), indent=2))

### 3. Perform search over them to find concepts you are interested in!

response = (
    client.query.get("Wikipedia", ["text", "title", "url", "views", "lang"])
    .with_near_text({"concepts": "Vacation spots in california"})
    .with_limit(5)
    .do()
)

json_print(response)

response = (
    client.query.get("Wikipedia", ["text", "title", "url", "views", "lang"])
    .with_near_text({"concepts": "Vacation spots in california"})
    .with_where({"path": ["lang"], "operator": "Equal", "valueString": "en"})
    .with_limit(3)
    .do()
)

json_print(response)

response = (
    client.query.get("Wikipedia", ["text", "title", "url", "views", "lang"])
    .with_near_text({"concepts": "Miejsca na wakacje w Kalifornii"})
    .with_where({"path": ["lang"], "operator": "Equal", "valueString": "en"})
    .with_limit(3)
    .do()
)

json_print(response)

response = (
    client.query.get("Wikipedia", ["text", "title", "url", "views", "lang"])
    .with_near_text({"concepts": "أماكن العطلات في كاليفورنيا"})
    .with_where({"path": ["lang"], "operator": "Equal", "valueString": "en"})
    .with_limit(3)
    .do()
)

json_print(response)

## Retrieval Augmented Generation

### Single Prompt

prompt = "Write me a facebook ad about {title} using information inside {text}"
result = (
    client.query.get("Wikipedia", ["title", "text"])
    .with_generate(single_prompt=prompt)
    .with_near_text({"concepts": ["Vacation spots in california"]})
    .with_limit(3)
).do()

json_print(result)

### Group Task

generate_prompt = "Summarize what these posts are about in two paragraphs."

result = (
    client.query.get("Wikipedia", ["title", "text"])
    .with_generate(grouped_task=generate_prompt)  # Pass in all objects at once
    .with_near_text({"concepts": ["Vacation spots in california"]})
    .with_limit(3)
).do()

json_print(result)
