from openai import OpenAI


class Mathwork:
    def __init__(self, settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.get_api_key(),
                             # http_client=httpx.Client(
                             #    proxies="http://127.0.0.1:7890",
                             #    transport=httpx.HTTPTransport(local_address='0.0.0.0')
                             # ),
                             )

    def get_homework(self, requirement):
        homework_prompt = self.settings.get_default_subject().get_homework_prompt()
        completion_ = self.client.chat.completions.create(model="gpt-3.5-turbo",
                                                          temperature=0.2,
                                                          messages=[{"role": "system", "content": homework_prompt},
                                                                    {"role": "user", "content": requirement}]
                                                          )
        return completion_.choices[0].message.content

    def get_markdown(self, homework):
        markdown_prompt = self.settings.get_default_subject().get_markdown_prompt()
        completion_ = self.client.chat.completions.create(model="gpt-3.5-turbo",
                                                          temperature=0.1,
                                                          messages=[{"role": "system", "content": markdown_prompt},
                                                                    {"role": "user", "content": homework}]
                                                          )
        return completion_.choices[0].message.content
