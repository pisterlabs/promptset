import logging
import openai
import backoff

#TODO сделать сообщение "не шмагла" если все попытки провалились

async def create_short_answer(text, system_message):
    res = await __answer(
        text=text,
        system_message=system_message
    )
    return res

#add backoff decorator
@backoff.on_exception(backoff.expo, 
                      openai.error.RateLimitError, 
                      max_tries=5)
@backoff.on_exception(backoff.expo, 
                      Exception, 
                      max_tries=5) #trying to catch all exceptions
async def __answer(text, system_message=''):
    logging.info(f'opean ai request: \'{system_message}\'\n{text}\'')
    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
    )

    answer = completion.choices[0].message.content
    logging.info(f'answer is: \n{answer}')
    return answer