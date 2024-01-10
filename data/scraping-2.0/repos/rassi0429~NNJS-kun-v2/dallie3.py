import openai
openai.api_key = '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'
async def genImg(prompt, rawPrompt = ""):
    print(prompt)
    res = "1024x1024"
    if ("+yoko" in rawPrompt):
        res = "1792x1024"
    if ("+tate" in rawPrompt):
        res = "1024x1792"
    print(res)
    try:
    # Official API Reference: https://beta.openai.com/docs/api-reference/images
        response = await openai.Image.acreate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size=res
        )
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        print(e)
        return ""

# image_url = genImg('a white cat')
# print(image_url)