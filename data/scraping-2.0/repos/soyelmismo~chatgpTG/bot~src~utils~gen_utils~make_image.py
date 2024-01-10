async def gen(self, prompt, model, current_api, style, ratio, seed=None, negative=None):
    try:
        from bot.src.utils.proxies import config

        from bot.src.utils.constants import Style
        prompt=f'{prompt + Style[style].value[3]}'

        from bot.src.apis import stablehorde
        api_key = config.api["info"][current_api].get("key", None)
        if current_api == "stablehorde":
            if isinstance(negative, str):
                prompt += f" ### {negative}"
            image, seed, model = await stablehorde.main(self, api_key, prompt=prompt, model=model, seed=seed)
            return image, seed, model
        import openai
        if self.proxies is not None:
            openai.proxy = {f'{config.proxy_raw.split("://")[0]}': f'{config.proxy_raw}'}
        openai.api_key = api_key
        openai.api_base = config.api["info"][current_api]["url"]
        r = await openai.Image.acreate(prompt=prompt, n=config.n_images, size="1024x1024")
        image_urls = [item.url for item in r.data]
        return image_urls, None, None
    except Exception as e:
        raise RuntimeError(f'make_image.gen > {e}')