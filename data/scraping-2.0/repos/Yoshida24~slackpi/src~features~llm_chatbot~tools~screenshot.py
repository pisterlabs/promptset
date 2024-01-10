from io import BytesIO
from playwright.sync_api import sync_playwright
from openai.types.chat import ChatCompletionToolParam


def take_screenshot_impl(url: str, **kwargs) -> dict:
    # BytesIOオブジェクトを作成します
    output = BytesIO()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        screenshot = page.screenshot(full_page=True)
        output.write(screenshot)
        browser.close()

    return {
        "message": "スクリーンショットを正常に取得しました。あなたのプラットフォームにアップロード機能があれば、それを使用してスクリーンショットをあなたのプラットフォームへアップロードします。",
        "file": output,
    }


take_screenshot_tool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "take_screenshot",
        "description": "Save screenshot of web page.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to take screenshot. e.g. https://www.google.com/",
                },
            },
        },
    },
}
