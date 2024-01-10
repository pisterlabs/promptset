
import os
import openai


prompts = {
    "blog_outline": lambda topic, tone, keywords: f"Generate a blog post outline for the following topic: {topic}. Write it in a “{tone}”  tone. The outline should contain various subheadings and include the starting sentence for each section. Include the following keywords: “{keywords}”.",
    "blog_post": lambda topic, tone, keywords: f"Create a blog post about {topic}. Write it in a {tone}  tone. Use transition words. Use active voice. Write over 200 words. Use very creative titles for the blog post. Add a title for each section. Ensure there are a minimum of 9 sections. Each section should have a minimum of two paragraphs. Include the following keywords: {keywords}. Create a good slug for this post and a meta description with a maximum of 100 words and add it to the end of the blog post.",
    "social_media_facebook": lambda topic, tone, keywords: f"""
Generate a Facebook  about {topic} tone. The post should:

1. Start with a catchy headline that grabs attention.
2. Provide a brief and engaging description of the {topic}.
3. Be friendly and approachable in tone, suitable for our audience.
4. Use persuasive language to highlight the benefits or unique aspects of the {topic}.
5. Write it in a {tone} tone.
6. Include the following keywords: {keywords}.
7. [Optional: Include any specific phrases, slogans, or hashtags that should be featured in the post.]

Target length: 200.
""",
    "social_media_twitter": lambda topic, tone, keywords: f"""
Generate a Twitter post for [Your Company/Personal Profile] about {topic}. The tweet should:

1. Start with a concise and engaging statement that captures attention.
2. Briefly describe the {topic}, focusing on key points due to the character limit.
4. Be written in a tone that is engaging and suitable for our audience.
5. Incorporate relevant hashtags and mentions (@) if necessary.
6. Write it in a {tone} tone.
7. Include the following keywords: {keywords}.
8. [Optional: Include any specific phrases, slogans, or any other requirements specific to the tweet.]

Target length: Twitter has a 280-character limit, so the message needs to be concise.
""",
"social_media_linkedin": lambda topic, tone, keywords: f"""""
            Generate a LinkedIn post  about {topic}. The post should:

            1. Start with an engaging opening that captures the professional nature of LinkedIn and draws attention.
            2. Elaborate on the {topic} insight, providing valuable information or insights suitable for a professional audience.
            3. Maintain a tone that is professional and informative.
            4. Highlight key benefits, innovations, or professional learnings without a direct call to action.
            5. [Optional: Include any specific phrases, industry jargon, or hashtags that are relevant and commonly used in your professional community.]
            6. Write it in a {tone} tone.
            7. Include the following keywords: {keywords}.
            Target length: 1000.
""""",
    "ad_content": lambda topic, tone, keywords: f"Generate ad copy for a promotion focusing on '{topic}'. The ad should highlight key aspects and create urgency, using keywords like {keywords}. Write it in a “{tone}” tone.",
    "email": lambda topic, tone, keywords: f"Write a professional email in relation to '{topic}'. The tone should be {tone}. The email should be concise, clear, and relevant. Include the following keywords: {keywords}. Include subject.",
    "essay": lambda topic, tone, keywords: f""""
Write an academic essay on the topic: {topic}. important: Write over 1000 words. The essay should:

1. Begin with an introduction that presents the topic and a clear thesis statement. The thesis should define the main argument or purpose of the essay.
2. Include several body paragraphs, each focusing on a specific aspect or argument related to the topic. Ensure each paragraph starts with a topic sentence and includes evidence, examples, or research to support the argument.
3. Use credible and scholarly sources for references and cite them appropriately. Ensure the information is accurate, relevant, and up-to-date.
4. Maintain formal and academic tone throughout the essay. Avoid colloquial language, slang, or informal expressions.
5. Include critical analysis and discussion, rather than merely describing or summarizing the topic.
6. Conclude with a strong closing paragraph that summarizes the main points and reiterates the thesis, providing a clear conclusion to the argument.
7. Ensure the essay is structured logically, with clear transitions between paragraphs and ideas.
8. [Optional: Include any specific instructions related to the assignment or professor's requirements.]


""",
}


def generate_content(content_type, topic, tone, keywords, language):
    openai_api_key = os.environ.get("OPEN_AI_KEY_AIHUMANIZE")
    client = openai.OpenAI(api_key=openai_api_key)
    # Framing the AI as an expert in the given field
    user_context = prompts[content_type](topic, tone, keywords)

    

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # Experiment with different models as needed
            messages=[
                {
                    "role": "system",
                    "content": f"You are proffesional writer in {language} language and. Return only result do not write anything extra. instead of markdown use html tags. return pure html content but without html and body.",
                },
                {
                    "role": "user",
                    "content": user_context,
                },
            ],
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# content_type = "blog_post"
# topic = "sustainable living practices"
# tone = "informative and engaging"
# keywords = "eco-friendly, green energy, recycling"

# content = generate_content(content_type, topic, tone, keywords)
# print(content)
# result = generate_content("How to make a website. Comprehensive guide.", "Informative", "HTML, CSS, JavaScript", "English", 1000, 900)
# print(result)
    

def extend_text(text, tone, keywords, language, max_words_count, min_words_count):
    openai_api_key = os.environ.get("OPEN_AI_KEY")


    client = openai.OpenAI(api_key=openai_api_key)

    

    # Updated system prompt
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # You can experiment with different models
            messages=[
                {
                    "role": "system",
                    "content": f"You are a skilled content writer. Continue the given text with a short, engaging, and informative paragraph. return only result.",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None