import openai
from openai import OpenAI


client = OpenAI()

def generate_seo_recommendations(blog_post, audience):
    seo_directive = f"""
    Please review the following text for SEO and provide recommendations.
    Be sure that the SEO created also fits the {audience} 

    Specifically, we need:
1. An SEO Focus Keyphrase: This should be a term or phrase you'd want to rank for in search engines.
2. An SEO Title: This is the title of your post as it should appear in search engine results. It should be clear, concise, and related to your keyphrase.
3. An SEO Slug: This is the part of the URL that distinguishes your post. It should be short, descriptive, and include your keyphrase.
4. An SEO Meta Description: This should be a brief summary of your post, not exceeding 125 characters (including spaces and punctuation), that will appear in search engine results.
5. Tags: Please provide at least 15 tags related to the post that can be used in WordPress. These should be relevant keywords or phrases, separated by commas.


    Here is the text for review:

    {blog_post}
    """
    seo_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an SEO expert working for Yoast."},
            {"role": "user", "content": seo_directive},
        ],
    )

    seo_recommendations = seo_response.choices[0].message.content
    print("\nSEO recommendations:")
    print(seo_recommendations)

    yoast_seo_data = extract_yoast_seo_data(seo_recommendations)
    tags = extract_tags(seo_recommendations)
    return yoast_seo_data, tags

def extract_yoast_seo_data(recommendations):
    # Parsing the recommendations string to extract SEO data
    lines = recommendations.split('\n')
    seo_data = {}
    for line in lines:
        if 'Focus Keyphrase:' in line:
            seo_data['focus_keyphrase'] = line.split(':', 1)[1].strip()
        elif 'SEO Title:' in line:
            seo_data['seo_title'] = line.split(':', 1)[1].strip()
        elif 'SEO Slug:' in line:
            seo_data['slug'] = line.split(':', 1)[1].strip()
        elif 'SEO Meta Description:' in line:
            seo_data['meta_description'] = line.split(':', 1)[1].strip()
    return seo_data

def extract_tags(recommendations):
    # Parsing the recommendations string to extract tags
    for line in recommendations.split('\n'):
        if "Tags:" in line:
            tags_part = line.split("Tags:", 1)[1]
            tags = [tag.strip() for tag in tags_part.split(',')]
            return tags
    return []

