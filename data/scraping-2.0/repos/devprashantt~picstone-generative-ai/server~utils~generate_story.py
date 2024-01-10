import openai


def generate_story(tags, tag_analysis, image_text, story_title, desc, themes, ai_content):
    # Extract detected moods, sentiments, and tones from tag_analysis
    detected_moods = tag_analysis.get("moods", [])
    detected_sentiments = tag_analysis.get("sentiments", [])
    detected_tones = tag_analysis.get("tones", [])

    # Define default values if not detected
    default_mood = "neutral"
    default_sentiment = "neutral"
    default_tone = "calm"

    # Use the detected values if available; otherwise, use defaults
    mood = ', '.join(detected_moods) if detected_moods else default_mood
    sentiment = ', '.join(
        detected_sentiments) if detected_sentiments else default_sentiment
    tone = ', '.join(detected_tones) if detected_tones else default_tone

    # Create a prompt with specific instructions for ChatGPT
    prompt = f"""Generate a captivating story based on the provided image and information. The image analysis has extracted tags, and further analysis has revealed moods: {mood}, sentiments: {sentiment}, and tones: {tone}. The OCR applied to the image has provided the following text: {image_text}. The user has contributed a story titled "{story_title}" with the description: "{desc}" and themes: {themes}. Additionally, an AI content analysis has generated the following caption: "{ai_content}". Create a narrative that seamlessly incorporates these elements into a coherent and engaging story."""

    try:
        # Generate a story/poem using ChatGPT
        response = openai.Completion.create(
            engine="text-davinci-003",
            temperature=0.7,  # Adjust temperature for creativity
            max_tokens=1000,  # Adjust max_tokens for desired length
            prompt=prompt,
            n=1             # Ensure only one response is generated
        )

        return response.choices[0].text
    except Exception as e:
        print(f"Error generating poem/story from ChatGPT: {str(e)}")
        raise e
