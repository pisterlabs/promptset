import openai

openai.ChatCompletion.create(
    engine="davinci",
    prompt="This is a test",
    temperature=0.9,
)
      const arrayBuffer = new Uint8Array(audioBuffer).buffer;
      const blob = new Blob([arrayBuffer], { type: contentType });

      formData.append("file", blob, "audio.wav");