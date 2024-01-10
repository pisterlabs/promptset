from Media_Class import Media
import openai

def GPTsummarize(text_content, number_of_tokens):
    tldr_tag = "\n tl;dr:"
    openai.organization = 'org-7Sugy3cO8qQNP9Tdv17gKJi8'
    openai.api_key = "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"

    response = openai.Completion.create(engine="davinci", prompt=text_content+tldr_tag, temperature=0.3,
                                        max_tokens=int(number_of_tokens),
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0,
                                        stop=["\n tl;dr:"]
                                        )
    return response["choices"][0]["text"]


def tokenManager(text_content, number_of_tokens_out):
    total_text_length = len(text_content.split())
    total_tokens = int(total_text_length) * 1.33 + int(number_of_tokens_out)

    if total_tokens > 2049:
        print("Big text")
        total_split = round(total_tokens / 2049)
        summary_length_per_split = number_of_tokens_out / total_split
        text_length_per_split = (2049 - summary_length_per_split) / total_split

        print(text_length_per_split)

        list_of_texts = []
        counter = 0
        list_of_words = []
        for word in text_content.split():
            counter += 1
            list_of_words.append(word)
            if counter == round(text_length_per_split):
                full_string = ' '.join(list_of_words)
                list_of_texts.append(full_string)
                list_of_words = []
                counter = 0

        print(list_of_texts)
        final_output = ''
        for text in list_of_texts:
            summary = GPTsummarize(text_content=text, number_of_tokens=summary_length_per_split)
            final_output += ' ' + summary
            return final_output

    else:
        summary = GPTsummarize(text_content=text_content, number_of_tokens=number_of_tokens_out)
        return summary


def media_management(content_to_summarise, percent_reduce, source, persistent_user_id, session_id, content_format, local):
    WorkingContent = Media(media=content_to_summarise, perc_reduction=percent_reduce,
                           source=source, user_id=persistent_user_id, session_id=session_id,
                           media_format=content_format, local=local)

    WorkingContent.convert_and_clean_media()
    tokens_out = WorkingContent.final_word_count_out * 1.33
    summary = tokenManager(text_content=WorkingContent.final_clean_text, number_of_tokens_out=tokens_out)

    return summary


if __name__ == "__main__":
    document_to_use = "FINAL_UCL Personal Statement - Copy.pdf"
    percent_reduce = 5
    summary_out = media_management(content_to_summarise=document_to_use, percent_reduce=percent_reduce,
                                   source='local', persistent_user_id=None, session_id=None, content_format="pdf",
                                   local=True)

    print(summary_out)