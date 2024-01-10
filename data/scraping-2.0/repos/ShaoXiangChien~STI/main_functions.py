import streamlit as st
import json
import arrow
from datetime import datetime as dt
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from pprint import pprint


def keyword_extract(method, df):
    if method == "tfidf":
        import keyword_extraction.tfidf_kw_extract as kw
        keywords = kw.tfidf_kw_extract(df)
    elif method == "textrank":
        import keyword_extraction.textrank_kw_extract as kw
        keywords = kw.textrank_kw_extract(df)
    elif method == "azure language service":
        import keyword_extraction.azure_kw_extract as kw
        keywords = kw.azure_kw_extract(df)
    elif method == "ckip":
        import keyword_extraction.ckip_kw_extract as kw
        keywords = kw.ckip_kw_extract(df)
    elif method == "ckip_tfidf":
        import keyword_extraction.ckip_tfidf_kw_extract as kw
        keywords = kw.ckip_tfidf_kw_extract(df)
    else:
        import keyword_extraction.openai_kw_extract as kw
        keywords = kw.openai_kw_extract(df)

    return keywords


def summarize(method, df):
    st.write("Initializing")
    summary = ""
    if method == "naive":
        import summarization.naive_summarize as sm
        sentences = cut_sentences("".join(df['full_text'].to_list()))
        tokenized_sentences = cut_sentences(
            " ".join(df['full_text_tokens'].to_list()))
        summary = sm.naive_summarize(sentences, tokenized_sentences)
    elif method == "kmeans":
        import summarization.kmeans_summarize as sm
        sentences = cut_sentences("".join(random.choices(
            df['full_text_tokens'].to_list(), k=int(len(df['full_text_tokens'].to_list()) * 0.3))))
        summary = sm.kmeans_summarize(sentences)
    elif method == "textrank":
        import summarization.textrank_summarize as sm
        sentences = cut_sentences("".join(df['full_text_tokens'].to_list()))
        summary = sm.textrank_summarize(sentences)
    elif method == "openai":
        import openai_services as sm
        docs = "".join(random.choices(
            df['full_text'].to_list(), k=10))[:1500]
        summary = sm.summarize(docs)
    elif method == "azure_language_service":
        import summarization.azure_summarize as sm
        docs = "".join(df['full_text'].to_list())[:100000]
        summary = sm.azure_summarize([docs])
    return summary


def cut_sentences(content):
    end_flag = ['?', '!', '？', '！', '。', '…']

    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        tmp_char += char

        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break

        if char in end_flag:
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char)
                tmp_char = ''

    return sentences


def choose_timeline_rp(df, kws):
    n = df['cluster'].max()
    timeline_rp = {i: "" for i in range(n)}
    for i in range(n):
        candidate = df[df['cluster'] == i].copy()
        max_score = 0
        target = ""
        for doc in candidate['Event']:
            score = 0
            for w in kws[i]:
                if w in doc:
                    score += 1
            if score > max_score:
                max_score = score
                target = doc
        timeline_rp[i] = target
    return timeline_rp


def generate_timeline(time_df, anomalies):
    anomaly_ft = time_df.timestamp.apply(lambda x: x in anomalies)
    anomaly_df = time_df[anomaly_ft].copy()
    anomaly_df.reset_index(inplace=True)
    anomaly_df.drop("index", axis=1, inplace=True)

    full_text = "".join(anomaly_df.Event.to_list())
    freq_table = {kw: full_text.count(kw)
                  for kw in st.session_state['keywords']}
    sentence_score = {}
    for sent in anomaly_df.Event:
        score = 0
        for k, v in freq_table.items():
            if k in sent:
                score += v
        sentence_score[sent] = score / len(sent)

    timeline = {timestamp: [] for timestamp in anomalies}
    for idx, row in anomaly_df.iterrows():
        event = row['Event']
        if row['timestamp'] in anomalies:
            timeline[row['timestamp']].append(event)

    for k, v in timeline.items():
        tmp = sorted(v, key=lambda x: sentence_score.get(
            x) if sentence_score.get(x) else 0, reverse=True)
        timeline[k] = tmp[0]

    with open(f"./Experiments/{st.session_state['event']}/timeline.json", "w") as fh:
        json.dump(timeline, fh)

    data = {
        "events": []
    }
    for k, v in timeline.items():
        time_obj = dt.strptime(k, "%Y-%m-%dT%H:%M:%SZ")
        date = {
            "year": time_obj.year,
            "month": time_obj.month,
            "day": time_obj.day,
        }
        text = {
            "text": v
        }
        data['events'].append({
            "start_date": date,
            "text": text
        })
    return data


def generate_timeline_beta(time_df, df, anomalies):
    anomaly_ft = time_df.timestamp.apply(lambda x: x in anomalies)
    anomaly_df = time_df[anomaly_ft].copy()
    anomaly_df.reset_index(inplace=True)
    anomaly_df.drop("index", axis=1, inplace=True)

    docs = anomaly_df['Event'].to_list()
    umap_model = UMAP(n_neighbors=5, n_components=20,
                      min_dist=0.1, metric='cosine')
    topic_model = BERTopic(language='multilingual', umap_model=umap_model)
    topics, probs = topic_model.fit_transform(docs)
    df['topic'] = topics
    representative_list = []
    for i, event in topic_model.get_representative_docs().items():
        representative_list.append(event[0])
        final_topics = pd.DataFrame({
            "Time": [],
            "Event": []
        })
        for event in representative_list:
            final_topics = final_topics.append({
                "Time": df[df["Event"] == event].iat[0, 0],
                "Event": event
            }, ignore_index=True)

    final_topics["Time"] = final_topics["Time"].astype('string')
    for idx, row in final_topics.iterrows():
        final_topics.loc[idx, "Time"] = final_topics.loc[idx,
                                                         "Time"].replace('年', '/')
        final_topics.loc[idx, "Time"] = final_topics.loc[idx,
                                                         "Time"].replace('月', '/')
        final_topics.loc[idx, "Time"] = final_topics.loc[idx,
                                                         "Time"].replace('日', '')
        final_topics.loc[idx, "Time"] = final_topics.loc[idx,
                                                         "Time"].replace('號', '')
        Time = arrow.get(final_topics.loc[idx, "Time"])
        final_topics.loc[idx, "Time"] = Time.format("YYYY-MM-DD")

    data = {
        "events": []
    }
    for idx, row in final_topics.items():
        time_obj = row['Time']
        date = {
            "year": time_obj.year,
            "month": time_obj.month,
            "day": time_obj.day,
        }
        text = {
            "text": row['Event']
        }
        data['events'].append({
            "start_date": date,
            "text": text
        })

    return final_topics
