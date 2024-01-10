# %%
import os
import base64
import tqdm
import wordcloud
from collections import Counter
import datetime
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd


# %%
def playback_errors(playback):
    w = ""
    checked = ""
    errors = []
    for event in playback["events"]:
        if event.get("clear"):
            checked = w
            w = ""
        if error := event.get("error"):
            if error.get("message") == "Not a word":
                if checked:
                    errors.append(checked.title())
        if event.get("score") is not None:
            w = ""
        if l := event.get("letter"):
            w = w[:4]
            w += l
        if event.get("backspace"):
            w = w[:-1]
    return errors


# %%
cxn = (os.environ.get("POSTGRES_URL") or "").replace("postgres://", "postgresql://")
# %%
df = pd.read_sql(
    "select * from submissions",
    cxn,
)
df

# %%
df.to_parquet("~/submissions.parquet")
# %%
base = (
    pd.read_parquet("~/submissions.parquet")
    .assign(name=lambda x: x.name.where(x.name != "natalie", "natnat"))
    .query("challenge_id.isna()")
    .assign(
        day=lambda x: pd.to_datetime(x.day),
        year=lambda x: x.day.dt.year,
        dow=lambda x: x.day.dt.dayofweek,
        words=lambda x: x.playback.apply(
            lambda x: [
                x.title()
                for x in (
                    "".join(
                        c["letter"]
                        for c in (
                            event.get("score", [])
                            if event.get("score", []) is not None
                            else []
                        )
                    )
                    for event in x.get("events", [])
                )
                if x and len(Counter(x)) > 1
            ]
        ),
    )
    .query("year == 2023")
    .assign(
        max_score=lambda x: x.groupby("day").score.transform("max"),
        normalized_score=lambda x: 1000 * (1 - ((x.score - 1) / (x.max_score - 1))),
        dow_score=lambda x: x.groupby(["name", "dow"])
        .normalized_score.transform("mean")
        .astype(np.int16),
        overall_score=lambda x: x.groupby(["name"])
        .normalized_score.transform("mean")
        .astype(np.int16),
        games_played=lambda x: x.groupby("name").name.transform("count"),
        starting_day=lambda x: x.groupby("name").day.transform(lambda x: x.min()),
        days_available=lambda x: (
            pd.to_datetime(datetime.datetime.utcnow()) - x.starting_day
        ).dt.days,
        games_missed=lambda x: (x.days_available - x.games_played).clip(0, 365),
        tardy_rate=lambda x: (x.days_available / x.games_missed)
        .clip(0, 365)
        .astype(np.int16),
        errors=lambda x: x.playback.apply(playback_errors),
    )
    .sort_values(["day", "name"], ascending=False)
)
# %%
base.query("rank == 1").name.value_counts().reset_index().rename(
    columns=dict(count="num_wins")
).to_sql("wins", cxn, schema="wrapped", if_exists="replace")
base[["name", "dow", "dow_score"]].drop_duplicates(["name", "dow"]).to_sql(
    "dow", cxn, schema="wrapped", if_exists="replace"
)
base.query("games_played > 10")[["name", "overall_score"]].drop_duplicates().to_sql(
    "overall_score", cxn, schema="wrapped", if_exists="replace"
)
# %%
base[["name", "words"]].explode("words").value_counts().reset_index().rename(
    columns=dict(count="word_count")
).assign(
    count_rank=lambda x: x.groupby("name").word_count.transform(
        "cumcount", ascending=True
    )
).query(
    "count_rank < 5"
).rename(
    columns=dict(word_count="count")
).to_sql(
    "top_words", cxn, schema="wrapped", if_exists="replace"
)
base["words"].explode().value_counts()[:40].reset_index().to_sql(
    "top_words_overall", cxn, schema="wrapped", if_exists="replace"
)
# %%
base.query("games_played > 10").drop_duplicates("name")[
    ["name", "games_missed", "tardy_rate"]
].sort_values("tardy_rate").apply(
    lambda col: col.astype(np.int16) if np.issubdtype(col.dtype, np.number) else col
).to_sql(
    "tardy_rate", cxn, schema="wrapped", if_exists="replace"
)
# %%
base[["name", "errors"]].explode("errors").value_counts().reset_index().rename(
    columns=dict(count="error_count")
).assign(
    count_rank=lambda x: x.groupby("name").error_count.transform(
        "cumcount", ascending=True
    )
).query(
    "count_rank < 20"
).rename(
    columns=dict(error_count="count")
).to_sql(
    "top_errors", cxn, schema="wrapped", if_exists="replace"
)
base["errors"].explode().value_counts()[:40].reset_index().to_sql(
    "top_errors_overall", cxn, schema="wrapped", if_exists="replace"
)
# %%


def make_wordcloud(words, background_color="white"):
    b = BytesIO()
    wordcloud.WordCloud(
        background_color=background_color, width=800, height=800, max_words=50
    ).generate_from_frequencies(words).to_image().save(b, "webp")
    b.flush()
    return b.getvalue()


# %%
pd.concat(
    [
        pd.DataFrame(
            dict(
                name="all",
                image=make_wordcloud(
                    base["errors"]
                    .explode()
                    .value_counts()[:100]
                    .reset_index()
                    .rename(columns=dict(errors="name"))
                    .set_index("name", drop=True)["count"]
                    .to_dict()
                ),
            )
        ),
        pd.DataFrame(
            base[["name", "errors"]]
            .explode("errors")
            .groupby("name")
            .errors.count()
            .reset_index()
            .value_counts()
            .reset_index()
            .rename(columns=dict(count="error_count"))
            .assign(
                count_rank=lambda x: x.groupby("name").error_count.transform(
                    "cumcount", ascending=True
                )
            )
            .query("count_rank < 20")
            .rename(columns=dict(error_count="count"))
        ),
    ]
)
# %%
pd.concat(
    [
        pd.DataFrame(
            base.groupby("name").apply(
                lambda x: make_wordcloud(Counter(sum(x.errors, []) or ["Perfy"]))
            )
        )
        .rename(columns={0: "image"})
        .reset_index(),
        pd.DataFrame(
            dict(
                name=["all"],
                image=[make_wordcloud(Counter(sum(base.errors, [])), "white")],
            )
        ),
    ]
).assign(
    image=lambda x: x.image.apply(lambda x: base64.encodebytes(x).decode())
).to_sql(
    "wordcloud_images", cxn, schema="wrapped", if_exists="replace", index=False
)
# %%
pd.concat(
    [
        pd.DataFrame(
            base.groupby("name").apply(
                lambda x: make_wordcloud(Counter(sum(x.words, []) or ["Perfy"]))
            )
        )
        .rename(columns={0: "image"})
        .reset_index(),
        pd.DataFrame(
            dict(
                name=["all"],
                image=[make_wordcloud(Counter(sum(base.words, [])), "white")],
            )
        ),
    ]
).assign(
    image=lambda x: x.image.apply(lambda x: base64.encodebytes(x).decode())
).to_sql(
    "wordcloud_images_words", cxn, schema="wrapped", if_exists="replace", index=False
)
# %%

import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    # os.environ.get("OPENAI_API_KEY"),
)
# %%
mistakes_df = (
    pd.DataFrame(
        base.groupby("name").apply(
            lambda x: ", ".join(list(Counter(sum(x.errors, []) or ["Perfy"]))[:20])
        )
    )
    .rename(columns={0: "words"})
    .query("words.str.contains(',')")
)

name, mistakes = (mistakes_df.iloc[0].name, mistakes_df.iloc[0].words)
# %%

# %%
results = {}
# %%
for _, row in tqdm.tqdm(mistakes_df.iterrows()):
    name = row.name
    mistakes = row.words
    print(name, mistakes)
    results[row.name] = (
        client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
I want to create a funny story using the made-up words that the users have mispelled.
Please construct a funny story about a person named {name}, about 5 sentences long, using some of the following words: {mistakes}
Please use the person's name ({name}) a lot in the story, and make them the hero.
Try and not make the made-up words proper nouns very often, instead preferring to make them verbs, adjectives, and such in a very creative fashion
""",
                }
            ],
            model="gpt-4",
        )
        .choices[0]
        .message.content
    )

# %%
pd.DataFrame(list(results.items()), columns=["name", "story"]).to_sql(
    "stories", cxn, schema="wrapped", if_exists="replace", index=False
)
