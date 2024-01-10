import random

from flask import Blueprint, render_template, request

from tttutor import ai, db

bp = Blueprint("posts", __name__, url_prefix="/posts")


@bp.route("/", methods=("GET", "POST"))
def posts():
    facts = []

    if request.method == "POST":
        topic = request.form.get("topic")
        dev_mode = request.form.get("dev", "prod")
    elif request.method == "GET":
        topic = request.args.get("topic")
        dev_mode = request.args.get("dev", "prod")
    else:
        raise RuntimeError(request.method)

    n = 10

    posts = []
    title = "Search"

    if topic or facts:
        if dev_mode == "cache-only":
            # Load from cache
            posts = db.load_posts(topic=topic, n=n)

        elif dev_mode == "no-cache":
            posts = ai.get_greentexts(topic=topic, n=n)

        elif dev_mode == "prod":
            # Load half from cache, half from openai
            cached_posts = db.load_posts(topic=topic, n=n // 2)

            n = n - len(posts)
            new_posts = ai.get_greentexts(topic=topic, n=n)

            posts = cached_posts + new_posts
            random.shuffle(posts)
        else:
            raise ValueError(dev_mode)
        title = topic

    return render_template("posts.html", posts=posts, title=title)
