```python
from openai_api import OpenAI_API
from wordpress_api import WordPress_API

class Optimizer:
    def __init__(self):
        self.openai_api = OpenAI_API()
        self.wordpress_api = WordPress_API()

    def optimize_posts(self):
        posts = self.wordpress_api.get_posts()
        if posts is not None:
            for post in posts:
                optimized_content = self.openai_api.optimize_content(post.content)
                if optimized_content is not None:
                    self.wordpress_api.update_post(post.id, optimized_content)
                else:
                    print(f"Could not optimize post with id {post.id}")
        else:
            print("Could not retrieve posts")

if __name__ == "__main__":
    optimizer = Optimizer()
    optimizer.optimize_posts()
```
