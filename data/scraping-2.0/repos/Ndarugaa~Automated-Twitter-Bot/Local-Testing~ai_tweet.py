import openai
import random

def generate_tweet(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        stop=["\n"]
    )
    return response['choices'][0]['text'].strip()

def generate_reply(prompt):
    return generate_tweet(prompt)

topic = [
    "ETL (Extract, Transform, Load) Pipelines", "Data Warehousing", "Real-time Data Streaming", "Data Lakes", "Data Governance",
    "Cloud Data Engineering", "Data Integration", "Batch Processing", "Data Pipelines", "Big Data Technologies", "Data Modeling",
    "Data Transformation", "Data Catalogs", "Data Architecture", "Data Security", "Data Backup and Recovery", "Data Migration",
    "Data Visualization", "Data Monitoring", "Data Performance Tuning", "Data Governance Frameworks", "Data Engineering Tools",
    "Data Ethics", "Data Quality Assurance", "Data Engineering Challenges", "Data Standards", "Data Collaboration",
    "Data Governance Policies", "Data Compliance", "Data Engineering Career", "Data Lineage", "Data Profiling", "Data Deduplication",
    "Data Ingestion", "Data Transformation", "Data Cleansing", "Data Orchestration", "Data Replication", "Data Partitioning",
    "Data Serialization", "Data Serialization Formats", "Data Compression", "Data Archiving", "Data Encryption", "Data Privacy",
    "Data Access Control", "Data Load Balancing", "Data Synchronization", "Data Governance Frameworks", "Data Cataloging Tools",
    "Data Pipelines Orchestration", "Data Versioning", "Data Storage Solutions", "Data Scalability", "Data Availability",
    "Data Virtualization", "Data Federation", "Data Streaming Architectures", "Data Pipeline Monitoring", "Data Pipeline Optimization"
]

tweet_prompt = f"Generate a joke as tweet on {random.choice(topic)}"
tweet = generate_tweet(tweet_prompt)
