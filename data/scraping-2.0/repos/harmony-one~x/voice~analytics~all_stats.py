from openai_stats import get_gpt_stats
from sentry_data import get_prod_error_count

def main():
    ## Elastic search source
    days_timeframe = 1

    print("GPT Stats from Elastic Search")
    stats = get_gpt_stats(days_timeframe)
    print("\n".join("{!r}: {!r},".format(k, v) for k, v in stats.items()), "\n")

    ## Sentry source
    organization_slug = "harmony-23"
    project_slug = "apple-ios"

    print("App error stats from Sentry")
    print(get_prod_error_count(organization_slug, project_slug), "\n")

if __name__ == '__main__':
    main()
