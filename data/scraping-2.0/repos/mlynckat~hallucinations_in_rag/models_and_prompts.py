from langchain.output_parsers import ResponseSchema

SETTINGS = {
    "summarize_section": {
        "template": """
            Summarize the following section of the paper: {section}. Pay attention to the following points:
            - Experiments
            - Novelty of the proposed approaches
            - Results
            Metrics to detect and / or mitigate the hallucinations of the large language models.
            If there are any metrics to measure or detect the hallucinations, mention them separately. Describe how they can be calculated and how they were used in the paper.
            Also mention the mitigation methods if there are any. Describe them.
            The summary of the previous section is: {prev_summary}.
            {format_instructions}
            """,
        "response_schemas": [ResponseSchema(name="summary",
                                            description="A rather short summary of the current section of the paper"),
                             ResponseSchema(name="metrics",
                                            description="The name of metrics mentioned to detect and measure the hallucinations of the large language models as well as the description. If there were no ,metrics mentioned, return empty string."),
                             ResponseSchema(name="metrics_description",
                                            description="The description of the metrics mentioned to detect and measure the hallucinations of the large language models. If there were no metrics mentioned, return empty string."),
                             ResponseSchema(name="mitigation",
                                            description="The name of mitigation methods mentioned to mitigate the hallucinations of the large language models as well as the description. If there were no mitigation methods mentioned, return empty string."),
                             ResponseSchema(name="mitigation_description",
                                            description="The description of the mitigation methods mentioned to mitigate the hallucinations of the large language models. If there were no mitigation methods mentioned, return empty string."),
                             ],
        "input_variables": ["section", "prev_summary"],
        "output_variables": ["summary", "metrics", "metrics_description", "mitigation", "mitigation_description"]
    },
    "summarize_paper": {
        "template": """
        Here are the summaries of the sections of the paper. 
        Summaries: {summaries}
        Describe the paper in one sentence. 
        What are the main contributions of the paper? 
        Does the paper propose further research directions?
        {format_instructions}
        """,
        "response_schemas": [ResponseSchema(name="overall_summary",
                                            description="The summary of the paper in one sentence"),
                             ResponseSchema(name="contributions",
                                            description="The main contributions of the paper"),
                             ResponseSchema(name="further_research",
                                            description="Further research directions proposed by the paper")],
        "input_variables": ["summaries"],
        "output_variables": ["overall_summary", "contributions", "further_research"]
    },
    "aggregate_metrics": {
        "template": """
            Here are all the metrics and the corresponding descriptions that were described in the paper in different sections. 
            Metrics: {metrics_description}
            Aggregate all the metrics and descriptions. If some metrics are mentioned multiple times, mention them only once but pay attention to all descriptions.
            {format_instructions}
            """,
        "response_schemas": [ResponseSchema(name="metrics_aggregated",
                                            description="The aggregated metrics and descriptions")],
        "input_variables": ["metrics_description"],
        "output_variables": ["metrics_aggregated"]

    },
    "aggregate_mitigation": {
        "template": """
            Here are all the mitigation techniques and the corresponding descriptions that were described in the paper in the different sections. 
            Mitigation techniques: {mitigations_description}
            Aggregate all the techniques and descriptions. If some mitigation techniques are mentioned multiple times, mention them only once but pay attention to all descriptions.
            {format_instructions}
            """,
        "response_schemas": [ResponseSchema(name="mitigations_aggregated",
                                            description="The aggregated mitigation techniques and descriptions")],
        "input_variables": ["mitigations_description"],
        "output_variables": ["metrics_aggregated"]

    }

}
