import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Initialize the OpenAI API key
openai.api_key = ""

class TwitterMetricsAnalyzer:
    def __init__(self, user_metrics_file, competitor_metrics_file):
        self.user_metrics = pd.read_csv(user_metrics_file)
        self.competitor_metrics = pd.read_csv(competitor_metrics_file)
    
    def clean_text(self, text):
        # Implement your text cleaning logic here
        pass
    
    def remove_urls(self, text):
        # Implement URL removal logic here
        pass

    def calculate_metric_gaps(self, user_username, competitor_usernames):
        metric_gaps = self.user_metrics.copy()
        metrics_to_compare = ['Followers', 'Likes', 'Retweets']
        for metric in metrics_to_compare:
            metric_gaps[f'{metric} Gap'] = self.user_metrics[metric] - self.competitor_metrics[metric].mean()
        return metric_gaps

    def generate_insights(self, metric_gaps_df, user_username, competitor_usernames):
        insights = []
        i = 0
        for index, row in metric_gaps_df.iterrows():
            username = row['Username']
            prompt = f"Generate insights for Twitter user @{username} to bridge the gap in metrics. Metrics gaps:\n"
            for metric in ['Followers Gap', 'Likes Gap', 'Retweets Gap']:
                gap = row[metric]
                prompt += f"- {metric}: {gap}\n"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.7,
            )
            generated_insight = response.choices[0].text.strip()
            insights.append({
                'Username': username,
                'Insight': generated_insight
            })
            time.sleep(15)
            if i >=10:
                break

        return insights

    def visualize_metric_gaps(self, metric_gaps_df, usernames_to_display):
        plt.figure(figsize=(12, 6))  
        gap_columns = [col for col in metric_gaps_df.columns if 'Gap' in col and col != 'Username']
        bar_width = 0.2
        x = np.arange(len(usernames_to_display))
        colors = sns.color_palette("husl", len(gap_columns))
        for i, gap_column in enumerate(gap_columns):
            selected_gaps = metric_gaps_df[metric_gaps_df['Username'].isin(usernames_to_display)]
            gap_data = selected_gaps[gap_column].tolist()[:len(usernames_to_display)]
            plt.bar(x + (i * bar_width), gap_data, width=bar_width, label=gap_column, color=colors[i])
        plt.xticks(x + (len(gap_columns) / 2) * bar_width, usernames_to_display, rotation=45)
        plt.title('Metric Gaps Comparison')
        plt.xlabel('Username')
        plt.ylabel('Gap Value')
        plt.legend(title='Gap Metric', loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    def analyze_twitter_metrics(self, user_username, competitor_usernames):
        metric_gaps_df = self.calculate_metric_gaps(user_username, competitor_usernames)
        self.visualize_metric_gaps(metric_gaps_df, [user_username] + competitor_usernames)
        insights_data = self.generate_insights(metric_gaps_df, user_username, competitor_usernames)
        return insights_data

if __name__ == '__main__':
    analyzer = TwitterMetricsAnalyzer('../data/user_metrics.csv', '../data/competitor_metrics.csv')
    user_username = 'Sam'
    competitor_usernames = ['Quincy', 'Finn']
    insights_data = analyzer.analyze_twitter_metrics(user_username, competitor_usernames)
    print(insights_data)
