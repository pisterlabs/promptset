import streamlit as st
#hints for debugging: https://awesome-streamlit.readthedocs.io/en/latest/vscode.html
#import plotly.express as px
import pandas as pd
#import matplotlib.pyplot as plt
#import pickle
import requests
#import openai
from utils import get_data, get_indicator_reason, filter_projects



#with open('../pred_lifeexp.pkl', 'rb') as file:
#    loaded_model = pickle.load(file)

st.title('Happy Graphs')

st.write("Group KMJ Do-Gooders proudly presents: Happy Graphs - Graphs which make us optimistic.")


'''Selection of further indicators for line charts'''
#ACTION: Search for an indicator by topic?

# Get the list of available indicators and countries
available_indicators = df.columns.tolist()[1:]
available_countries = df.columns.tolist()[1:]

# User selection
selected_indicator = st.selectbox("Select an indicator", available_indicators)
selected_countries = st.multiselect("Select countries", available_countries) #ACTION: make worldwide as a default
selected_year_range = st.slider("Select a year range", min_value=df['Year'].min(), max_value=df['Year'].max(),
                                value=(df['Year'].min(), df['Year'].max()), step=10)
selected_start_year, selected_end_year = selected_year_range

# Check if an indicator is selected
if not selected_indicator:
    st.text("Please choose an indicator to see its development.")
else:
    # Filter the data for selected countries and time period
    filtered_data = df[['Year'] + selected_countries]
    filtered_data = filtered_data[(filtered_data['Year'] >= selected_start_year) & (filtered_data['Year'] <= selected_end_year)]

    # Create a new figure and set the chart properties
    fig, ax = plt.subplots()
    ax.set_title(selected_indicator)
    ax.set_xlabel('Year')
    ax.set_ylabel(selected_indicator)
    ax.set_ylim(df[selected_indicator].min() - 10, df[selected_indicator].max() + 10)

    # Plot the data for selected countries
    for country in selected_countries:
        country_data = filtered_data[['Year', country]]
        line_color = st.color_picker(f"Select color for {country}", key=country)
        ax.plot(country_data['Year'], country_data[country], label=country, color=line_color)

        # Add a tooltip to show country and indicator value on hover
        tooltip = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                              bbox=dict(boxstyle="round", fc="white", edgecolor="gray"),
                              arrowprops=dict(arrowstyle="->"))
        tooltip.set_visible(False)

        def update_tooltip(event):
            if event.inaxes == ax:
                x = int(event.xdata)
                y = int(event.ydata)
                tooltip.xy = (x, y)
                tooltip.set_text(f"{country}: {y}")
                tooltip.set_visible(True)
                fig.canvas.draw_idle()
            else:
                tooltip.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", update_tooltip)
    # Adjust x-axis scale based on the selected time period
    ax.set_xlim(selected_start_year, selected_end_year)
    # Add a legend
    ax.legend()


#ACTION: Let automatically analyse whether it's getting better? Or do we only make good indicators available?


'''Explanation why it's changing'''
df_year_max = selected_end_year
df_year_min = selected_start_year

answer = get_indicator_reason(selected_indicator, df_year_max, df_year_min, selected_countries) #ACTION: selected countries still has worldwide as a default in utils. we might need to change it or see whether it works
print(answer)



'''You can do this to help'''
#ACTION: write what should be written here
# Example usage
charity_country = ['Afghanistan', 'India']
charity_title = ''
charity_region = ''
charity_theme_name = 'Education'

projects = filter_projects(charity_country, charity_title, charity_region, charity_theme_name)
filtered_projects = filter_projects(country=country, title=charity_title, region=charity_region, theme_name=charity_theme_name)

if filtered_projects:
    for project in filtered_projects:
        print("Project Title:", project['title'])
        # Print other project details
else:
    print("No data found for the specified filters.")

#See data
filtered_df #ACTION

#More interesting charts (combined dashboards)?

#spread the word
# api to automatically create posts