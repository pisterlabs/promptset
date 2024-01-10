import openai
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, chi2_contingency

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from wordcloud import WordCloud, STOPWORDS

import nltk

nltk.data.path.append('nltk_data/')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
import category_encoders as ce


class DataViz():
    def __init__(self, dataloader):
        self.df = dataloader
        # self.df = pd.read_csv('train.csv')
        self.num_df = self.df.select_dtypes(include='number')
        self.cat_df = self.df.select_dtypes(exclude='number')
        self.date_cols = self.df.select_dtypes(
            include='datetime64[ns]').columns
        # sorry duplicate dulu
        self.integer_cols = self.df.select_dtypes(['int16', 'int32',
                                                   'int64']).columns
        self.numeric_cols = self.df.select_dtypes(
            ['int16', 'int32', 'int64', 'float16', 'float32',
             'float64']).columns
        self.float_cols = self.df.select_dtypes(
            ['float16', 'float32', 'float64']).columns
        # self.categorical_cols = self.df.select_dtypes(exclude=[
        #     'int16', 'int32', 'int64', 'float16', 'float32', 'float64'
        # ]).columns
        self.categorical_cols = self.df.select_dtypes(include=[
                'object'
            ]).columns

    def format_value(self, valu):
        '''
        This Code is used for formating Number Value
        '''
        if valu >= 1e9:
            return f"{valu/1e9:.1f} Bn"
        elif valu >= 1e6:
            return f"{valu/1e6:.1f} Mn"
        elif valu >= 1e3:
            return f"{valu/1e3:.1f} K"
        else:
            return f"{valu:.1f}"

    # Preprocess the text data
    def preprocess_text(self,
                        text,
                        standardize_text=False,
                        lemmatize_text=False,
                        remove_stopwords=False):
        if standardize_text:
            text = text.lower()

        words = word_tokenize(text)

        if lemmatize_text:
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]

        if remove_stopwords:
            custom_stopwords = set(stopwords.words("english"))
            words = [
                word for word in words
                if word not in custom_stopwords and word not in STOPWORDS
            ]

        return " ".join(words)

    def variable_selection(self,
                           selected_variables,
                           progress_bar,
                           threshold_dist_uniform=0.9,
                           alpha_chi2=0.05):

        progress_bar.progress(0.1)

        # Step 1: Split data into numerical and categorical variables
        numerical_vars = self.num_df.columns.tolist()
        categorical_vars = self.cat_df.columns.tolist()
        data = self.df.copy()
        variables_to_remove = []
        for var in numerical_vars:
            if self.df[var].value_counts(
                    normalize=True).max() >= threshold_dist_uniform:
                variables_to_remove.append(var)
            elif self.df[var].std(
            ) == 0:  # Check if standard deviation is zero (incremental)
                variables_to_remove.append(var)
            elif self.df[var].diff().dropna().unique().size == 1:
                # Check if the variable values are incremental
                variables_to_remove.append(var)

        numerical_vars = [
            x for x in numerical_vars if x not in variables_to_remove
        ]

        if selected_variables in numerical_vars:
            numerical_vars = [
                x for x in numerical_vars if x != selected_variables
            ]

            # Step 2: Check the number of numerical and categorical variables
            if len(numerical_vars) >= 5:
                # Step 3: Permutation Importance
                data[numerical_vars] = self.df[numerical_vars].fillna(0)                
                X = data[numerical_vars]
                y = data[selected_variables]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

                clf = RandomForestRegressor(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                progress_bar.progress(0.25)

                perm_importance = permutation_importance(clf, X_test, y_test)
                progress_bar.progress(0.5)

                feature_importance = perm_importance.importances_mean

                # Step 4: Select best 5 numerical and 5 categorical variables based on permutation score
                top_numerical_vars = np.array(numerical_vars)[np.argsort(
                    feature_importance)[-5:]]

                remaining_variables = list(top_numerical_vars)

            else:
                progress_bar.progress(0.25)
                remaining_variables = numerical_vars
                progress_bar.progress(0.5)

            if len(categorical_vars) >= 5:
                # Step 3: Permutation Importance
                data[categorical_vars] = self.df[categorical_vars].fillna('Missing Data') 
                X = data[categorical_vars]
                y = data[selected_variables]

                # Encode target variable using target encoding
                encoder = ce.TargetEncoder(cols=categorical_vars)
                X_encoded = encoder.fit_transform(X, y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.2, random_state=42)

                clf = RandomForestRegressor(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)

                perm_importance = permutation_importance(clf, X_test, y_test)
                progress_bar.progress(0.75)

                feature_importance = perm_importance.importances_mean

                # Step 4: Select best 5 numerical and 5 categorical variables based on permutation score
                top_categorical_vars = np.array(categorical_vars)[np.argsort(
                    feature_importance)[-5:]]

                remaining_variables += list(top_categorical_vars)
                progress_bar.progress(1)

            else:
                progress_bar.progress(0.75)
                remaining_variables += list(categorical_vars)
                progress_bar.progress(1)

            return remaining_variables

        elif selected_variables in categorical_vars:
            categorical_vars = [
                x for x in categorical_vars if x != selected_variables
            ]
            for var in numerical_vars:
                if self.df.groupby(selected_variables)[var].mean(
                ).value_counts(normalize=True).max() >= threshold_dist_uniform:
                    variables_to_remove.append(var)
                elif self.df.groupby(selected_variables)[var].std().std(
                ) == 0:  # Check if standard deviation is zero (incremental)
                    variables_to_remove.append(var)
                elif self.df.groupby(selected_variables)[var].mean().diff(
                ).dropna().unique().size == 1:
                    # Check if the variable values are incremental
                    variables_to_remove.append(var)

                numerical_vars = [
                    x for x in numerical_vars if x not in variables_to_remove
                ]

            if len(numerical_vars) >= 5:
                data[numerical_vars] = self.df[numerical_vars].fillna(0) 
                X = data[numerical_vars]
                y = data[selected_variables]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                progress_bar.progress(0.25)

                perm_importance = permutation_importance(clf, X_test, y_test)
                progress_bar.progress(0.5)

                feature_importance = perm_importance.importances_mean

                # Step 4: Select best 5 numerical and 5 categorical variables based on permutation score
                top_numerical_vars = np.array(numerical_vars)[np.argsort(
                    feature_importance)[-5:]]

                remaining_variables = list(top_numerical_vars)

            else:
                progress_bar.progress(0.25)
                remaining_variables = list(numerical_vars)
                progress_bar.progress(0.5)

            # Step 6: Check chi2 values for categorical variables
            chi2_results = []
            for var in categorical_vars:
                contingency_table = pd.crosstab(self.df[var],
                                                self.df[selected_variables])
                chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
                chi2_results.append((var, chi2_stat, p_val))
                progress_bar.progress(0.75)

            # Sort chi2 results based on p-value
            chi2_results.sort(key=lambda x: x[2])

            # Remove variables with p-value >= alpha_chi2
            top_categorical_vars = []
            for var, chi2_stat, p_val in chi2_results:
                if p_val < alpha_chi2 and len(top_categorical_vars) < 5:
                    top_categorical_vars.append(var)
            remaining_variables += list(top_categorical_vars)
            progress_bar.progress(1)

            return remaining_variables

        else:
            return numerical_vars + categorical_vars

    def visualization(self):
        st.markdown(
            "<h1 style='text-align: center; color: #4F4F4F;'>Automated Data Visualizations (by Ulikdata)</h1>",
            unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            'Custom Business Viz', 'Custom Statistical Viz', 'Custom Text Viz',
            '1-Click Analyze!', '1-Click Summarize!'
        ])

        with tab1:
            if len(self.cat_df.columns) < 1:
                st.write("Your Data do not have any Categorical Values")
            else:
                ## BAR PLOT
                c1 = st.container()
                desc_col, img_col = c1.columns(2)

                desc_col.subheader("Bar Plot")

                # Widget selection for user input
                VAR = desc_col.selectbox('Bar Plot variable:',
                                         self.cat_df.columns.tolist())

                dc1, dc2 = desc_col.columns(2)

                selected_aggregator = dc1.selectbox(
                    "Select Aggregator:",
                    ['None'] + self.num_df.columns.tolist())
                aggregation_options = [
                    'sum', 'mean', 'median', 'No Aggregation'
                ]
                selected_aggregation = dc2.selectbox(
                    "Select Aggregation Function:", aggregation_options
                    if selected_aggregator != 'None' else ['No Aggregation'])
                selected_total_category = dc1.selectbox(
                    "Select # of Top Categories:", ['5', '10', '20'])
                sorting_options = [
                    'Ascending', 'Descending', 'None (Alphabetical Order)'
                ]
                selected_sorting = dc2.selectbox(
                    "Select Bar Plot Sorting Order:", sorting_options)

                data = self.df.copy()

                # Check if the number of unique categories is greater than N
                if data[VAR].nunique() > int(selected_total_category):
                    # Get the top N categories
                    top_categories = data[VAR].value_counts().nlargest(
                        int(selected_total_category)).index.tolist()
                    # Create a new categorical variable that groups top N categories and labels the rest as 'Others'
                    data[f'{VAR}NEW'] = data[VAR].apply(
                        lambda x: x if x in top_categories else 'Others')
                else:
                    data[f'{VAR}NEW'] = data[VAR]

                # Apply selected aggregation function or counting
                if selected_aggregation == 'sum':
                    aggregated_data = data.groupby(
                        f'{VAR}NEW')[selected_aggregator].sum()
                    LABEL_PLOT = f'Sum of {selected_aggregator} by {VAR}'
                elif selected_aggregation == 'mean':
                    aggregated_data = data.groupby(
                        f'{VAR}NEW')[selected_aggregator].mean()
                    LABEL_PLOT = f'Mean of {selected_aggregator} by {VAR}'
                elif selected_aggregation == 'median':
                    aggregated_data = data.groupby(
                        f'{VAR}NEW')[selected_aggregator].median()
                    LABEL_PLOT = f'Median of {selected_aggregator} by {VAR}'
                else:
                    aggregated_data = data[f'{VAR}NEW'].value_counts()
                    LABEL_PLOT = f'Count of {VAR}'

                # Sort the data based on the selected sorting order
                if selected_sorting == 'Ascending':
                    aggregated_data = aggregated_data.sort_values(
                        ascending=True)
                elif selected_sorting == 'Descending':
                    aggregated_data = aggregated_data.sort_values(
                        ascending=False)
                else:
                    aggregated_data = aggregated_data.sort_index()

                # Insight 1: Identify the category with the highest value
                max_category = aggregated_data.idxmax()
                max_value = aggregated_data.max()

                # Create an aggregated bar plot using Plotly Express
                fig = px.bar(x=aggregated_data.index.astype(str),
                             y=aggregated_data,
                             labels={
                                 'y': LABEL_PLOT.replace(f' by {VAR}', ''),
                                 'x': VAR
                             },
                             title=LABEL_PLOT)

                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    desc_col.write(
                        f"The category with the highest value is '{max_category}' with a value of {max_value:.2f}."
                    )

                with img_col:
                    # Create a plot using Plotly
                    img_col.plotly_chart(fig)

                st.write('---')

                ## PIE PLOT
                c2 = st.container()
                desc_col, img_col = c2.columns(2)

                desc_col.subheader("Pie Chart")
                # Widget selection for user input
                VAR2 = desc_col.selectbox('Pie Chart variable:',
                                          self.cat_df.columns.tolist())
                selected_style = desc_col.selectbox("Select Pie Chart Style:",
                                                    ['Pie', 'Doughnut'])

                dc1, dc2 = desc_col.columns(2)
                selected_aggregator = dc1.selectbox(
                    "Select Aggregator for Pie Chart:",
                    ['None'] + self.num_df.columns.tolist())
                aggregation_options = ['Sum', 'No Aggregation']
                selected_aggregation = dc2.selectbox(
                    "Select Aggregation Function  for Pie Chart:",
                    aggregation_options
                    if selected_aggregator != 'None' else ['No Aggregation'])

                # selected_total_category = dc1.selectbox(
                #     "Select # of Top Categories for Pie Chart:", ['5', '10', '20'])

                # Check if the number of unique categories is greater than 3
                if data[VAR2].nunique() > 3:
                    # Get the top N categories
                    top_categories = data[VAR2].value_counts().nlargest(
                        3).index.tolist()
                    # Create a new categorical variable that groups top N categories and labels the rest as 'Others'
                    data[f'{VAR2}NEW'] = data[VAR2].apply(
                        lambda x: x if x in top_categories else 'Others')
                else:
                    data[f'{VAR2}NEW'] = data[VAR2]

                # Apply selected aggregation function or counting
                if selected_aggregation == 'Sum':
                    aggregated_data = data.groupby(
                        f'{VAR2}NEW')[selected_aggregator].sum()
                    LABEL_PLOT = f'Sum of {selected_aggregator} by {VAR2}'
                else:
                    aggregated_data = data[f'{VAR2}NEW'].value_counts()
                    LABEL_PLOT = f'Percent Count of {VAR2}'

                # Calculate the total aggregated value
                total_aggregated_value = aggregated_data.sum()

                # Create an aggregated bar plot using Plotly Express
                if selected_style == 'Pie':
                    fig = px.pie(values=aggregated_data,
                                 names=aggregated_data.index.astype(str),
                                 title=LABEL_PLOT)
                else:
                    fig = px.pie(values=aggregated_data,
                                 names=aggregated_data.index.astype(str),
                                 hole=0.45,
                                 title=LABEL_PLOT)
                    # Add annotation for the total value in the center of the donut chart
                    fig.add_annotation(
                        text=
                        f"Total: {self.format_value(total_aggregated_value)}",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=20))

                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    # Insight 1: Aggregated value of each category
                    desc_col.write("Aggregated Value of Each Category:")
                    for category, value in aggregated_data.items():
                        percentage = (value / total_aggregated_value) * 100
                        desc_col.write(
                            f"- Category '{category}': {value:.2f} ({percentage:.2f}%)"
                        )

                with img_col:
                    # Create a plot using Plotly
                    img_col.plotly_chart(fig)

                st.write('---')

                # ini error kalo ngga ada kolom numerik yg has no null
                try:
                    ## LINE CHART
                    c3 = st.container()
                    desc_col, img_col = c3.columns(2)
    
                    desc_col.subheader("Line Chart")
                    # Widget selection for user input
                    VAR3 = desc_col.selectbox(
                        'Line Chart Horizontal Variable (Date / Category):',
                        self.cat_df.columns.tolist())
                    VAR4 = desc_col.selectbox(
                        'Line Chart Vertical Variable (Value):',
                        self.num_df.columns.tolist())
                    VAR5 = desc_col.selectbox(
                        'Line Chart Categorical Variable (Hue):', ['None'] +
                        [x for x in self.cat_df.columns.tolist() if x != VAR3])
    
                    dc1, dc2 = desc_col.columns(2)
                    # Streamlit checkbox for smooth line option
                    smooth_line = dc1.checkbox("Smooth Line")
    
                    # Streamlit checkbox for wide format data
                    wide_format = dc2.checkbox("Wide Format Data")
    
                    if VAR5 != 'None':
                        data_line = data[[VAR3, VAR4, VAR5]].copy()
                        # Check if the number of unique categories is greater than 3
                        if data_line[VAR5].nunique() > 3:
                            # Get the top N categories
                            top_categories = data_line[VAR5].value_counts().nlargest(
                                3).index.tolist()
                            # Create a new categorical variable that groups top N categories and labels the rest as 'Others'
                            data_line[VAR5] = data_line[VAR5].apply(
                                lambda x: x if x in top_categories else 'Others')
                    else:
                        data_line = data[[VAR3, VAR4]].copy()
    
                    df_to_plot = data_line.sort_values(
                        by=VAR4,
                        ascending=True).reset_index().drop(columns=['index'])
    
                    if wide_format:
                        # Reshape the data for Plotly Express
                        df_to_plot = df_to_plot.stack().reset_index()
    
                    try:
                        # Calculate insights
                        max_value = df_to_plot[VAR4].max()
                        min_value = df_to_plot[VAR4].min()
                        max_date = df_to_plot.loc[df_to_plot[VAR4] ==
                                                  max_value][VAR3].values[0]
                        min_date = df_to_plot.loc[df_to_plot[VAR4] ==
                                                  min_value][VAR3].values[0]
    
                        # Create a line chart to visualize monthly transaction trends
                        if VAR5 != 'None':
                            fig = px.line(
                                df_to_plot,
                                x=VAR3,
                                y=VAR4,
                                color=VAR5,
                                title=f"{VAR4} Trends by {VAR3}",
                                line_shape='spline' if smooth_line else 'linear')
                        else:
                            fig = px.line(
                                df_to_plot,
                                x=VAR3,
                                y=VAR4,
                                title=f"{VAR4} Trends by {VAR3}",
                                line_shape='spline' if smooth_line else 'linear')
    
                        with desc_col:
                            desc_col.write('##')
                            desc_col.markdown("#### :blue[Insight!]")
                            # Display insights
                            desc_col.write(
                                f"Maximum Value: {max_value} on {max_date}")
                            desc_col.write(
                                f"Minimum Value: {min_value} on {min_date}")
    
                            # Calculate trend direction
                            last_value = df_to_plot.iloc[-1][VAR4]
                            first_value = df_to_plot.iloc[0][VAR4]
                            TREND = "Stagnant"
                            if last_value > first_value:
                                TREND = "Positive"
                            elif last_value < first_value:
                                TREND = "Negative"
    
                            # Display trend direction insight
                            desc_col.write(
                                f"Trend Direction: {TREND.capitalize()} trend")
    
                        with img_col:
                            # Create a plot using Plotly
                            img_col.plotly_chart(fig)
    
                    except:
                        with desc_col:
                            desc_col.write('##')
                            desc_col.markdown(
                                "Seems your data is not in a correct format, please uncheck Wide Data Format or crosscheck your data format"
                            )
    
                    st.write('---')
                except:
                    pass

                ## PIVOT TABLE
                c4 = st.container()
                desc_col, img_col = c4.columns(2)

                desc_col.subheader("Pivot Table")

                # Streamlit selectbox for rows, columns, and values
                row_options = [None] + self.cat_df.columns.tolist()
                col_options = [None] + self.cat_df.columns.tolist()
                value_options = [None] + self.num_df.columns.tolist()

                selected_rows = desc_col.selectbox("Select Rows:", row_options)
                selected_columns = desc_col.selectbox("Select Columns:",
                                                      col_options)
                if selected_rows is not None and selected_columns is not None:
                    selected_values = desc_col.selectbox(
                        "Select Values:", value_options)
                else:
                    selected_values = desc_col.selectbox(
                        "Select Values:", [None])
                if selected_values is None:
                    selected_value_adjustment = desc_col.selectbox(
                        "Select Value Adjustment:", [
                            None, 'percentage by rows',
                            'percentage by columns', 'percentage by all'
                        ])
                else:
                    selected_value_adjustment = desc_col.selectbox(
                        "Select Value Adjustment:", [
                            'sum', 'average', 'percentage by rows',
                            'percentage by columns', 'percentage by all'
                        ])

                # Create a custom pivot table using pandas
                if selected_rows is None and selected_columns is None:
                    pivot_table = pd.DataFrame({'Count': [data.shape[0]]})
                    pivot_table['Count'] = pivot_table['Count'].map(
                        self.format_value)
                elif selected_rows is not None and selected_columns is None:
                    pivot_table = data[selected_rows].value_counts(
                    ).reset_index()
                    pivot_table.columns = [selected_rows, 'Count']
                    pivot_table['Count'] = pivot_table['Count'].map(
                        self.format_value)
                elif selected_rows is None and selected_columns is not None:
                    pivot_table = data[selected_columns].value_counts(
                    ).reset_index()
                    pivot_table.columns = [selected_columns, 'Count']
                    pivot_table['Count'] = pivot_table['Count'].map(
                        self.format_value)
                elif selected_rows is not None and selected_columns is not None and selected_values is None:
                    if selected_value_adjustment == 'percentage by columns':
                        pivot_table = pd.crosstab(
                            index=data[selected_rows],
                            columns=data[selected_columns])

                        column_totals = pivot_table.sum(axis=0)

                        # Calculate percentages by columns
                        pivot_table = pivot_table.div(column_totals,
                                                      axis=1) * 100

                        sum_all = pivot_table.sum().sum()

                        column_totals = pivot_table.sum(axis=0)

                        # Create a DataFrame for the column totals
                        column_totals_df = pd.DataFrame(
                            [column_totals],
                            columns=pivot_table.columns,
                            index=['All'])

                        pivot_table = pd.concat(
                            [pivot_table, column_totals_df])

                        # Calculate row and column totals
                        row_totals = pivot_table.sum(axis=1) * 100 / sum_all

                        # Create a DataFrame for the row totals
                        row_totals_df = pd.DataFrame(row_totals,
                                                     columns=['All'])

                        # Concatenate the crosstab table, row totals, and column totals
                        pivot_table = pd.concat([pivot_table, row_totals_df],
                                                axis=1)
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    elif selected_value_adjustment == 'percentage by rows':
                        pivot_table = pd.crosstab(
                            index=data[selected_rows],
                            columns=data[selected_columns])

                        row_totals = pivot_table.sum(axis=1)

                        # Calculate percentages by rows
                        pivot_table = pivot_table.div(pivot_table.sum(axis=1),
                                                      axis=0) * 100

                        sum_all = pivot_table.sum().sum()

                        row_totals = pivot_table.sum(axis=1)

                        # Create a DataFrame for the row totals
                        row_totals_df = pd.DataFrame(row_totals,
                                                     columns=['All'])
                        pivot_table = pd.concat([pivot_table, row_totals_df],
                                                axis=1)

                        # Calculate row and column totals

                        column_totals = pivot_table.sum(axis=0) * 100 / sum_all

                        # Create a DataFrame for the column totals
                        column_totals_df = pd.DataFrame(
                            [column_totals],
                            columns=pivot_table.columns,
                            index=['All'])

                        # Concatenate the crosstab table, row totals, and column totals
                        pivot_table = pd.concat(
                            [pivot_table, column_totals_df])

                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    elif selected_value_adjustment == 'percentage by all':
                        pivot_table = pd.crosstab(
                            index=data[selected_rows],
                            columns=data[selected_columns])

                        all_values = pivot_table.values.sum()

                        # Calculate row and column totals
                        row_totals = pivot_table.sum(axis=1)
                        column_totals = pivot_table.sum(axis=0)

                        # Create a DataFrame for the column totals
                        column_totals_df = pd.DataFrame(
                            [column_totals],
                            columns=pivot_table.columns,
                            index=['All'])

                        # Create a DataFrame for the row totals
                        row_totals_df = pd.DataFrame(row_totals,
                                                     columns=['All'])

                        # Concatenate the crosstab table, row totals, and column totals
                        crosstab_with_totals = pd.concat(
                            [pivot_table, column_totals_df])
                        crosstab_with_totals = pd.concat(
                            [crosstab_with_totals, row_totals_df], axis=1)
                        crosstab_with_totals = crosstab_with_totals / all_values * 100

                        # Format the percentages
                        pivot_table = crosstab_with_totals.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    else:
                        pivot_table = pd.crosstab(
                            index=data[selected_rows],
                            columns=data[selected_columns],
                            margins=True)
                else:
                    if selected_value_adjustment == 'sum':
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='sum',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table.applymap(self.format_value)
                    elif selected_value_adjustment == 'average':
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='mean',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table.applymap(self.format_value)
                    elif selected_value_adjustment == 'percentage by rows':
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='sum',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table.div(pivot_table.iloc[:, -1],
                                                      axis=0) * 100
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    elif selected_value_adjustment == 'percentage by columns':
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='sum',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table.div(pivot_table.iloc[-1, :],
                                                      axis=1) * 100
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")
                    else:
                        pivot_table = pd.pivot_table(self.df,
                                                     index=selected_rows,
                                                     columns=selected_columns,
                                                     values=selected_values,
                                                     aggfunc='sum',
                                                     fill_value=0,
                                                     margins=True)
                        pivot_table = pivot_table / pivot_table.iloc[
                            -1, :-1].sum() * 100
                        pivot_table = pivot_table.applymap(
                            lambda x: f"{x:.2f}%" if not np.isnan(x) else "-")

                # Display the pivot table using Streamlit
                img_col.dataframe(pivot_table, use_container_width=True)

        with tab2:
            if len(self.num_df.columns) < 1:
                st.write("Your Data do not have any Numerical Values")
            else:
                ## HISTOGRAM / DENSITY PLOT
                c1 = st.container()
                desc_col, img_col = c1.columns(2)

                desc_col.subheader("Histogram/Density Plot")

                # Widget selection for user input
                VAR = desc_col.selectbox('Histogram/density variable:',
                                         self.num_df.columns.tolist())
                BIN = desc_col.selectbox('Number of Bins:', ['10', '25', '50'])
                TYPE = desc_col.selectbox('Type:', ['Histogram', 'Density'])

                NORM = None if TYPE == 'Histogram' else 'probability density'
                DENSITY = False if TYPE == 'Histogram' else True

                # Create Plot using user input
                fig = px.histogram(self.num_df[VAR],
                                   nbins=int(BIN),
                                   histnorm=NORM)
                fig.update_traces(marker=dict(
                    line=dict(color='black', width=1)))
                f = fig.full_figure_for_development(warn=False)

                xbins = f.data[0].xbins
                plotbins = list(
                    np.arange(start=xbins['start'],
                              stop=xbins['end'] + xbins['size'],
                              step=xbins['size']))
                data_for_hist = [val for val in f.data[0].x if val is not None]
                counts, bins = np.histogram(data_for_hist,
                                            bins=plotbins,
                                            density=DENSITY)
                IDX = np.argmax(counts)
                s, e = bins[IDX], bins[IDX + 1]
                COUNTS = max(counts)
                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    desc_col.markdown(
                        f"**Highest Bins**\t: {s:.2f} to {e-0.1:.2f}")
                    desc_col.markdown(
                        f"**Highest Values**\t\t: {counts[IDX]:.2f}")

                with img_col:
                    # Create a plot using Plotly
                    img_col.plotly_chart(fig)

                st.write('---')

                ## BOX PLOT
                c2 = st.container()
                desc_col, img_col = c2.columns(2)

                desc_col.subheader("Box Plot")

                # Widget selection for user input
                VAR2 = desc_col.selectbox('Box Plot Variable:',
                                          self.num_df.columns)

                # Create Plot using user input
                fig = px.box(self.num_df[VAR2])

                # Calculate outliers using the Interquartile Range (IQR) method
                q1 = np.percentile(self.num_df[VAR2].dropna(), 25)
                q3 = np.percentile(self.num_df[VAR2].dropna(), 75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                stats_table = self.num_df[VAR2].describe()
                stats_table['Lower Bound'] = lower_bound
                stats_table['Upper Bound'] = upper_bound

                selected_indexes = [
                    'Lower Bound', '25%', '50%', '75%', 'Upper Bound'
                ]
                selected_desc_stats = stats_table.loc[selected_indexes]

                upper_outliers = [
                    val for val in self.num_df[VAR2] if val > upper_bound
                ]
                lower_outliers = [
                    val for val in self.num_df[VAR2] if val < lower_bound
                ]

                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    table, outlier_desc = st.columns([1, 2])
                    table.dataframe(selected_desc_stats)
                    if len(upper_outliers) > 1:
                        outlier_desc.markdown(
                            f"**Variables {VAR2} have {len(upper_outliers)} upper outliers ranged from:**\n {min(upper_outliers)} to {max(upper_outliers)}"
                        )
                    elif len(upper_outliers) == 1:
                        outlier_desc.markdown(
                            f"**Variables {VAR2} have {len(upper_outliers)} upper outlier:**\n {min(upper_outliers)}"
                        )
                    if len(lower_outliers) > 1:
                        outlier_desc.markdown(
                            f"**Variables {VAR2} have {len(lower_outliers)} upper outliers ranged from:**\n {min(lower_outliers)} to {max(lower_outliers)}"
                        )
                    elif len(lower_outliers) == 1:
                        outlier_desc.markdown(
                            f"**Variables {VAR2} have {len(lower_outliers)} upper outlier:**\n {min(lower_outliers)}"
                        )
                    if len(upper_outliers) == 0 and len(lower_outliers) == 0:
                        outlier_desc.markdown("No outliers found.")

                with img_col:
                    # Create a plot using Plotly
                    img_col.plotly_chart(fig)

                st.write('---')

                ## Percentile PLOT
                c3 = st.container()
                desc_col, img_col = c3.columns(2)
                # Create a Streamlit app
                desc_col.subheader("Percentile Plot")

                # Widget selection for user input
                VAR3 = desc_col.selectbox('Percentile Variable:',
                                          self.num_df.columns)

                # Select a percentile for analysis
                selected_percentile = desc_col.slider("Select Percentile:", 1,
                                                      99, 80)
                percentile_value = np.percentile(self.num_df[VAR3].dropna(),
                                                 selected_percentile)
                percentiles = np.arange(1, 101)
                percentile_values = np.percentile(self.num_df[VAR3].dropna(),
                                                  percentiles)

                # Create a line plot using Plotly
                fig = px.line(x=percentiles,
                              y=percentile_values,
                              labels={
                                  'x': 'Percentile',
                                  'y': 'Value'
                              })

                # Add horizontal and vertical cross indicators with semi-transparent color
                fig.add_shape(type="line",
                              x0=selected_percentile,
                              x1=selected_percentile,
                              y0=0,
                              y1=max(self.num_df[VAR3].dropna()),
                              line=dict(color="rgba(255, 0, 0, 0.4)",
                                        width=2,
                                        dash='dash'))

                fig.add_shape(type="line",
                              x0=1,
                              x1=100,
                              y0=percentile_value,
                              y1=percentile_value,
                              line=dict(color="rgba(255, 0, 0, 0.4)",
                                        width=2,
                                        dash='dash'))

                with desc_col:
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    desc_col.markdown(
                        f"{selected_percentile}th Percentile Value: {percentile_value:.2f}, this mean that {selected_percentile}% of data in {VAR3} are more than {percentile_value:.2f}"
                    )

                with img_col:
                    # Display the line plot using Streamlit
                    img_col.plotly_chart(fig)

                st.write('---')

            if len(self.num_df.columns) < 2:
                st.write("Your Data only have Single Numerical Values")
            else:
                ## Scatter PLOT
                c1 = st.container()
                desc_col, img_col = c1.columns(2)
                # Create a Streamlit app
                desc_col.subheader("Scatter Plot")

                # Widget selection for user input
                VARX = desc_col.selectbox('Scatter Variable X:',
                                          self.num_df.columns)
                VARY = desc_col.selectbox(
                    'Scatter Variable Y:',
                    [x for x in self.num_df.columns if x != VARX])

                # Calculate correlation coefficient
                x_data = self.num_df[VARX].dropna()
                y_data = self.num_df[VARY].dropna()
                min_length = min(len(x_data), len(y_data))
                x_data = x_data[:min_length]
                y_data = y_data[:min_length]
                correlation = np.corrcoef(x_data, y_data)[0, 1]

                # Create a scatter plot using Plotly Express
                fig = px.scatter(x=self.num_df[VARX],
                                 y=self.num_df[VARY],
                                 labels={
                                     'x': VARX,
                                     'y': VARY
                                 })

                # Annotate the scatter plot with correlation value
                fig.update_layout(annotations=[
                    dict(x=0.5,
                         y=1.15,
                         showarrow=False,
                         text=f'Correlation: {correlation:.2f}',
                         xref="paper",
                         yref="paper")
                ])

                with desc_col:
                    # Determine the strength and direction of the correlation
                    if correlation > 0.7:
                        RELATION = 'Strong Positive'
                    elif correlation > 0.5 and correlation <= 0.7:
                        RELATION = 'Relatively Positive'
                    elif correlation < -0.5 and correlation >= -0.7:
                        RELATION = 'Relatively Negative'
                    elif correlation < -0.7:
                        RELATION = 'Strong Negative'
                    else:
                        RELATION = 'Weak'

                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    desc_col.markdown(
                        f"Linear Correlation between {VARX} and {VARY} is {RELATION}"
                    )

                with img_col:
                    # Display the line plot using Streamlit
                    img_col.plotly_chart(fig)

                st.write('---')

                ## Heatmap Correlation PLOT
                c2 = st.container()
                desc_col, img_col = c2.columns(2)
                # Create a Streamlit app
                desc_col.subheader("Correlation Plot")

                # Selectbox for correlation method
                selected_corr_method = desc_col.selectbox(
                    "Select Correlation Method:",
                    ['pearson', 'kendall', 'spearman'])

                # Selectbox to choose target variable
                selected_target = desc_col.selectbox(
                    "Select Target Variable for Correlation Insight:",
                    self.num_df.columns)
                selected_colorscale = desc_col.selectbox(
                    "Select Colorscale:",
                    ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'])

                # Calculate the correlation matrix
                corr_matrix = self.num_df.corr(method=selected_corr_method)

                # Create a heatmap using Plotly
                fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                                x=corr_matrix.columns,
                                                y=corr_matrix.index,
                                                colorscale=selected_colorscale,
                                                text=corr_matrix.round(2)))

                # Customize the heatmap layout
                fig.update_layout(title="Correlation Heatmap",
                                  xaxis_title="Features",
                                  yaxis_title="Features",
                                  width=600,
                                  height=600)
                with desc_col:
                    # Write insights about strong correlations with the selected target variable
                    desc_col.write('##')
                    desc_col.markdown("#### :blue[Insight!]")
                    NO_INSIGHT = True
                    THRESHOLD = 0.7
                    for variable in self.num_df.columns:
                        if variable != selected_target and abs(
                                corr_matrix.loc[selected_target,
                                                variable]) > THRESHOLD:
                            NO_INSIGHT = False
                            desc_col.write(
                                f"Strong correlation between {selected_target} and {variable}: {corr_matrix.loc[selected_target, variable]:.2f}"
                            )
                    if NO_INSIGHT:
                        desc_col.write(
                            f"No variables have strong correlation with {selected_target}"
                        )
                with img_col:
                    # Display the heatmap using Streamlit
                    img_col.plotly_chart(fig)

        with tab3:
            if len(self.cat_df.columns.tolist()) != 0:
                ## WORD CLOUDS
                st.subheader("Word Cloud!")
                c1 = st.container()
                desc_col, img_col = c1.columns(2)
                # User input options
                ngrams = desc_col.selectbox("Select n-grams", [1, 2, 3])
                variables = desc_col.selectbox(
                    "Select Text Variables to be analyzed",
                    self.cat_df.columns.tolist())
                standardize_text = desc_col.checkbox(
                    "Standardize text (lowercase)")
                lemmatize_text = desc_col.checkbox("Lemmatize text")
                remove_stopwords = desc_col.checkbox("Remove stopwords")
    
                processed_text = self.df.apply(lambda row: self.preprocess_text(
                    row[variables], standardize_text, lemmatize_text,
                    remove_stopwords),
                                               axis=1)
    
                # Your existing code for vectorizing and calculating word frequencies
                note_text = None
                if all(len(text.split()) < ngrams for text in processed_text):
                    ngrams = max(len(text.split()) for text in processed_text)
                    note_text = f"Using {ngrams}-grams because records have at max {ngrams} word."
                    desc_col.write(note_text)
                    
                    
                vectorizer = CountVectorizer(ngram_range=(ngrams, ngrams))
                bag_of_words = vectorizer.fit_transform(processed_text)
                sum_words = bag_of_words.sum(axis=0)
                words_freq = [(word, sum_words[0, idx])
                              for word, idx in vectorizer.vocabulary_.items()]
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
                # Generating wordcloud
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color="white").generate_from_frequencies(
                        dict(words_freq[:20]))
    
                # Display the word cloud using Matplotlib within Streamlit
                img_col.write("Generated Word Cloud:")
                img_col.image(wordcloud.to_array(), use_column_width=True)
    
                st.write('---')

        with tab4:
            # Membuat peta warna
            color_map = {
                "Olive Green": "#808000",
                "Moss Green": "#8A9A5B",
                "Sage": "#B2AC88",
                "Taupe": "#483C32",
                "Beige": "#F5F5DC",
                "Burnt Sienna": "#E97451",
                "Terracotta": "#E2725B",
                "Umber": "#635147",
                "Default Plotly": "plotly"  # Default Plotly color
            }

            color_palettes = {
                "Blue Shades": "Blues",
                "Red Shades": "Reds",
                "Green Shades": "Greens",
                "Grayscale": "Greys",
                "Hot (Red-Yellow-Black)": "hot",
                "Cool (Blue-Cyan-Purple)": "YlGnBu",
                "Picnic (Green-Yellow-Red)": "Picnic",
                "Portland (Blue-Purple-Red)": "Portland",
                "Jet (Blue-Cyan-Yellow-Red)": "Jet",
                "Default Plotly": "plotly"  # Default Plotly color scale
            }

            color_palettes_discrete = {
                "Earth Tones":
                ["#a52a2a", "#654321", "#c2b280", "#808000", "#6a5acd"],
                "Cool": ["#99e6e6", "#e6e6ff", "#a3d1ff"],
                "Warm": ["#ffdb58", "#ffb366", "#ff7f50"],
                "Sunset":
                ["#FF4500", "#FF6347", "#FF7F50", "#FF8C00", "#FFA500"],
                "Ocean":
                ["#1E90FF", "#20B2AA", "#4682B4", "#5F9EA0", "#00CED1"],
                "Forest":
                ["#228B22", "#006400", "#2E8B57", "#3CB371", "#32CD32"],
                "Berry":
                ["#8B0000", "#B22222", "#DC143C", "#FF0000", "#FF4500"],
                "Pastel":
                ["#FFDAB9", "#E6E6FA", "#FFFACD", "#D8BFD8", "#F0E68C"],
                "Neon":
                ["#FF1493", "#00FF00", "#FFD700", "#FF69B4", "#00BFFF"],
                "Monochrome":
                ["#2F4F4F", "#708090", "#778899", "#B0C4DE", "#F5F5F5"],
                "Default Plotly":
                "plotly"  # Default Plotly color scale
                # ... you can further expand and add more palettes as needed
            }

            chart_width = 300  # width of the chart to fit within the column
            chart_height = 400  # height of the chart

            if len(self.float_cols) != 0:
                # Histogram
                st.write("## Histogram for Numeric Columns")
                col1, col2, col3 = st.columns(3)
    
                # Membiarkan pengguna memilih warna dengan default pilihan
                hist_color_choice = col1.selectbox(
                    "Choose color for histogram bars:",
                    list(color_map.keys()),
                    index=list(color_map.keys()).index("Default Plotly"))
    
                edge_color_choice = col2.selectbox(
                    "Choose color for histogram bar edges:",
                    list(color_map.keys()),
                    index=list(color_map.keys()).index(
                        "Default Plotly")  # Default ke "Umber"
                )
    
                hist_color = color_map[hist_color_choice]
                edge_color = color_map[edge_color_choice]
    
                # Initialize the columns
                left_col, center_col, right_col = st.columns(3)
                columns = [left_col, center_col, right_col]
                chart_col_idx = 0  # counter to keep track of columns
    
                for col in self.float_cols:
                    if hist_color == 'plotly':
                        fig = px.histogram(self.df,
                                           x=col,
                                           marginal="box",
                                           nbins=40,
                                           title=f'Histogram of {col}',
                                           width=chart_width,
                                           height=chart_height)
    
                    else:
                        fig = px.histogram(self.df,
                                           x=col,
                                           marginal="box",
                                           nbins=40,
                                           title=f'Histogram of {col}',
                                           color_discrete_sequence=[hist_color],
                                           width=chart_width,
                                           height=chart_height)
                        # fig.update_traces(marker=dict(
                        # color=hist_color, line=dict(color=edge_color, width=1)))
                        fig.update_traces(marker=dict(color=hist_color))
    
                    if edge_color == 'plotly':
                        fig.update_traces(marker=dict(line=dict(width=1)))
                    else:
                        fig.update_traces(marker=dict(
                            line=dict(color=edge_color, width=1)))
    
                    # Check if fig is a tuple and extract the figure if it is.
                    if isinstance(fig, tuple):
                        fig = fig[0]
    
                    fig.update_layout(title={
                        'font': {
                            'size': 12
                        }  # Increase font size for title
                    })
    
                    columns[chart_col_idx % 3].plotly_chart(fig)
    
                    # # Calculate skewness and kurtosis
                    # col_skewness = skew(self.df[col])
                    # col_kurtosis = kurtosis(self.df[col])
    
                    # # Determine the distribution type based on skewness and kurtosis
                    # if abs(col_skewness) < 0.5 and abs(col_kurtosis) < 0.5:
                    #     distribution_type = "approximately normal"
                    # elif abs(col_skewness) > 1:
                    #     distribution_type = "highly skewed"
                    # elif abs(col_kurtosis) > 0.5:
                    #     distribution_type = "has heavier tails than a normal distribution"
                    # else:
                    #     distribution_type = "moderately skewed"
    
                    # columns[chart_col_idx % 3].markdown(
                    #     f"<sub>The distribution of {col} is {distribution_type}.</sub>",
                    #     unsafe_allow_html=True)
    
                    chart_col_idx += 1

            if len(self.numeric_cols) != 0:
                st.write("## Scatter plot with Regression Line")
    
                # Kolom dropdown untuk memilih jumlah plot
                num_of_plots = st.selectbox("Choose number of plots to display:",
                                            list(range(1, 11)),
                                            index=1,
                                            key="num_of_plot")  # Default ke 10
    
                # Filter kolom numerik yang uniq valuenya di atas 10
                filtered_cols = [
                    col for col in self.numeric_cols if self.df[col].nunique() > 10
                ]
    
                # Menghitung matriks korelasi
                correlation_matrix = self.df[filtered_cols].corr()
    
                # Membuat daftar dari pasangan kolom dan nilai korelasinya
                from itertools import combinations
    
                pairs_correlation = []
                for col1, col2 in combinations(filtered_cols, 2):
                    pairs_correlation.append(
                        ((col1, col2), abs(correlation_matrix.loc[col1, col2])))
    
                # Mengurutkan pasangan berdasarkan nilai korelasi (absolut) tertinggi
                sorted_pairs = sorted(pairs_correlation,
                                      key=lambda x: x[1],
                                      reverse=True)
    
                # Mengambil pasangan dengan korelasi tertinggi sesuai dengan pilihan pengguna
                top_pairs = [pair[0] for pair in sorted_pairs[:num_of_plots]]
    
                col1, col2, col3 = st.columns(3)
    
                # Membiarkan pengguna memilih warna dengan default pilihan
                scatter_color_choice = col1.selectbox(
                    "Choose color for scatter points:",
                    list(color_map.keys()),
                    index=list(color_map.keys()).index(
                        "Default Plotly")  # Default ke "Terracotta"
                )
    
                line_color_choice = col2.selectbox(
                    "Choose color for regression line:",
                    list(color_map.keys()),
                    index=list(color_map.keys()).index(
                        "Umber")  # Default ke "Umber"
                )
    
                scatter_color = color_map[scatter_color_choice]
                line_color = color_map[line_color_choice]
    
                left_col, center_col, right_col = st.columns(3)
                columns = [left_col, center_col, right_col]
                chart_col_idx = 0
    
                # Membuat scatter plot untuk pasangan kolom dengan korelasi tertinggi
                for col1, col2 in top_pairs:
                    if scatter_color == 'plotly':
                        fig = px.scatter(self.df,
                                         x=col1,
                                         y=col2,
                                         trendline="ols",
                                         title=f'Scatter plot of {col1} vs {col2}',
                                         width=chart_width,
                                         height=chart_height)
                        fig.update_traces(marker=dict(size=5),
                                          selector=dict(mode='markers'))
                    else:
                        fig = px.scatter(self.df,
                                         x=col1,
                                         y=col2,
                                         trendline="ols",
                                         color_discrete_sequence=[scatter_color],
                                         title=f'Scatter plot of {col1} vs {col2}',
                                         width=chart_width,
                                         height=chart_height)
                        fig.update_traces(marker=dict(size=5, color=scatter_color),
                                          selector=dict(mode='markers'))
    
                    if line_color == 'plotly':
                        pass
                    else:
                        fig.update_traces(line=dict(color=line_color),
                                          selector=dict(type='scatter',
                                                        mode='lines'))
    
                    fig.update_layout(title={'font': {'size': 12}})
    
                    columns[chart_col_idx % 3].plotly_chart(fig)
                    chart_col_idx += 1

            # Bar chart
            st.write("## Bar Chart")

            # Filter kolom numerik yang memiliki nilai unik tidak lebih dari 20
            valid_numeric_cols = [col for col in self.numeric_cols]
            valid_categorical_cols = [
                col for col in self.categorical_cols
                if self.df[col].nunique() <= 20
            ]

            col1, col2, col3 = st.columns(3)

            # Pilih kolom numerik dan kategorikal
            selected_numeric_col = col1.selectbox(
                'Choose numeric column for aggregation', valid_numeric_cols)
            selected_categorical_hue = col2.selectbox(
                'Choose categorical column for hue', valid_categorical_cols)

            # Pilih palet warna
            selected_palette = col3.selectbox(
                "Choose a bar chart color palette:",
                list(color_palettes_discrete.keys()),
                index=list(
                    color_palettes_discrete.keys()).index("Default Plotly"))

            left_col, center_col, right_col = st.columns(3)
            columns = [left_col, center_col, right_col]
            chart_col_idx = 0

            for col in valid_categorical_cols:
                if selected_palette == 'Default Plotly':
                    fig = px.bar(
                        self.df,
                        x=col,
                        y=selected_numeric_col,
                        color=selected_categorical_hue,
                        title=
                        f'Bar Chart of {col} grouped by {selected_categorical_hue}',
                        width=chart_width,
                        height=chart_height
                    )  # Gunakan palet warna yang terpilih
                else:
                    fig = px.bar(
                        self.df,
                        x=col,
                        y=selected_numeric_col,
                        color=selected_categorical_hue,
                        title=
                        f'Bar Chart of {col} grouped by {selected_categorical_hue}',
                        width=chart_width,
                        height=chart_height,
                        color_discrete_sequence=color_palettes_discrete[
                            selected_palette]
                    )  # Gunakan palet warna yang terpilih
                fig.update_layout(title={
                    'font': {
                        'size': 12
                    }  # Increase font size for title
                })
                columns[chart_col_idx % 3].plotly_chart(fig)
                chart_col_idx += 1

            if len(self.numeric_cols) != 0:
                # Heatmap of correlation
                title_placeholder = st.empty()
                correlation_methods = ["pearson", "kendall", "spearman"]
                selected_method = st.selectbox("Choose a correlation method:",
                                               correlation_methods)
                default_palette_idx = list(
                    color_palettes.keys()).index("Cool (Blue-Cyan-Purple)")
                selected_palette = st.selectbox("Choose a heatmap color palette:",
                                                list(color_palettes.keys()),
                                                index=default_palette_idx)
                title_placeholder.write(
                    f"## Heatmap of {selected_method.capitalize()} Correlation")
                corr = self.df[self.numeric_cols].corr(method=selected_method)
    
                if selected_palette == 'plotly':
                    fig = ff.create_annotated_heatmap(
                        z=corr.values,
                        x=list(corr.columns),
                        y=list(corr.index),
                        annotation_text=corr.round(2).values)
                else:
                    fig = ff.create_annotated_heatmap(
                        z=corr.values,
                        x=list(corr.columns),
                        y=list(corr.index),
                        annotation_text=corr.round(2).values,
                        colorscale=color_palettes[selected_palette])
    
                st.plotly_chart(fig)

            if len(self.categorical_cols) > 1:
                # Chi square for Categorical Columns
                st.write("## Chi Square for Categorical Columns")
                results = []
    
                for col1 in self.categorical_cols:
                    for col2 in self.categorical_cols:
                        if col1 != col2:
                            contingency = pd.crosstab(self.df[col1], self.df[col2])
                            chi2, p, _, _ = chi2_contingency(contingency)
    
                            if p < 0.05:
                                correlation_strength = "High"
                                explanation = "Statistically Significant"
                            else:
                                correlation_strength = "Low"
                                explanation = "Not Statistically Significant"
    
                            results.append((col1, col2, chi2, p,
                                            correlation_strength, explanation))
    
                results_df = pd.DataFrame(results,
                                          columns=[
                                              "Column 1", "Column 2", "Chi2 Value",
                                              "P Value", "Correlation Strength",
                                              "Description"
                                          ])
                results_df = results_df.sort_values(by='P Value')
    
                # Color the cells with high correlation
                def color_cells(val):
                    if val == "High":
                        return 'background-color: yellow'
                    else:
                        return ''
    
                styled_df = results_df.style.applymap(
                    color_cells, subset=["Correlation Strength"])
    
                st.write(styled_df)
    
                import scipy.stats as stats

            st.write("## Box Plot")

            if (len(self.numeric_cols) != 0) & (len(self.categorical_cols) != 0):
                # Filter the numeric columns based on unique value criteria
                valid_numeric_cols_for_box = [
                    col for col in self.numeric_cols if self.df[col].nunique() > 20
                ]
    
                # Filter the categorical columns based on unique value criteria
                valid_categorical_cols_for_box = [
                    col for col in self.categorical_cols
                    if self.df[col].nunique() < 10
                ]
    
                # Calculate Point-Biserial correlation for each combination of numeric and categorical columns
                for column in self.df.columns:
                    if self.df[column].dtype in ['float64', 'int64']:
                        self.df[column] = self.df[column].fillna(
                            self.df[column].mean())
    
                correlations = []
                for num_col in valid_numeric_cols_for_box:
                    for cat_col in valid_categorical_cols_for_box:
                        if len(self.df[num_col].unique()) > 1 and len(
                                self.df[cat_col].unique()) > 1:
                            correlation, _ = stats.pointbiserialr(
                                self.df[num_col],
                                self.df[cat_col].astype('category').cat.codes)
                            correlations.append(((num_col, cat_col), correlation))
    
                # Sort the pairs by absolute value of Point-Biserial correlation
                sorted_correlations = sorted(correlations,
                                             key=lambda x: abs(x[1]),
                                             reverse=True)
    
                # Let the user choose how many plots to display, default is 9
                num_of_plots = st.selectbox("Choose number of plots to display:",
                                            list(range(1, 10)),
                                            index=8)  # Default to 9
    
                top_pairs = [
                    pair[0] for pair in sorted_correlations[:num_of_plots]
                ]
    
                left_col, center_col, right_col = st.columns(3)
                columns = [left_col, center_col, right_col]
                chart_col_idx = 0
    
                # Create box plots for the top correlated pairs
                for num_col, cat_col in top_pairs:
                    fig = px.box(
                        self.df,
                        x=cat_col,
                        y=num_col,
                        title=f'Box Plot of {num_col} grouped by {cat_col}',
                        width=chart_width,
                        height=chart_height)
                    fig.update_layout(title={'font': {'size': 12}})
    
                    columns[chart_col_idx % 3].plotly_chart(fig)
                    chart_col_idx += 1

            # Pairplot
            # st.write("## Pairplot")
            # sns.pairplot(self.df)
            # st.pyplot(plt)

            try:
                st.write("## Pie Charts for Categorical Columns")

                # Filter categorical columns based on unique value count
                filtered_categorical_cols = [
                    col for col in self.categorical_cols
                    if len(self.df[col].unique()) < 10
                ]
                filtered_numeric_cols = [
                    col for col in self.numeric_cols
                    if len(self.df[col].unique()) > 20
                ]

                # Choose aggregation method
                aggregation_method = st.selectbox('Choose aggregation method',
                                                  ['count', 'mean', 'sum'])

                # If aggregation method is not count, let user choose a numeric column
                selected_numeric = None
                if aggregation_method != 'count':
                    selected_numeric = st.selectbox(
                        'Choose a numeric column for aggregation',
                        filtered_numeric_cols)

                left_col, center_col, right_col = st.columns(3)
                columns = [left_col, center_col, right_col]
                chart_col_idx = 0

                # Aggregate the data
                if aggregation_method == 'count':
                    aggregated_data = self.df[
                        filtered_categorical_cols].groupby(
                            filtered_categorical_cols).size().reset_index(
                                name='count')
                else:
                    aggregated_data = self.df.groupby(
                        filtered_categorical_cols)[selected_numeric].agg(
                            aggregation_method).reset_index()

                # Create pie charts for all filtered categorical columns
                for col in filtered_categorical_cols:
                    if aggregation_method == 'count':
                        fig = px.pie(
                            aggregated_data,
                            names=col,
                            values='count',
                            title=
                            f'Pie Chart of {col}<br>(Aggregated by count)',
                            width=chart_width,
                            height=chart_height)
                        fig.update_layout(title={
                            'font': {
                                'size': 12
                            }  # Increase font size for title
                        })
                    else:
                        fig = px.pie(
                            aggregated_data,
                            names=col,
                            values=selected_numeric,
                            title=
                            f'Pie Chart of {col}<br>(Aggregated by {aggregation_method} of {selected_numeric})',
                            width=chart_width,
                            height=chart_height)
                        fig.update_layout(title={
                            'font': {
                                'size': 12
                            }  # Increase font size for title
                        })
                    columns[chart_col_idx % 3].plotly_chart(fig)
                    chart_col_idx += 1

            except:
                pass

            try:
                # Line Plots
                st.write("## Line Plots")

                def is_arithmetic_sequence(lst):
                    if len(lst) < 2:
                        return False
                    diff = lst[1] - lst[0]
                    for i in range(2, len(lst)):
                        if lst[i] - lst[i - 1] != diff:
                            return False
                    return True

                # Filter integer columns based on unique value criteria and arithmetic sequence
                valid_integer_cols_for_line = [
                    col for col in self.integer_cols
                    if self.df[col].nunique() > 5
                    and is_arithmetic_sequence(sorted(self.df[col].unique()))
                ]

                # Calculate Pearson correlation for each combination of valid integer and numeric columns
                correlations = {}
                for int_col in valid_integer_cols_for_line:
                    for num_col in self.numeric_cols:  # Assuming you have numeric_cols defined
                        if int_col != num_col:
                            correlation = self.df[int_col].corr(
                                self.df[num_col])
                            correlations[(int_col, num_col)] = correlation

                # Sort the pairs by absolute value of Pearson correlation in descending order
                sorted_correlations = sorted(correlations.items(),
                                             key=lambda x: abs(x[1]),
                                             reverse=True)

                # Let the user choose how many plots to display, default is 9
                num_of_plots = st.selectbox(
                    "Choose number of plots to display:",
                    list(range(1,
                               len(sorted_correlations) + 1)),
                    index=1)  # Default to 9

                # Select the top pairs based on user input
                top_pairs = [
                    pair[0] for pair in sorted_correlations[:num_of_plots]
                ]

                # Aggregation options for numeric columns
                agg_option = st.selectbox("Select aggregation for Y-axis:",
                                          ['mean', 'median', 'sum'])

                # Columns for layout
                left_col, center_col, right_col = st.columns(3)
                columns = [left_col, center_col, right_col]
                chart_col_idx = 0

                def plot_all_line_charts(df, integer_cols, float_cols,
                                         agg_option, chart_width, chart_height,
                                         columns, chart_col_idx):
                    for x_col in integer_cols:
                        for y_col in float_cols:
                            # Skip plotting if x and y columns are the same
                            if x_col == y_col:
                                continue

                            fig = create_line_plot(df, x_col, y_col,
                                                   agg_option, chart_width,
                                                   chart_height)
                            columns[chart_col_idx % 3].plotly_chart(fig)
                            # chart_col_idx += 1

                    # chart_col_idx = 0

                def create_line_plot(df, x_col, y_col, aggregation,
                                     chart_width, chart_height):
                    if aggregation == 'mean':
                        df_agg = df.groupby(x_col,
                                            as_index=False)[y_col].mean()
                    elif aggregation == 'median':
                        df_agg = df.groupby(x_col,
                                            as_index=False)[y_col].median()
                    else:  # sum
                        df_agg = df.groupby(x_col, as_index=False)[y_col].sum()

                    # Create line plot
                    fig = px.line(df_agg, x=x_col, y=y_col)

                    # Apply custom style
                    fig.update_layout(
                        title={
                            'text':
                            f'Line Plot of {y_col} by {x_col}<br>(Aggregated by {aggregation})',
                            # 'x': 0.5,  # Center the title
                            # 'xanchor': 'center',
                            'font': {
                                'size': 12
                            }  # Increase font size for title
                        },
                        width=chart_width,
                        height=chart_height,
                        paper_bgcolor="white",  # set background color to white
                        plot_bgcolor=
                        "white",  # set plot background color to white
                        xaxis_showgrid=True,  # Show x-axis gridlines
                        xaxis_gridcolor=
                        'rgba(200,200,200,0.2)',  # Lighten gridlines
                        yaxis_showgrid=True,  # Show y-axis gridlines
                        yaxis_gridcolor=
                        'rgba(200,200,200,0.2)',  # Lighten gridlines
                        margin=dict(t=40, b=40, l=40,
                                    r=10)  # Add margins to the plot
                    )

                    # Update line style
                    fig.update_traces(
                        line=dict(dash="solid",
                                  width=2.5),  # Set line style and width
                        marker=dict(size=8),  # Adjust marker size
                        mode="lines+markers"  # Add markers to the line
                    )

                    return fig

                chart_width = 300  # width of the chart to fit within the column
                chart_height = 400  # height of the chart

                # Create line plots for the selected pairs
                for int_col, num_col in top_pairs:
                    plot_all_line_charts(self.df, [int_col], [num_col],
                                         agg_option, chart_width, chart_height,
                                         columns, chart_col_idx)
                    chart_col_idx += 1
            except:
                pass

        with tab5:
            st.title("1-Click Summarize!")
            st.subheader("A. Data Summary")
            st.dataframe(self.df.head(), use_container_width=True)
            r, c = self.df.shape
            date_cols = self.date_cols.tolist()
            integer_cols = self.integer_cols.tolist()
            numeric_cols = self.numeric_cols.tolist()
            float_cols = self.float_cols.tolist()
            categorical_cols = self.categorical_cols.tolist()
            text1 = f'''We have {r} records with {c} columns. The Columns consits of:  
            1. Have {len(date_cols)} date columns : {date_cols}.  
            2. Have {len(numeric_cols)} numeric columns : {numeric_cols}.  
            2.a. Have {len(integer_cols)} integer columns : {integer_cols}.  
            2.b. Have {len(float_cols)} float columns : {float_cols}.  
            3. Have {len(categorical_cols)} categorical columns : {categorical_cols}.'''
            data = self.df.copy()
            for col in categorical_cols:
                data[col] = data[col].cat.add_categories(['Missing Data'])
            data[categorical_cols] = data[categorical_cols].fillna('Missing Data')
            data[numeric_cols] = data[numeric_cols].fillna(0)
            st.markdown(text1)
            st.write('---')
            selected_variable = st.selectbox(
                "Select a Variables for detail Summary:",
                data.columns.tolist())
            st.session_state.button_clicked = False
            message_placeholder = st.empty()
            message_placeholder.write("Processing... please wait!!!")

            progress_bar = st.progress(0)
            cols_used = self.variable_selection(selected_variable,
                                                progress_bar)
            progress_bar.empty()
            message_placeholder.empty()

            st.write("---")

            data = data[cols_used + [selected_variable]].copy()
            date_cols = data.select_dtypes(
                include='datetime64[ns]').columns.tolist()
            integer_cols = data.select_dtypes(['int16', 'int32',
                                               'int64']).columns.tolist()
            numeric_cols = data.select_dtypes(
                ['int16', 'int32', 'int64', 'float16', 'float32',
                 'float64']).columns.tolist()
            float_cols = data.select_dtypes(['float16', 'float32',
                                             'float64']).columns.tolist()
            categorical_cols = data.select_dtypes(exclude=[
                'int16', 'int32', 'int64', 'float16', 'float32', 'float64',
                'datetime64[ns]'
            ]).columns.tolist()
            if selected_variable in numeric_cols:
                top_bottom_text = ''
                corr_text = ''
                # Grouping Data into Deciles and Getting Group with Highest Frequency
                description = data[selected_variable].describe()
                missing_values = 100 - (description['count'] * 100 / len(data))
                skewness = data[selected_variable].skew()
                kurt = data[selected_variable].kurtosis()

                # Calculate deciles using numpy.percentile
                percentiles = np.percentile(data[selected_variable],
                                            np.arange(0, 101, 10))
                # group_labels = [
                #     f'Decile {i+1}' for i in range(len(percentiles) - 1)
                # ]

                # Add a new column to the DataFrame indicating the decile group
                data['decile_group'] = pd.cut(data[selected_variable],
                                              bins=percentiles,
                                              duplicates='drop')

                # Count the occurrences in each decile group
                group_counts = data['decile_group'].value_counts()
                top_group, max_val_group = group_counts.idxmax(), max(
                    group_counts)
                st.write("##")
                st.write(
                    f"Variable {selected_variable} have the following summary:"
                )
                i = 1
                for idx, val in description.items():
                    if i == 1:
                        text = f"{i}. have {idx} values : {val:.2f} \n "
                        i += 1
                    else:
                        new_idx = idx.replace("%", "th percentile")
                        text += f"{i}. have {new_idx} values : {val:.2f} \n "
                        i += 1
                text += f'{i}. have skewness {skewness:.3f} and kurtosis {kurt:.3f} \n '
                i += 1
                text += f'{i}. have top decile group {top_group} with frequency {max_val_group} \n '
                i += 1
                text += f'{i}. have missing values {missing_values:.2f} % \n '
                i += 1

                description['skewness'] = skewness
                description['kurtosis'] = kurt
                description['top_decile'] = top_group
                description['top_decile_freq'] = max_val_group
                description['missing_pct'] = missing_values / 100

                st.subheader("B. Distribution")
                st.dataframe(description, use_container_width=True)
                st.write('---')
                st.subheader("C. Grouped Summary")
                j = 1
                if len(categorical_cols) > 0:
                    for c_ in categorical_cols:
                        grouped_summary = data.groupby(
                            c_)[selected_variable].describe()
                        mean_values = grouped_summary['mean'].sort_values()

                        # Create formatted text for top and bottom categories
                        top_mean = mean_values.tail(3).sort_values(
                            ascending=False)
                        bottom_mean = mean_values.head(3)

                        top_mean_text = ", ".join(
                            f"{name} ({mean:.2f})" for name, mean in zip(
                                top_mean.index, top_mean.values))
                        bottom_mean_text = ", ".join(
                            f"{name} ({mean:.2f})" for name, mean in zip(
                                bottom_mean.index, bottom_mean.values))

                        top_bottom_text += f"{i}. Top 3 {c_} with highest Average {selected_variable}: {top_mean_text} \n "
                        i += 1
                        top_bottom_text += f"{i}. Bottom 3 {c_} with lowest Average {selected_variable}: {bottom_mean_text} \n "
                        i += 1

                        st.subheader(
                            f"{j}. Summary of {selected_variable} grouped by {c_}"
                        )
                        j += 1
                        st.dataframe(grouped_summary, use_container_width=True)

                        # Display top and bottom categories (based on mean)
                        c1, c2 = st.columns(2)
                        with c1:

                            c1.write(
                                f"Top 3 {c_} categories by average {selected_variable}:"
                            )
                            c1.dataframe(mean_values.tail(3).sort_values(
                                ascending=False),
                                         use_container_width=True)
                        with c2:
                            c2.write(
                                f"Bottom 3 {c_} categories by average {selected_variable}:"
                            )
                            c2.dataframe(mean_values.head(3),
                                         use_container_width=True)

                        st.write("---")
                else:
                    st.write(f"There are no significant categories variables that related with {selected_variable} based on our criteria.")
                    st.write('---')

                st.subheader("D. Correlation Summary")
                if len(numeric_cols) > 1:
                    # Calculate correlations
                    correlations = data[numeric_cols].corr()[selected_variable]
                    strong_counter = 0
                    for idx, val in correlations.items():
                        if idx != selected_variable:
                            map_conditions = 'weak' if np.abs(
                                val
                            ) < 0.6 else 'positive strong' if val >= 0.6 else 'negative strong'
                            if map_conditions != 'weak':
                                strong_counter += 1
                                st.write(
                                    "Correlation of {selected_variable} and {idx} are {map_conditions} with value {val:.3f}.  "
                                )
                            corr_text += f"{i}. correlation of {selected_variable} and {idx} are {map_conditions} with value {val:.3f}. \n "
                            i += 1
                    if strong_counter == 0:
                        st.write(
                            f"No Variables have strong correlation with {selected_variable}."
                        )
                else:
                    st.write(f"There are no significant numerical variables that related with {selected_variable} based on our criteria.")
                st.write('---')
                all_text = f"Variable {selected_variable} have the following summary: \n " + text + top_bottom_text + corr_text
                all_text_without_corr = f"Variable {selected_variable} have the following summary:  \n" + text + top_bottom_text

            elif selected_variable in categorical_cols:
                description = data[selected_variable].describe()
                missing_values = 100 - (description['count'] * 100 / len(data))
                st.write("##")
                st.write(
                    f"Variable {selected_variable} have the following summary:"
                )
                i = 1
                for idx, val in description.items():
                    if i == 1:
                        text = f"{i}. have {idx} values : {val} \n "
                        i += 1
                    else:
                        new_idx = idx.replace("top", "top values").replace(
                            "freq", "frequency of top values")
                        text += f"{i}. have {new_idx} values : {val} \n "
                        i += 1
                text += f'{i}. have missing values {missing_values:.2f} % \n '
                i += 1
                description['missing_pct'] = missing_values / 100

                st.subheader("B. Distribution")
                st.dataframe(description, use_container_width=True)
                st.write('---')

                top_bottom_text = ''
                if len(numeric_cols) > 0:
                    st.subheader("C. Grouped Summary")
                    grouped_summary = data.groupby(
                        selected_variable)[numeric_cols].describe()
                    j = 1
                    for n_ in numeric_cols:
                        st.subheader(
                            f"{j}. Summary of {selected_variable} grouped by {n_}"
                        )
                        j += 1
                        mean_values = grouped_summary[n_]['mean'].sort_values()

                        # Create formatted text for top and bottom categories
                        top_mean = mean_values.tail(3).sort_values(
                            ascending=False)
                        bottom_mean = mean_values.head(3)

                        top_mean_text = ", ".join(
                            f"{name} ({mean:.2f})" for name, mean in zip(
                                top_mean.index, top_mean.values))
                        bottom_mean_text = ", ".join(
                            f"{name} ({mean:.2f})" for name, mean in zip(
                                bottom_mean.index, bottom_mean.values))

                        top_bottom_text += f"{i}. Top 3 {selected_variable} with highest Average of {n_}: {top_mean_text} \n "
                        i += 1
                        top_bottom_text += f"{i}. Bottom 3 {selected_variable} with lowest Average of {n_}: {bottom_mean_text} \n "
                        i += 1
                        st.dataframe(grouped_summary[n_], use_container_width=True)
                        # Display top and bottom categories (based on mean)
                        c1, c2 = st.columns(2)
                        with c1:
                            c1.write(
                                f"Top 3 {selected_variable} categories by average of {n_}:"
                            )
                            c1.dataframe(top_mean, use_container_width=True)
                        with c2:
                            c2.write(
                                f"Bottom 3 {selected_variable} categories by average of {n_}:"
                            )
                            c2.dataframe(bottom_mean, use_container_width=True)

                        st.write("---")
                else:
                    st.write(f"There are no significant numerical variables that related with {selected_variable} based on our criteria.")
                    st.write("---")

                st.subheader("D. Correlation Summary")
                highest_values = dict()
                lowest_values = dict()
                corr_text = ''
                j = 1
                if len(categorical_cols) > 1:
                    for var2 in categorical_cols:
                        if var2 != selected_variable:
                            contingency_table = pd.crosstab(
                                data[selected_variable], data[var2])
                            # Find highest and lowest values
                            max_value = contingency_table.values.max()
                            min_value = contingency_table.values.min()

                            # Get indices of highest and lowest values
                            max_indices = np.where(
                                contingency_table.values == max_value)
                            min_indices = np.where(
                                contingency_table.values == min_value)

                            # Limit to only three indices if there are more than 3
                            if max_indices[0].size > 3:
                                max_indices = (max_indices[0][:3], max_indices[1][:3])
                            
                            if min_indices[0].size > 3:
                                min_indices = (min_indices[0][:3], min_indices[1][:3])

                            # Get corresponding row and column labels
                            max_rows = [
                                contingency_table.index[i]
                                for i in max_indices[0]
                            ]
                            max_cols = [
                                contingency_table.columns[i]
                                for i in max_indices[1]
                            ]

                            min_rows = [
                                contingency_table.index[i]
                                for i in min_indices[0]
                            ]
                            min_cols = [
                                contingency_table.columns[i]
                                for i in min_indices[1]
                            ]

                            # Store results in dictionaries
                            highest_values[var2] = {
                                'value': max_value,
                                'rows': max_rows,
                                'columns': max_cols
                            }
                            lowest_values[var2] = {
                                'value': min_value,
                                'rows': min_rows,
                                'columns': min_cols
                            }
                            st.subheader(
                                f"{j}. Summary of {selected_variable} grouped by {var2}"
                            )
                            j += 1
                            st.dataframe(contingency_table,
                                         use_container_width=True)
                            st.write("---")
                            corr_text += f"{i}. Highest Values of {var2}, Value: {highest_values[var2]['value']}, {selected_variable}: {highest_values[var2]['rows']}, {var2}: {highest_values[var2]['columns']}. \n "
                            corr_text += f"{i+1}. Lowest Values of {var2}, Value: {lowest_values[var2]['value']}, {selected_variable}: {lowest_values[var2]['rows']}, {var2}: {lowest_values[var2]['columns']}. \n "
                            i += 2

                else:
                    st.write(f"There are no significant categories variables that related with {selected_variable} based on our criteria.")
                    st.write("---")

                all_text = f"Variable {selected_variable} have the following summary:  \n" + text + top_bottom_text + corr_text
                all_text_without_corr = f"Variable {selected_variable} have the following summary:  \n" + text + top_bottom_text
            else:
                st.write(
                    f"{selected_variable} is a Date columns, we will not show any analytics for it."
                )

            st.subheader("Summarize It!")
            api_model = st.selectbox('Choose LLM Model to Summarize:',
                                     ('GPT4', 'GPT3.5'))
            language = st.selectbox('Choose Language:',
                                    ('Indonesia', 'English', 'Sunda'))
            style_choosen = st.selectbox('Choose the Formality:',
                                         ('Formal', 'Non-Formal'))
            objective = st.selectbox('Choose the Style:',
                                     ('Narative', 'Persuasive', 'Descriptive',
                                      'Argumentative', 'Satire'))
            format = st.selectbox(
                'Choose the Format:',
                ('Paragraph', 'Youtube Script', 'Thread', 'Caption Instagram'))
            button = st.button("Give Me Summarize!")

            # openai.api_key = st.secrets['user_api']
            def request_summary_wording(text_summary, language, style_choosen,
                                        objective, format, api_model):
                messages = [{
                    "role":
                    "system",
                    "content":
                    f"Aku akan menjabarkan summary kamu dengan menggunakan bahasa {language}. Dalam format {format}."
                }, {
                    "role":
                    "user",
                    "content":
                    f"""Buatkan laporan yang insightful dengan gaya {style_choosen} dan {objective}, menggunakan bahasa {language}, dalam format {format}, serta berikan opinimu dari informasi umum yang diketahui untuk setiap point dari informasi berikut: {text_summary}."""
                }]

                if api_model == 'GPT3.5':
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        # model="gpt-4",
                        messages=messages,
                        max_tokens=3000,
                        temperature=0.6)
                    script = response.choices[0].message['content']
                else:
                    response = openai.ChatCompletion.create(
                        # model="gpt-3.5-turbo",
                        model="gpt-4",
                        messages=messages,
                        max_tokens=3000,
                        temperature=0.6)
                    script = response.choices[0].message['content']

                return script

            def split_text_into_lines(text, words_per_line=20):
                words = text.split()
                lines = []
                for i in range(0, len(words), words_per_line):
                    line = ' '.join(words[i:i + words_per_line])
                    lines.append(line)
                return '\n'.join(lines)

            # Contoh penggunaan fungsi
            # text_explanation = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque nec eros erat. Duis nulla lectus, vehicula id dictum quis, ullamcorper quis arcu. Vivamus et eros eu erat suscipit mollis. Curabitur a eros quis nisl porttitor pretium."
            # formatted_text = split_text_into_lines(text_explanation)
            # print(formatted_text)

            if button:
                st.session_state.button_clicked = True

            if st.session_state.get('button_clicked', False):
                st.success("Summary Generated!")
                # st.text(all_text)
                with st.spinner(
                        'Generating insights...(it may takes 1-2 minutes)'):
                    try:
                        response = request_summary_wording(
                            str(all_text), language, style_choosen,
                            objective, format, api_model)
                        # st.text(split_text_into_lines(response))
                        st.write(response)
                    except:
                        response = request_summary_wording(
                            str(all_text_without_corr), language, style_choosen,
                            objective, format, api_model)
                        # st.text(split_text_into_lines(response))
                        st.write(response)
