import streamlit as st
import plotly.express as px
import pandas as pd
# from pandasai import PandasAI
# from pandasai.llm.openai import OpenAI
# import os
# import openai
# import geopandas as gpd

# Page settings
st.set_page_config(
    page_title="ProsperaLima: Illuminating Pathways to Urban Excellence",
    page_icon="🏙️",
    layout="wide",
)

# Define a list of file paths
file_paths = [
    r"Datasets/Large Companies 2021 (Manufacturing Sector).csv",
    # r"Large Files/MSME_2020_2021_data.csv"
]

@st.cache_data
# Read data from a file path
def get_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


df1 = get_data(file_paths[0])
# df2 = get_data(file_paths[1]) 

# List to store DataFrames from individual chunks
chunks = []
# Number of chunks
num_chunks = 6  
# Combine chunks into a single DataFrame
for i in range(1, num_chunks + 1):
    chunk_path = f'Datasets/MSME_{i}.csv'
    chunk_df = pd.read_csv(chunk_path)
    chunks.append(chunk_df)

# Concatenate all DataFrames
df2 = pd.concat(chunks, ignore_index=True)

def display_section(Heading,Content):
    st.markdown(
        f"""
        <div style="
            background-color: #70d1d0;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        ">
        <h3>{Heading}</h3>
        <p>{Content}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Skill sets data
skill_sets = {
    "##### Manufacture of Plastic Products (ISIC Code: 2520)": [
        "Plastic molding and extrusion",
        "Quality control in plastic manufacturing",
        "Polymer chemistry",
        "Machine operation and maintenance for plastic production",
    ],
    "##### Manufacture of Garment (ISIC Code: 1810)": [
        "Sewing and stitching",
        "Pattern making and cutting",
        "Textile knowledge",
        "Quality control in garment production",
        "Machine operation in garment manufacturing",
    ],
    "##### Processing and Preserving of Fruit and Vegetables (ISIC Code: 1513)": [
        "Food processing techniques",
        "Quality control in food processing",
        "Knowledge of preservation methods",
        "Equipment operation in food processing",
    ],
    "##### Manufacture of Engines and Turbines, Except Aircraft, Vehicle, and Cycle Engines (ISIC Code: 2811)": [
        "Mechanical engineering",
        "Precision machining",
        "Engine assembly and testing",
        "Quality assurance in engine manufacturing",
    ],
    "##### Manufacture of Luggage, Handbags and the Like, Saddlery, and Harness (ISIC Code: 1512)":[
        "Leatherworking",
        "Design and pattern making for bags and luggage",
        "Sewing and stitching for leather goods",
        "Quality control in leather product manufacturing",
    ],
}


# creating column
col1, col2 = st.columns([2, 8])

# Create a session state
if "button_state_1" not in st.session_state:
    st.session_state.button_state_1 = False
if "button_state_2" not in st.session_state:
    st.session_state.button_state_2 = False

with col1:
    Obj = "Objective"
    Cont_1 = "To gain insights into the landscape of Micro, Small, and Medium Enterprises (MSMEs) in Lima, Peru. And to provide a comprehensive directory of large companies operating in the Manufacturing sector in Lima. " \
             "The goal is to provide actionable insights that can contribute to the improvement of Lima's economy and the overall quality of life for its residents."

    display_section(Obj, Cont_1)

    Header = "MSME - Micro, Small, Medium Enterprises & Large Companies: "
    Cont_2  = "MSMEs play a significant role in many economies, contributing to job creation, economic growth, and innovation. " \
           "They often face unique challenges compared to larger corporations, such as access to finance, technology, and markets. " \
              "However, they also offer several advantages, including agility, flexibility, and a close connection to their communities.\n " \
              "The Manufacturing companies are identified based on their significant contributions to the Manufacturing GDP and are accredited by the National Superintendence of Customs and Tax Administration (SUNAT)."
    display_section(Header,Cont_2)

with col2:
    st.title("Industrial Insights: Unveiling Trends in Large Manufacturing and MSME Sectors")

    # Buttons for Dataset
    button_col1, button_col2 = st.columns(2)

    if button_col1 or button_col2:
        st.session_state["button_state"] = not st.session_state.get("button_state", False)

    # Display the sample of the data if the button is pressed
    if button_col1.button("Large Companies 2021", key="button_col1"):
        st.session_state.button_state_1 = not st.session_state.button_state_1
        st.session_state.button_state_2 = False

    if button_col2.button("MSME Companies 2020&2021", key="button_col2"):
        st.session_state.button_state_2 = not st.session_state.button_state_2
        st.session_state.button_state_1 = False

        # Display the sample of the data based on button states
    if st.session_state.button_state_1:
        st.dataframe(df1[["RUC", "Company_Name", "DESCRIPTION_CIIU", "CIIU", "District", "UBIGEO", "Sector"]])

    if st.session_state.button_state_2:
        st.dataframe(df2)

    else:
        tab1, tab2, tab3 = st.tabs(["Large Companies", "MSME Companies", "Comparative Analysis"])

        with tab1:
            # 1. Dominant industries within manufacturing and assess their resilience.
            industry_counts = df1['DESCRIPTION_CIIU'].value_counts().reset_index()
            industry_counts.columns = ['Industry', 'Company_Count']
            fig_1 = px.bar(
                industry_counts.head(10),
                x='Industry',
                y='Company_Count',
                labels={'Company_Count': 'Number of Companies', 'Industry': 'Industry'},
                title='Top 10 Dominant Industries in Manufacturing',
            )
            fig_1.update_layout(xaxis_tickangle=-45, xaxis_title='Industry', yaxis_title='Number of Companies')
            st.plotly_chart(fig_1)
            st.write("Manufacture of plastic products - 119, Manufacture of Garment - 76, Processing and preserving of fruit and vegetables- 69, Manufacture Of Engines And Turbines, Except Aircraft, "
                     "Vehicle And Cycle Engines - 61, Manufacture Of Luggage, Handbags And The Like, Saddlery And Harness - 60, Manufacture of other food products - 47, Manufacture of other chemical products - 45, "
                     "Preparation and Weaving of textile fibres - 44, Manufacture of pharmaceuticals, medicinal chemicals and botanical products - 39, Manufacture of other fabricated metal product - 33")

            # 2. Geographical distribution of large manufacturing companies within Lima
            dist = df1['District'].value_counts().reset_index()
            dist.columns = ['District', 'Company_Count']
            fig_2 = px.bar(
                dist,
                x='District',
                y='Company_Count',
                labels={'District': 'Number of Companies', 'index': 'District'},
                title='Geographical Distribution of Large Manufacturing Companies within Lima (District-level)',
            )
            fig_2.update_layout(xaxis_tickangle=-45, xaxis_title='District', yaxis_title='Number of Companies')
            st.plotly_chart(fig_2)
            st.write("###### Top 5 Districts with the Highest Number of Manufacturing Companies:")
            st.text("ATE - 162\nLIMA - 95\nSANTIAGO DE SURCO - 88\nSAN ISIDRO - 77\nSAN JUAN DE LURIGANCHO - 67")

            # 3. Skill Requirements Analysis
            # Display skill sets for each sector
            st.header('"Skill sets required for jobs in the Top 5 manufacturing sector":')
            for sector, skills in skill_sets.items():
                st.markdown(sector)
                st.write("\n".join(f"- {skill}" for skill in skills))

        with tab2:
            # 1. Industries within sectors that make the most significant contribution to Lima's GDP.
            # Extract data for each period
            data_2020 = df2[df2['PERIOD_2020'] == 2020]
            data_2021 = df2[df2['PERIOD_2021'] == 2021]
            # Top 10 industries for each year
            df_2020 = data_2020['DESCRIPTION_CIIU'].value_counts().reset_index().head(5)
            df_2021 = data_2021['DESCRIPTION_CIIU'].value_counts().reset_index().head(5)
            # Rename columns for clarity
            df_2020.columns = ['Industry', 'Number of Companies']
            df_2021.columns = ['Industry', 'Number of Companies']
            # Chart for 2020
            fig_2020 = px.bar(
                df_2020,
                x='Industry',
                y='Number of Companies',
                title='Top 5 Dominant Industries in 2020',
                height=400,
                width=500,
            )
            # Chart for 2021
            fig_2021 = px.bar(
                df_2021,
                x='Industry',
                y='Number of Companies',
                title='Top 5 Dominant Industries in 2021',
                height=400,
                width=500,
            )
            # Display side-by-side charts using columns
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_2021)
            with col2:
                st.plotly_chart(fig_2020)
            st.write("The top industries contributing to Lima's GDP show consistency across both years, indicating the resilience and stability of certain sectors.")

            # 2. Sectoral Distribution Analysis
            # Count the number of MSMEs in each sector for 2020 and 2021
            sector_distribution_2020 = data_2020['Sector'].value_counts()
            sector_distribution_2021 = data_2021['Sector'].value_counts()
            # Plot the sectoral distribution for 2020
            fig_2020_2 = px.bar(
                sector_distribution_2020,
                x=sector_distribution_2020.index,
                y=sector_distribution_2020.values,
                labels={'y': 'Number of MSMEs', 'x': 'Sector'},
                title='Sectoral Distribution of MSMEs in 2020',
                height=400,
                width=450,
            )
            # Plot the sectoral distribution for 2021
            fig_2021_2 = px.bar(
                sector_distribution_2021,
                x=sector_distribution_2021.index,
                y=sector_distribution_2021.values,
                labels={'y': 'Number of MSMEs', 'x': 'Sector'},
                title='Sectoral Distribution of MSMEs in 2021',
                height=400,
                width=450,
            )
            # Display side-by-side charts using columns
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_2021_2)
            with col2:
                st.plotly_chart(fig_2020_2)
            st.write("In both years, Commerce remains the predominant sector among MSMEs. The number of MSMEs in the Services sector has increased from 325,272 (2020) to 377,018 (2021). "
                     "The number of MSMEs in the Manufacturing sector has increased from 81,772 (2020) to 94,512 (2021).")

            # 3. Geographical Concentration Analysis
            # Count the number of MSMEs in each district for 2020 and 2021
            district_distribution_2020 = data_2020['District'].value_counts().head(10)
            district_distribution_2021 = data_2021['District'].value_counts().head(10)
            # Plot the geographical distribution for 2020
            fig_2020_3 = px.bar(
                district_distribution_2020,
                x=district_distribution_2020.index,
                y=district_distribution_2020.values,
                labels={'y': 'Number of MSMEs', 'x': 'District'},
                title='Geographical Distribution of MSMEs(Top Districts) - 2020',
                height=400,
                width=450,
            )
            # Plot the geographical distribution for 2021
            fig_2021_3 = px.bar(
                district_distribution_2021,
                x=district_distribution_2021.index,
                y=district_distribution_2021.values,
                labels={'y': 'Number of MSMEs', 'x': 'District'},
                title='Geographical Distribution of MSMEs(Top Districts) - 2021',
                height=400,
                width=450,
            )
            # Display side-by-side charts using columns
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_2021_3)
            with col2:
                st.plotly_chart(fig_2020_3)
            st.write("The top districts with the highest number of manufacturing companies have a consistent presence in both years, with slight increases in numbers. "
                     "There is an overall growth in the number of MSMEs in the Manufacturing sector across districts.")

            st.write("## Conclusion: ")
            st.write("The MSME landscape in Lima has experienced growth, particularly in the Services and Manufacturing sectors, "
                     "between 2020 and 2021. Commerce continues to be a robust sector. The geographical distribution of manufacturing companies has expanded, reflecting a positive trend in economic activities across districts. "
                     "Understanding these trends can guide policymakers and stakeholders in formulating strategies to support the growth and resilience of MSMEs in Lima.")

        with tab3:
            # Comparison of MSME Companies (Commerce, Services, Manufacturing) 2020-2021 and Large Companies in Manufacturing 2021
            st.markdown(
                "## Comparison of MSME Companies (Commerce, Services, Manufacturing) 2020-2021 and Large Companies in Manufacturing 2021")

            # MSME Companies (Commerce, Services, Manufacturing) 2020-2021
            st.markdown("### MSME Companies (Commerce, Services, Manufacturing) 2020-2021:")
            st.markdown(
                "- The trends in the most common CIIU codes remain similar, with slight increases in the number of MSMEs in 2021.")
            st.markdown(
                "- The top industries contributing to Lima's GDP show consistency across both years, indicating the resilience and stability of certain sectors.")
            st.markdown(
                "- There is an overall increase in the number of MSMEs in all sectors, with the most significant growth observed in Commerce and Services.")
            st.markdown(
                "- The geographical distribution of manufacturing companies has expanded, reflecting a positive trend in economic activities across districts.")

            # Large Companies in Manufacturing 2021
            st.markdown("### Large Companies in Manufacturing 2021:")
            st.markdown("#### Overall Growth:")
            st.markdown(
                "- The overall number of MSMEs in Commerce, Services, and Manufacturing has increased from 2020 to 2021.")
            st.markdown("- The Manufacturing sector, in particular, shows substantial growth among MSMEs.")

            st.markdown("#### Consistency in Key Industries:")
            st.markdown("- The top industries contributing to Lima's GDP show consistency across both years for MSMEs.")
            st.markdown(
                "- In the large manufacturing sector, the top industries are different but also reflect the diversity of manufacturing activities.")

            st.markdown("#### Geographical Distribution:")
            st.markdown(
                "- The geographical distribution of manufacturing companies has expanded in both MSMEs and large companies.")

            st.markdown("#### Key Industries in Large Companies:")
            st.markdown(
                "- Large manufacturing companies are concentrated in districts like ATE, Lima, Santiago de Surco, and San Isidro, similar to MSMEs.")

            st.markdown("#### Industry Concentration:")
            st.markdown(
                "- Certain industries, such as the Manufacture of Plastic Products and Garments, have a significant presence in both MSMEs and large companies.")

            # Recommendations
            st.markdown("### Recommendations:")
            st.markdown(
                "1. **Support for Growing Sectors:**\n   - Government and stakeholders should continue to support the growth of sectors such as Commerce and Services, where the majority of MSMEs operate.")
            st.markdown(
                "2. **Tailored Support for Manufacturing:**\n   - Recognizing the growth in the Manufacturing sector, targeted support programs should be designed to address the specific needs of MSMEs in manufacturing.")
            st.markdown(
                "3. **Infrastructure Development:**\n   - Infrastructure development in districts with high concentrations of manufacturing companies can further boost economic activities and job creation.")
            st.markdown(
                "4. **Skill Development Initiatives:**\n   - Tailor skill development initiatives to meet the specific needs of MSMEs and large companies in key industries, fostering innovation and competitiveness.")
            st.markdown(
                "5. **Regional Economic Development:**\n   - Encourage regional economic development by identifying and supporting districts with emerging clusters of manufacturing activities.")


# # Create a list of DataFrames
# dataframes = [df1, df2]
#
# # Instantiate a LLM
# # openai_api_key = os.environ["OPENAI_API_KEY"]
# openai_api_key = st.secrets["OPEN_API_KEY"]
# llm = OpenAI(api_token=openai_api_key)
# pandas_ai = PandasAI(llm)
#
# # Streamlit app
# st.title("Pandas AI Chat Assistant")
#
# # Chatbox in the sidebar
# with st.sidebar:
#     st.write("Get Started: Embark on a journey of discovery. Click, explore, and empower Lima's future with ProsperaLima.")
#     st.sidebar.title("ChatGPT like Clone")
#
#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
#
#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         with st.chat_message(message['role']):
#             st.markdown(message['content'])
#
#     # React to user input
#     prompt = st.text_input("Hello, Do you have any questions?")
#     if prompt:
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         # Add user message to chat history
#         st.session_state.messages.append({'role': 'user', 'content': prompt})
#
#         # Run the assistant on each DataFrame and aggregate responses
#         responses = [pandas_ai.run(df, prompt=prompt) for df in dataframes]
#
#         # Display aggregated responses
#         st.text("AI Responses:")
#         for i, response in enumerate(responses, 1):
#             st.write(f"Response for DataFrame {i}:")
#             st.write(response)
#             with st.chat_message("assistant"):
#                 st.markdown(response)
#                 st.session_state.messages.append({"role": "assistant", "content": response})
