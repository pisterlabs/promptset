import streamlit as st
import psycopg2
import openai
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import json
import os

# Initialize OpenAI client
openai.api_key = "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"

# Connect to your PostgreSQL database
def connect_to_database():
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="luolex",
            host="34.173.71.254",
            port=5432,
            database="libraries",
        )
        return connection
    except psycopg2.Error as e:
        st.error(f"Error connecting to the database: {e}")
        return None

# Query the database based on given visits and states
def get_libname_from_database(visits, states):
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor()
            query = f"SELECT libname FROM pls_fy2014_pupld14a WHERE visits = {visits} AND stabr = '{states}'"
            cursor.execute(query)
            libname = cursor.fetchone()
            cursor.close()
            connection.close()
            return libname[0] if libname else "The library doesn't exist."
        except psycopg2.Error as e:
            st.error(f"Error executing the query: {e}")
            return "Error in database query."


# Function to run the conversation with OpenAI
def run_conversation(visits, states):
    messages = [{"role": "user", "content": f"What is the libname with visits={visits} and states='{states}'?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_libname_from_database",
                "description": "Get libname from database based on visits and states",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "visits": {"type": "integer"},
                        "states": {"type": "string"},
                    },
                    "required": ["visits", "states"],
                },
            },
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response['choices'][0]['message']
    tool_calls = response_message.get('tool_calls', [])
    print(tool_calls)

    if not tool_calls:
        return "Error in OpenAI response: No tool calls found."

    messages.append(response_message)
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        function_to_call = globals().get(function_name)
        
        if function_to_call is None:
            return f"Error: Function {function_name} not found."
        print(tool_call['function']['arguments'])
        function_args = json.loads(tool_call['function']['arguments'])
        function_response = function_to_call(
            visits=function_args.get("visits"),
            states=function_args.get("states"),
        )

        print(function_response)

        messages.append(
            {
                "tool_call_id": tool_call['id'],
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )

    second_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    content = second_response['choices'][0]['message'].get('content')
    
    if content is not None:
        return content
    else:
        return "Error in OpenAI response: Null content."


# Function to estimate expected visits for a new library
def estimate_expected_visits(states, lat_range, lon_range):
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor()
            # Query the database to get average visits for libraries within the specified region
            query = f"SELECT AVG(visits) FROM pls_fy2014_pupld14a WHERE stabr = '{states}' AND latitude BETWEEN {lat_range[0]} AND {lat_range[1]} AND longitud BETWEEN {lon_range[0]} AND {lon_range[1]}"
            cursor.execute(query)
            avg_visits = cursor.fetchone()[0]
            cursor.close()
            connection.close()

            return avg_visits if avg_visits else 0  # Return the average visits or 0 if no data found
        except psycopg2.Error as e:
            st.error(f"Error executing the query: {e}")
            return 0  # Return 0 for error in database query


# Streamlit app
def main():
    st.title("OpenAI + PostgreSQL + Folium Streamlit App")

    # User input for states, latitude, and longitude range
    states = st.text_input("Enter the state code", value="")
    lat_range = st.slider("Select Latitude Range", -90.0, 90.0, (-90.0, 90.0))
    lon_range = st.slider("Select Longitud Range", -180.0, 180.0, (-180.0, 180.0))

    # Estimate expected visits for a new library
    expected_visits = estimate_expected_visits(states, lat_range, lon_range)

    # Display the result
    st.success(f"The expected visits for a new library in the specified region are: {expected_visits:.2f} visits")

    # Create a Folium map
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=4)  # Default location (you can adjust this)

    # Add a marker for the estimated location of the new library
    if expected_visits:
        st.sidebar.subheader("Expected Library Location")
        st.sidebar.text(f"Expected Visits: {expected_visits:.2f} visits")

        # You can customize the marker based on the actual data
        marker_expected = folium.Marker(
            location=[lat_range[0], lon_range[0]],
            popup=f"Expected Visits: {expected_visits:.2f} visits",
            tooltip=f"Expected Visits: {expected_visits:.2f} visits",  # Show visits on mouse hover
        )
        marker_expected.add_to(m)

    # Display the map
    folium_static(m)

if __name__ == "__main__":
    main()




# # Streamlit app
# def main():
#     st.title("OpenAI + PostgreSQL + Folium Streamlit App")

#     # User input for states, latitude, and longitude range
#     states = st.text_input("Enter the state code", value="CA")
#     lat_range = st.slider("Select Latitude Range", -90.0, 90.0, (-90.0, 90.0))
#     lon_range = st.slider("Select Longitud Range", -180.0, 180.0, (-180.0, 180.0))

#     # Estimate expected visits for a new library
#     expected_visits = estimate_expected_visits(states, lat_range, lon_range)

#     # Display the result
#     st.success(f"The expected visits for a new library in the specified region are: {expected_visits:.2f} visits")

#     # ...

# if __name__ == "__main__":
#     main()

# # Streamlit app
# def main():
#     st.title("OpenAI + PostgreSQL + Folium Streamlit App")

#     # User input for visits and states
#     visits = st.number_input("Enter the number of visits", min_value=0, step=1, value=100)
#     states = st.text_input("Enter the state code", value="CA")

#     # Run the conversation and get the libname
#     libname = run_conversation(visits, states)

#     # Display the result
#     st.success(f"The libname is: {libname}")

#     # Create a Folium map
#     m = folium.Map(location=[37.7749, -122.4194], zoom_start=10)

#     # Add a marker for the specified location
#     if libname:
#         st.sidebar.subheader("Library Location")
#         st.sidebar.text(f"{libname} - {states}")

#         # You can customize the marker based on the actual data
#         marker = folium.Marker(location=[37.7749, -122.4194], popup=libname)
#         marker.add_to(m)

#         # Add marker cluster for better visualization if needed
#         marker_cluster = MarkerCluster().add_to(m)

#     # Display the map
#     folium_static(m)

# if __name__ == "__main__":
#     main()
