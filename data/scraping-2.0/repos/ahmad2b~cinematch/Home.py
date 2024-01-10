import json
import streamlit as st

from pydantic import ValidationError

from db import create_database_connection, UserOperations, MovieOperations, UserBase
from tmdb_api import MovieDB, MovieResponse
from openai_api import AsyncManager, OpenAIBot, MessageItem

database_engine = create_database_connection()
user_operations = UserOperations(database_engine)
movie_operations = MovieOperations(database_engine)

if "username" not in st.session_state:
    st.session_state["username"] = ""

if "user" not in st.session_state:
    st.session_state["user"] = None

async_manager = AsyncManager()
movie_database = MovieDB()
openai_bot = OpenAIBot()

st.title("Cinematch: Your Movie Mood Matcher :popcorn:")
st.markdown("---")
st.write(
    """
Cinematch is an interactive movie recommendation system built using Streamlit and the OpenAI/Gemini API. 
It leverages the power of AI to provide personalized movie recommendations based on user preferences.
"""
)

sidebar = st.sidebar

# Sidebar for user login
with sidebar:
    if st.session_state["username"] != "":
        st.subheader(f"Welcome, {st.session_state['username']}! :wave:")
        if st.button("Logout"):
            st.session_state["username"] = ""

        st.header("Your Watch-list 🎬")
        wishlist = movie_operations.get_movies_for_user(st.session_state["user"].id)
        for movie in wishlist:
            st.markdown("---")
            st.image(f"https://image.tmdb.org/t/p/w500{movie.image}", width=80)
            st.markdown(f"**{movie.title}**")
            if st.button("Remove ❌", key=f"delete_{movie.id}"):
                movie_operations.delete_movie_by_id(
                    st.session_state["user"].id, movie.id
                )
                st.success(f"**{movie.title}** has been removed from your watch-list.")
    else:
        if "show_form" not in st.session_state:
            st.session_state["show_form"] = "login"

        if st.button("Join Cinematch! 🎉"):
            st.session_state["show_form"] = "signup"

        if st.button("Already a member? Login 🔑"):
            st.session_state["show_form"] = "login"

        if st.session_state["show_form"] == "login":
            with st.form(key="login_form"):
                st.subheader("Login")
                st.write("Please enter your username and password to login.")
                username = st.text_input("👤 Username")
                password = st.text_input("🔒 Password", type="password")
                submit_button = st.form_submit_button("🚀 Login")

                if submit_button:
                    # Authenticate user
                    response = user_operations.authenticate_user(username, password)
                    if response["status"] == "success":
                        st.session_state["username"] = username
                        st.session_state["user"] = response["user"]
                        st.success(f"{username} Logged in successfully! 🎉")
                    else:
                        st.error(response["message"])
        elif st.session_state["show_form"] == "signup":
            with st.form(key="signup_form"):
                st.subheader("Sign Up")
                st.write("Please fill in the following details to create an account.")
                username = st.text_input("👤 Username")
                password = st.text_input("🔒 Password", type="password")
                submit_button = st.form_submit_button(":door: Sign Up")

                if submit_button:
                    try:
                        # Create user
                        user_data = UserBase(username=username, password=password)
                        response = user_operations.register_new_user(user_data)
                        if response["status"] == "success":
                            st.success(
                                f"{username} Registered successfully! \n Please login to continue"
                            )
                        else:
                            st.error(response["message"])
                    except ValidationError as e:
                        for error in e.errors():
                            st.error(
                                f"Validation error in field '{error['loc'][0]}': {error['msg']}"
                            )


# User input for movie preferences
st.header("Help us help you find your next movie :tv:")

available_movie_genres = movie_database.get_movie_genres()
list_of_genres = [genre.name for genre in available_movie_genres.genres]

with st.form(key="movie_preferences_form"):
    selected_movie_genres = st.multiselect(
        "What's your flavor for the day? :icecream:",
        list_of_genres,
        placeholder="Pick a genre, any genre!",
    )

    user_movie_preference = st.text_input(
        "Got a movie idea in mind? :thinking_face:", placeholder="Type it here!"
    )
    movie_rating_range = st.slider(
        "How picky are we feeling today? :sunglasses:", 0.0, 10.0, (0.0, 10.0)
    )

    submit_button = st.form_submit_button("Find my movie match! :heart:")

    # Button to start the recommendation process
if submit_button:
    # get keywords and query from the openai bot
    message = f"{[selected_movie_genres]} + {[user_movie_preference]} + {[str(movie_rating_range[0]), str(movie_rating_range[1])]}"
    query = openai_bot.send_message(message)
    # print("query: ", query)

    if openai_bot.isCompleted():
        # print("completed: ")
        _response: MessageItem = openai_bot.get_lastest_response()
        # print("response: ", _response.content)
        content = _response.content.strip("`").replace("json", "").strip()

        params = json.loads(content)
        # st.markdown(params)

        movies = movie_database.discover_movies_with_params(params)
        # print("movies: ", movies)

        # Store the search results in the session state
        st.session_state["search_results"] = movies.results

        if "search_results" in st.session_state:
            for movie in st.session_state["search_results"]:
                st.markdown(f"<div >", unsafe_allow_html=True)

                # Display movie poster
                st.image(
                    f"https://image.tmdb.org/t/p/w500{movie.poster_path}", width=340
                )  # Adjust the width as needed

                # Display movie title
                st.markdown(
                    f"<div style='font-size: 20px; font-weight: bold; overflow: hidden; text-overflow: ellipsis'>{movie.title}</div>",
                    unsafe_allow_html=True,
                )

                # Display movie details
                st.markdown(
                    f"<div style='color: #f4a261; font-size: 16px; margin-top: 8px; margin-bottom: 4px'>Rating: <b>{movie.vote_average}</b></div>",
                    unsafe_allow_html=True,
                )
                st.write(movie.release_date)

                if st.button(
                    "Add to my Watch-list! :popcorn:",
                    key=f"watchlist_button_{movie.title}",
                ):
                    if (
                        "user" in st.session_state
                        and st.session_state["user"] is not None
                    ):
                        movie_operations.add_movie_for_user(
                            st.session_state["user"].id, movie.title, movie.poster_path
                        )
                        st.success(
                            f"**{movie.title}** is on your watch-list! :partying_face:"
                        )
                        st.balloons()
                    else:
                        st.error(
                            "You must be logged in to add movies to your watchlist."
                        )


st.markdown("---")

# Get the movies from the API
movies = movie_database.discover_movies()

st.header("Here are some movies you might fancy :film_projector:")

# Display each movie in a grid
for i in range(
    0, len(movies.results), 2
):  # Adjust the step size to change the number of columns
    cols = st.columns(2)  # Adjust the number of columns as needed

    for j in range(2):  # Adjust the range to match the number of columns
        if i + j < len(movies.results):
            movie = movies.results[i + j]
            with cols[j]:
                st.markdown(f"<div >", unsafe_allow_html=True)

                # Display movie poster
                st.image(
                    f"https://image.tmdb.org/t/p/w500{movie.poster_path}", width=340
                )  # Adjust the width as needed

                # Display movie title
                st.markdown(
                    f"<div style='font-size: 20px; font-weight: bold; overflow: hidden; text-overflow: ellipsis'>{movie.title}</div>",
                    unsafe_allow_html=True,
                )

                # Display movie details
                st.markdown(
                    f"<div style='color: #f4a261; font-size: 16px; margin-top: 8px; margin-bottom: 4px'>Rating: <b>{movie.vote_average}</b></div>",
                    unsafe_allow_html=True,
                )
                st.write(movie.release_date)
                # st.markdown(f"<div style='color: #808080; font-size: 16px;'>{movie.overview}</div>", unsafe_allow_html=True)

                if st.button(
                    "Add to my Watch-list! :popcorn:",
                    key=f"watchlist_button_{movie.id}",
                ):
                    if st.session_state["user"] is not None:
                        movie_operations.add_movie_for_user(
                            st.session_state["user"].id, movie.title, movie.poster_path
                        )
                        st.success(
                            f"**{movie.title}** is on your watch-list! :partying_face:"
                        )
                        st.balloons()
                    else:
                        st.error(
                            "You must be logged in to add movies to your watchlist."
                        )
