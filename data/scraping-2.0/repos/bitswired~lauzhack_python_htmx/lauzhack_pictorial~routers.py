import base64
from pathlib import Path
from typing import Annotated, Optional
from uuid import uuid4

import aiofiles
from litestar import Controller, Request, Router, get, post
from litestar.datastructures import Cookie, State
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException
from litestar.params import Body
from litestar.response import Redirect, Template
from litestar.status_codes import HTTP_401_UNAUTHORIZED
from openai import AsyncClient

from .db.models import User
from .dtos import CreateUserDto, GenerateImageDto
from .state import AppState

# Initialize an asynchronous client for the OpenAI API.
openai_client = AsyncClient()


class MainController(Controller):
    """
    The MainController handles the main routes for the application, such as the homepage,
    user login, logout, and signup. It utilizes decorators to link methods to HTTP request types.
    """

    # Base path to associate with the controller.
    path = "/"

    @get()
    async def index_view(
        self, request: Request[Optional[User], str, State]
    ) -> Template:
        """
        Renders the homepage with a welcoming template, passing the user context if available.

        Args:
            request: The current request object, which may include a user instance.

        Returns:
            A rendered template response to be displayed as the homepage.
        """
        # Return the primary template with context data, including user information.
        return Template(template_name="base.html", context={"user": request.user})

    @get("/logout")
    async def logout(self) -> Redirect:
        """
        Logs out the current user by deleting the session cookie and redirects to the homepage.

        Returns:
            A Redirect response leading to the homepage, with session cookie deletion.
        """
        response = Redirect("/")
        response.delete_cookie("pictorial-session")
        return response

    @get("/login")
    async def login_view(
        self, request: Request[Optional[User], str, State]
    ) -> Template:
        """
        Presents the login page if the user is not already logged in.

        Args:
            request: The current request object, which may include a user instance.

        Returns:
            A login page template if the user isn't authenticated; otherwise, redirect to homepage.
        """
        return Template(template_name="login.html", context={"user": request.user})

    @post("/login")
    async def login(
        self,
        data: Annotated[
            CreateUserDto, Body(media_type=RequestEncodingType.URL_ENCODED)
        ],
        state: AppState,
    ) -> Redirect:
        """
        Processes the login form, authenticates the user, and redirects to the homepage or back to login page if failed.

        Args:
            data: DTO containing user login details such as email and password.
            state: The application state holding the repository to interact with the database.

        Returns:
            A Redirect response either to the homepage on successful login or back to login page.
        """
        user = await state.repository.get_user_by_credentials(data.email, data.password)
        if not user:
            raise HTTPException(
                detail="Invalid Credentials", status_code=HTTP_401_UNAUTHORIZED
            )

        redirect_response = Redirect("/")
        redirect_response.cookies.append(
            Cookie(key="pictorial-session", value=user.id, httponly=True)
        )
        return redirect_response

    @get("/signup")
    async def signup_view(
        self, request: Request[Optional[User], str, State]
    ) -> Template:
        """
        Renders the user registration or signup page.

        Args:
            request: The current request object, which may include a user instance.

        Returns:
            A template response with the signup form.
        """
        return Template(template_name="signup.html", context={"user": request.user})

    @post("/signup")
    async def create_user(
        self,
        data: Annotated[
            CreateUserDto, Body(media_type=RequestEncodingType.URL_ENCODED)
        ],
        state: AppState,
    ) -> Redirect:
        """
        Takes a CreateUserDto with user details, saves it to the database, and redirects to the login page.

        Args:
            data: DTO with registration details such as email and password.
            state: The application state with the repository for database access.

        Returns:
            A Redirect response to the login page following successful registration.
        """
        user_id = await state.repository.create_user(data.email, data.password)
        return Redirect("/login")


# Router for main application routes, delegating requests to the MainController.
main_router = Router(path="/", route_handlers=[MainController])


async def save_image(b64_string: str) -> (str, str):
    """
    Saves the base64-encoded image as a PNG file to the local filesystem.

    Args:
        b64_string (str): The base64-encoded string of the image data.

    Returns:
        Tuple[str, str]: A tuple containing the unique identifier for the image and the file path.
    """
    # Decode the base64 image string to binary data.
    image_data = base64.b64decode(b64_string)

    # Create a unique filename using UUID and construct the file path in the 'static' directory.
    name = str(uuid4())
    path = Path("static") / f"{name}.png"

    # Write the image data to a file asynchronously.
    async with aiofiles.open(path, "wb") as file:
        await file.write(image_data)

    # Return the image ID and the file path.
    return name, str(path)


class GenerateController(Controller):
    """
    The GenerateController is responsible for handling routes related to image generation,
    including rendering the image generation page and processing generation requests.
    """

    path = "/"

    @get()
    async def index_view(
        self, request: Request[Optional[User], str, State]
    ) -> Template:
        """
        Renders the image generation input form.

        Args:
            request (Request): The HTTP request object containing user and state data.

        Returns:
            Template: Renders the page where users can input prompts for image generation.
        """
        return Template(
            template_name="generate/index.html", context={"user": request.user}
        )

    @post("image")
    async def generate_image(
        self,
        request: Request[Optional[User], str, State],
        data: Annotated[
            GenerateImageDto, Body(media_type=RequestEncodingType.URL_ENCODED)
        ],
        state: AppState,
    ) -> Template:
        """
        Generates an image based on user input and saves it.

        Receives user input from the htmx-powered interactivity, forwards it to the
        OpenAI API to generate an image, then saves the image and outputs the results.

        Args:
            request (Request): The HTTP request object containing user and state data.
            data (GenerateImageDto): DTO carrying the prompt for the OpenAI Image Generation API.
            state (AppState): The shared state containing the repository for database operations.

        Returns:
            Template: Renders the page displaying the generated image and relevant information.
        """
        # Generate an image using the OpenAI API based on the supplied prompt.
        res = await openai_client.images.generate(
            model="dall-e-3",
            prompt=data.prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )

        # Save the generated image using the helper function.
        b64_string = res.data[0].b64_json
        img_id, img_path = await save_image(b64_string)

        # Persist the generation metadata in the database.
        await state.repository.create_generation(request.user.id, img_id, data.prompt)

        # Render the generated image output.
        return Template(
            template_name="generate/generate-image-output.html",
            context={"prompt": data.prompt, "url": f"/{img_path}"},
        )


# The router that handles requests coming to the '/generate' endpoint
generate_router = Router(path="/generate", route_handlers=[GenerateController])


class LibraryRouter(Controller):
    """
    The LibraryRouter manages routes related to the user's library of generated items.
    This is where users can view their previously generated content.
    """

    path = "/"

    @get()
    async def index_view(
        self, request: Request[Optional[User], str, State], state: AppState
    ) -> Template:
        """
        Renders the library view which showcases the user-generated content.

        A list of items generated by the user is fetched from the database,
        and this page presents those items.

        Args:
            request (Request): The HTTP request object containing user and state data.
            state (AppState): The shared state containing the repository for database operations.

        Returns:
            Template: A rendered template of the library page with the user's generated content.

        Notes:
            - This route handler assumes that a check has already been made to ensure that
              the user is logged in; if not, it should redirect to the login page.
            - The 'request.user.id' attribute is used, which implies that the 'user'
              should have been set in the request state by an authentication middleware.
        """
        # Retrieve the list of generated content for the current user.
        generations = await state.repository.get_user_generations(request.user.id)

        # Output the user generations to the console (useful for debugging purposes).
        print(generations)

        # Render the library template, passing the necessary context.
        return Template(
            template_name="library/index.html",
            context={"user": request.user, "generations": generations},
        )


# The Router handles requests directed at '/library' and delegates them to the LibraryRouter.
library_router = Router(path="/library", route_handlers=[LibraryRouter])
