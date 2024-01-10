from langchain.tools import BaseTool
from ...weatherapi.api import OpenWeatherMapApi
from ...schema import LatLng, UnavailableUserLocationError


class LocateUserTool(BaseTool):
    user_lat_lng: LatLng | None

    def __init__(self, user_lat_lng: LatLng | None):
        super().__init__(
            name="locate-user",
            description="Use this tool when you need to find out where the user is. Use this when the human mentions 'my city', 'my location', 'here', 'my air', etc.",
            user_lat_lng=user_lat_lng,  # type: ignore
        )

    def _run(self, input: str) -> str:
        raise NotImplementedError()

    async def _arun(self, input: str) -> str:
        lat_lng = self.parse_location()

        # This bubbles up all the way to asking the user for permissions
        if not lat_lng:
            raise UnavailableUserLocationError()

        place = await OpenWeatherMapApi().rev_geocode(
            LatLng(lat=lat_lng.lat, lng=lat_lng.lng)
        )
        return f"{place.name}, {place.state}, {place.country_code}"

    def parse_location(self) -> LatLng | None:
        if (
            self.user_lat_lng is None
            or self.user_lat_lng.lat == 0
            or self.user_lat_lng.lng == 0
        ):
            return None

        try:
            float(self.user_lat_lng.lat)
            float(self.user_lat_lng.lng)
            return self.user_lat_lng
        except ValueError:
            return None
