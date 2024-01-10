import openai  # type: ignore

token: str = ""
cache: str = "mem://"
prefix: str = ","
owners: list[int] = [1129559813144191096]
openai.api_key: str = ""


class Color:
    neutral: int = 0x2B2D31
    approval: int = 0xA9E97A
    error: int = 0xFFCC00


class Emoji:
    class Paginator:
        navigate: str = "<:navigate:1126195978622480436>"
        previous: str = "<:left:1127669838089834677>"
        next: str = "<:next:1126196000579665951>"
        cancel: str = "<:cancel:1126195945864957982>"

    approve: str = "<:approve:1126212283161387150>"
    deny: str = "<:deny:1126212292787310752>"
    warn: str = "<:warn:1126212303851892756>"


class Database:
    host: str = "localhost"
    port: int = 5432
    user: str = "vampire"
    name: str = "lain"
    password: str = ""


class Authorization:
    class Spotify:
        client_id: str = ""
        client_secret: str = ""

    google_cloud: str = ""
    fortnite: str = ""
    henrik: str = ""


class API:
    keys: list[str] = [
        "sipher_Qx4IOl2pu1HKzFqUQzvk2mKCClc44aP6T2vrg066oZ48JnJkUjmWqDM6znP4vyyK",
        "claqz_Qt5c73ExW5pEtjjCknvL4djlPUK6RA1D4Ll4vx8SJcp1HPfFchYiWCarMHtuNCbm",
        "rx_yke2mapidBZpUWfBfYtbjXmYFB49XButIh8zYjdAkL5q3MxBmY8J5tBpDiqlCEt9",
        "carter_lDgsUMqxYuLr25SI3YmGtIOJB3cMdR1aERNsrrQzMdaNFQBt4QMv9Go4ucYR8hgL",
        "artist_W88ExDSEEY1HttkQUou3N27xI7HZVPrsRXQ3OD0iBEekzX4EGpKUrgdDdgCUIiO7",
        "nunu_rz8BYQH0B22TsPQrPyI2Fl6zUwlJFWsovG5nIcqsx24AnkDe35Td6r54Ri3O1H1u",
    ]


class Lavalink:
    host: str = "0.0.0.0"
    port: int = 3030
    password: str = "youshallnotpass"
    spotify_client_id: str = Authorization.Spotify.client_id
    spotify_client_secret: str = Authorization.Spotify.client_secret
