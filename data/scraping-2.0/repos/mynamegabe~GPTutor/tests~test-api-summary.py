from revChatGPT.ChatGPT import Chatbot
import requests
import openai
import sqlite3

SESSION_TOKEN = "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..jTdpmUzyWKhUnUl8.yYFBVmqM9kGU6XXCzzpgavw0EZsRWeZqLIQlDb1iMU-cjRZXHon7qnd9tCS15IX2mXf7bHW8r7zewe-kUcfaaSMSTBbDskNYlrBgKFt0UabAqgB-UlOxIGqE1xrv45sNDDp4p3osJbX9hfEB3n4fAa347C3PrJfzWRSGb-BRNZHjD7Ji5B48TxfHQHIoaQXfH7oG--HoPJ2j4izDJOQDwdyaU2o3WMymTjiLrhPtgstfmbKiAw7y6KNIE8JybDQvMvHGUcFAysgmK3b6sklysGX0xI8AG-Dn18Oru838pBoJQ63gKFBJU1Q8VadNIemomTv1W0tFsvXcOGrCE56mPcqV0UCN20ntcB0YueofswMIDAghbdFhg0Cy0VW4flnPJNHYSdVdmwlM_pzdPzUNX-_H76oDVSuIfEdXH5qKNyGoYD4OPaVsKr5I7sJupUvRNQfV_nivkxNIy-ekRNrcKnF2mkpCcXHma4XKQsc-FAwGVl-3fIdugjJVtYRCVaFBwrVtHiK22_KwWfkHxeMF1zWdqGS5mpxwJ7Go24Ux00CuSLB9rWt2wM9dnjojOnIYy4DqzTHhBcv_F_XoLpJQwGgNqDcR9VV5NaMMN_FHgVIlwVmb_Y2jg5VqwGYy8DdjjlCoXX4UaOEi01QQrXlimpjYdzvdQWsVv1-iabRCS7UFNPqnZmmOH-M78Obg4zTPaN4wK8qIiiZv6b9WJKBNMbYrGeujBJyi-1a5m_IX8gTaA1s0T-WjC4I1cryWrdUzlOxocROTgq-z9DAOM_MtuoLBQguqUX-Zk4Bm0bdm8Ak305p98feWyk4fy3paDGExLJ-18eWznncf8C06V5BnmeZ_E7VqUWm-SX1gR6htjRXHMQvSFwVuTvYSb09fP4DA9zqSpL7H8HACnQ8HtAaz4d_-dTMXnONQcaKgg8LL2BA8l_eGxMz5zStXMJRRxx8Ot23vSwJ5fhpbA0vnwGo6Mmmr6WBwaAiJpQ-jFhTQjqWdHAqDrmwRUNjUAA3ve0ExZnyeZpkR8DtD_8EdB56YcyQYoNxqyPS0rVvkFxg286GOb6DyVyEhXpZz_eYToFU6tOy4R1G-PHDTd5JjSy60GcbasuFFDuRDxZzUws0Pe0yfAHWwleuKOZL6vHOLPnjLgyzKhS2YlDCS0TFufiiYlc65qWqx3hEqPn6ET1n-trXUVKRceOMsJWZ1gND6jhG72MRVgl3joOooQHQ2xsF-Z1shVWhbpF4gBsF1yco0EEtVlmzQrFf0-Li_GjLhs9KSR7jc8an9dt24iHE5sBHAnp3pNPhHeXQxVlqD1AThmBD54II4o7ZgbTFBTRsKt1sEC9O6b9MQ-TqPmUpDN1BVgpRbgPLC_ZbsKylb-zjJ_CPaLyVGUcxhZAiyirsUlABC4qm7BqCTGI3JQJTj71m15C_S5hCyvmjJPOv5zKKsbH3ddgDZ7dA55zqvw-k07lTyHQBZp28OTZcQDcfkJ2sjOQjn2fN9GSqZIBNaFui4CVT9GzQO5KG5Er26PiqGG4Ls4APGgsZzK7EgVsQzOfNvYaHIK0bWP2bc1-onGlZDK4tQ1tpeDdXbigNJ1yB_k50C7J1p1o3lNZNNbC8gjNOQw32F2uUB0CEzuQM9ZzcJv8lORtQ-9nn1IF8oHr3Omj-go93px1fTZEwy4i4STUab3LiHhkD1ORU0kgowU9zwGCPPmwxe6HFaYE9dXG7mS4ddHWWBs5EjZObbKjtPR-m0cKxzeyGStt1BWULtfK6Ah3WaPl2zQS8lrxV4IBmkv0q4g-shxg6T-4FSfO1DvMngb2HfI9mFRD16uDZ63U5Ij_-Fi7GglSbG3Tw6hyimtd0IWo4G4LTldxm_EdivC-z-G4awcZve60zYa_z-GxBY1paD1oY7vw_TTcBQWF68XjdH8-bkj9N-vJdKe-NbM3LmQOyRCFFTvKNR03WIzAFGVDf9wz4svBMAvNDWsQjAuXFVgMr-rhlfVOl4i7YR_MJHZZpyDCf4luHnC5N22vOjeNg2DV2VsP4zoET1InnjLD-ple2SSyPCmA_KJjdaydDqMMA1HrHWJIfQxf2E9gjcnGBwW5aGQ8UcHsdX6wtM1UVaT5at8XzWVg454Vb1FLkAw3FtfNhELveh6-8GjsjjnZXGlRuN0Iv2Ye-yEHxDotzzqkFemWrn0v6w2IPAkiIxRmxClg6v2nDCCi7aYOxEQuv320NZMRjNsCizne1IHI5jIwLOBmmwKUOaD_He-Q1VJnL1DAM0yN9yrrH_JxI.XJWcQfEx7EOBFvK2nom3Ag"
# SESSION_TOKEN = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..sD5Lg0EHL7pqnkaO.Cv_XGsMFJDgC4oIlV3kLtGKIC4rcMjnfQ4e1mZ5KhTdbnOQbIV-AEUUFQ6s6kyEiaKpygj1ka-RcmOw8aatqcq81GhwlOeXdqmuzbkDmd55WlHKmVLkh9x4CDAqfQIYDfCo9vgavINPdOVN6I3DyCTOj403Do1AmFN0cMuuR0ydlvGdnB65SoBjCaGEBX4msRknhvvsj8cls9G7ORCIQelaiNnGD3CsQ4cuiTZY1iAc0F2N0gF4M3iwQXDLo24sKnq8B8fHW-tq1s3Y2YELIhbNiWf30TXxwfYnz5NseZ8ZUw31RSeOpOPppEp_hT8KP2C0wE8pjMFsfhmYNEiqDfZR-lNJdH5yy2Kn_-SoJSNrBagJFR_WsVQv6AiC6WfNTaf9UL9XlGR8yTVBTVUOMIrHTRCqqgxtP8pemITaOOUQPRYSw9UAyFwdFiF7UZ-FCDvFedpx78vQZnhfjqREtY8zVFdMlyLEh37nY-8FupRHYrUSoEylDt6V2mS9iBK42W9Ccu_g67Cts-Ju0QYKdp5BwY4XMnnGEa-Vpb1J8fpRegdiBNMjAKrZQXY_FW2zRxGguBUeePeYQbxA1yaV_2QFHYlqi_6hEmT4B2nyGfn_GbWX7g_CvTdv3qNwmEet3H6qwjBRr0Ut3CB2z8X0SZP0Dl2hSstc_f8GiD2cP7ax5s2YPGoKZnlhz7U3yA7NT9mvxDBOjBn3ddefx5Mmw7s3uQW_JLXifs9dmctfQvcjI5hVtwqF-un7cmA0xYDoCoU2lutbzIl8kiJjrjvfHiq2UwkJOMZrRf_l5FanOy1JBtq0cxKCHtbFunUmsdhs4N4PDV6cSPHbrU00xedi1hAHtq2-OVzcvbMRBICZGfHXNQEguMnWnUKqSKVDGUhZ-T1eVqM61VUtjj-UM_g7om6y41HfX5baxxOrKH8grfNLUY58n6kZUW0o5SNq-CxfJeGgZQlavXYV7BWq_sFah2pdj4Qd0VXh5dbD0kmYSAcfkb8c9Z_nvA5I5xBAO_Qxwj9cHu0RWeJw486_b6HtEwd4NFDz5owKlGrE5QCszxI4YNaVdUXT3BvwlTpnUfzi5NsaFivIVPP5OM9711lGe_iuMaOzOX1QLpZmf8VwvMDwc68_q-Cbc5_SZhHzVWVQYdx8DUK5ALeztds5RNu1I_xCmQKi29ZUKksOut_X_kx3AdPZX2qbFggF_10wzMxt-OJSrTYKatdEp6L2Mao-LyHq-lJJwCA9qfB5C9zAmXCHI_ls34YsKKUwGg4IfTETPZsojfOz-k-2v9iEyoEJIAOZGNPnGLYCkl-y09wapyMyXnJQ2hhlaCHuTMweOOcl6DJHVTEwUtQnFfzAhI4GLO6ShipijfOTYjwKL8eK2ROSFom145KNqzM3-iyt02GhBk3yoCByNYZLi4DbO0SWcwTWM9W59ICVAOwZJ1N-L7wsT8LHsY3ibCizDjFGXrqBQvCUJPw6jvVGocBp13o81ERefw9zfCe_aSjnMF7bPeFulQ20BbVr0JvqDsSCG9A2lwHXvpSjG7Aw9g6zwemkY3fwbVpkr-BOLwp_iT0L08ASxuu8xobhWlClUWMs1DBCI6QnvV59MBchEcKmcX4ACKtlBeStmhdwcA2Ioq__tdzWPI1K7xeHT9NubRz5EW9uykrg7HCDy6O5SWRtLoQao_CZ53V8OWK3hs3thBv2y90Cl33a52ed7frkab9kyn6rpyQhhD5hFs6G9CH2SBXvtaoiM3j_krrZ0lmrZjk8PfZR8zW5ABYHnS-xZk0zq2WkNYbIRPcNROsnl-angZaPiVqe5EMVYv2b2FKGzZ5ly-V1bGyk5n2UPQmULDCYoGqlDLjbDBjresLAcqCLbRSsdyGU13Kk0L7cie57r_Wcu3N9e79EU8cmLCLw6YNG2gbAUzSVrCRfsCOeHgAHv7U2h6NYSfEtOUcj3Hbvh3yLr0VArm5EZH0VmNVWCuPiV3pzD3-AMtFRcu2MMP-GjKXfmybdGCgVDaH9EcUhD9WoKmSol7fDGnozamwmVwNFkxuyLBPVT8XJ7HCsXNgjuK16YMymHL0W3jFJmtk2oR54t3gaTkiZvOgLgpu3FO4nWCM5BRsX56g1exj6W42rxykKYtuS7qvX8jSnqmOtUaElzUIK93mNtYtmY7oSuugdSvU-1qAW3JfNoraJ4Qj5U4PykoIZ-IHwJ0LxlQiqthYqjyve17JpA5AD4aXHDggBmxbK_7tIDqRMx8zOHLgbOSMGrvTPJgyF3HLxQMe_eQQP4GiwPlEuGmCsz_Mnc5XNHBUZtw-vNY3O8ppVqzwGR0phvqev0jB68GeZXOGaykJR8HGo._b8Hglo_hXf0NA7DOzSIsA'
OPENAI_KEY = "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"
openai.api_key = OPENAI_KEY
# Login to OpenAI
model = "text-curie-001"

# Create the Database
conn = sqlite3.connect("./database.db")
cur = conn.cursor()
try:
    cur.execute("CREATE TABLE QNA(id INTEGER PRIMARY KEY, question TEXT, answer TEXT)")
    conn.commit()  # Execute the statement
except:
    print("Table already exists, continuing...")
    pass


def getSessionToken():
    s = requests.Session()
    response = s.post()

    cookies = response.cookies
    sessionToken = cookies.get_dict()["__Secure-next-auth.session-token"]
    return sessionToken


# Initiate the chatbot

chatbot = Chatbot(
    {
        "session_token": SESSION_TOKEN,  ## Comment out when not in use.
        # "email": "hnr.bbw@outlook.com",
        # "password": "securepassword123",
        # "isMicrosoftLogin": True
    },
    conversation_id=None,
    parent_id=None,
)  # You can start a custom conversation


def askGPT(
    prompt, conversation_id=None, parent_id=None
):  # You can start a custom conversation
    try:
        if CONV_ID == "" or PARENT_ID == "":
            response = chatbot.ask(
                prompt, conversation_id=conversation_id, parent_id=parent_id
            )
        else:
            response = chatbot.ask(prompt, conversation_id=CONV_ID, parent_id=PARENT_ID)
        return response["message"]
    except:  # Route to Curie API if load is too high
        ## Running on Curie
        print("================ [!] Running on Curie API ================")
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=1500,
            temperature=0,
            n=1,
        )
    return response


print("Asking ChatGPT Now")
content = "In cryptography, a cipher block chaining message authentication code (CBC-MAC) is a technique for constructing a message authentication code (MAC) from a block cipher. The message is encrypted with some block cipher algorithm in cipher block chaining (CBC) mode to create a chain of blocks such that each block depends on the proper encryption of the previous block. This interdependence ensures that a change to any of the plaintext bits will cause the final encrypted block to change in a way that cannot be predicted or counteracted without knowing the key to the block cipher."
instruction = ""
prompt = (
    "Generate study flashcards for the following content:\n " + content + instruction
)
rsp = askGPT(prompt)
