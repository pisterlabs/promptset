import os
import openai



API_KEY = ""

openai.api_key = API_KEY

some_text = "A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei."
metaverse_text = "In futurism and science fiction, the metaverse is a hypothetical iteration of the Internet as a single, universal and immersive virtual world that is facilitated by the use of virtual reality (VR) and augmented reality (AR) headsets.[1][2] In colloquial use, a metaverse is a network of 3D virtual worlds focused on social connection.[2][3][4] The term 'metaverse' originated in the 1992 science fiction novel Snow Crash, as a portmanteau of 'meta' and 'universe'.[5][6][page needed] Metaverse development is often linked to advancing virtual reality technology due to increasing demands for immersion.[7][8][9] Recent interest in metaverse development is influenced by Web3,[10][11] a concept for a decentralized iteration of the internet. Web3 and the Metaverse have been used as buzzwords[1][12] to exaggerate development progress of various related technologies and projects for public relations purposes.[13] Information privacy, user addiction, and user safety are concerns within the metaverse, stemming from challenges facing the social media and video game industries as a whole.[1][14][15]. Components of metaverse technology have already been developed within online video games.[16] The 2003 virtual world platform Second Life is often described as the first metaverse,[17][18] as it incorporated many aspects of social media into a persistent three-dimensional world with the user represented as an avatar, however historical claims of metaverse development started soon after the term was coined. Early projects included Active Worlds[19] and The Palace. Popular games described as part of the metaverse include Habbo Hotel,[7] World of Warcraft,[20] Minecraft,[7] Fortnite,[21] VRChat,[22][23] and game creation platform Roblox[24][25][26] which has since employed significant usage of the term in marketing.[27] In a January 2022 interview with Wired, Second Life creator Philip Rosedale described metaverses as a three-dimensional Internet that is populated with live people.[28] Social interaction and 3D virtual worlds are often an integral feature in many massively multiplayer online games. In 2017, Microsoft acquired the VR company AltspaceVR,[29] and has since implemented virtual avatars and meetings held in virtual reality into Microsoft Teams.[30] In 2019, the social network company Facebook launched a social VR world called Facebook Horizon.[31] In 2021, Facebook was renamed 'Meta Platforms' and its chairman Mark Zuckerberg[32] declared a company commitment to developing a metaverse.[33] Many of the virtual reality technologies advertised by Meta Platforms remain to be developed.[34][35][36] Facebook whistleblower Frances Haugen criticised the move, adding that Meta Platforms' continued focus on growth-oriented projects is largely done to the detriment of ensuring safety on their platforms.[37] Meta Platforms has also faced user safety criticism regarding Horizon Worlds due to sexual harassment occurring on the platform.[38][39][40] In 2021, Meta made a loss of over $10 billion on its metaverse development department, with Mark Zuckerberg saying he expected operating losses to 'increase meaningfully' in 2022.[41] Some metaverse implementations rely on digital currencies, and often cryptocurrency. Assets within the metaverse are sometimes traded as non-fungible tokens (NFTs) and track ownership using blockchain technology.[42] Proposed applications for metaverse technology include improving work productivity,[30][43] interactive learning environments,[14] e-commerce,[14][44] real estate[14] and fashion.[45]"

response_1 = openai.Completion.create(
  model="text-davinci-002",
#   prompt=f"Create 5 study questions from this text: {some_text}",
#   prompt="Create 5 study questions for this topic: neutron star",
  prompt=f"Summarize this into 5 sentences: {metaverse_text}",
  temperature=0.7,
  max_tokens=256,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print(response_1["choices"][0])

response_2 = openai.Completion.create(
  model="text-davinci-002",
  prompt=f"Create 5 study questions from this text: {response_1}",
  temperature=0.6,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.3,
  presence_penalty=0.0
)

print(response_2["choices"][0]["text"])

response = openai.Completion.create(
  model="text-davinci-002",
  prompt=f"Verify correctness of this answers based on the provided text \n Question: What is metaverse? \n Answer: Metaverse is a virtual reality. Text:{metaverse_text}",
  temperature=0.6,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.3,
  presence_penalty=0.0
)

print(response["choices"][0]["text"])
