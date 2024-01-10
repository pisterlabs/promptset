import modal
import os

stub = modal.Stub(name="Instagram")
image = modal.Image.debian_slim().pip_install("apify-client", "spacy", "python-dateutil", "langchain", "supabase", "openai", "tiktoken")

club_ins_url = [
    'https://www.instagram.com/rice.csa/',
    'https://www.instagram.com/tedxriceu/',
    'https://www.instagram.com/riceapps/',
    'https://www.instagram.com/rice_boxing/',
    'https://www.instagram.com/astr.mag/',
    'https://www.instagram.com/basmatibeats/',
    'https://www.instagram.com/basyk.rice/?hl=en',
    'https://www.instagram.com/chimacappella/',
    'https://www.instagram.com/ricecraftandcare/',
    'https://www.instagram.com/arthistoryrice/',
    'https://www.instagram.com/ktruriceradio/',
    'https://www.instagram.com/losbuhosdelnorte/',
    'https://www.instagram.com/ricelowkeys/',
    'https://www.instagram.com/rice_mobstagram/',
    'https://www.instagram.com/mariachilunallena/',
    'https://www.instagram.com/moodystudentcollaborative/',
    'https://www.instagram.com/ricenocturnal/',
    'https://www.instagram.com/paint4strength/',
    'https://www.instagram.com/ricerawphotography/',
    'https://www.instagram.com/riceartclub/',
    'https://www.instagram.com/riceballroom/',
    'https://www.instagram.com/ricebhangra/',

    'https://www.instagram.com/rice.ksa/',
    'https://www.instagram.com/ricethresher/',
    'https://www.instagram.com/rice_wcvb/',
    'https://www.instagram.com/ricecsclub/',
    'https://www.instagram.com/ricedatasci/',
    'https://www.instagram.com/ricerallyclub/',
    'https://www.instagram.com/riceconsulting.ug/',
    'https://www.instagram.com/ricesas/',
    'https://www.instagram.com/riceathletics/',
    'https://www.instagram.com/ricearchimarket/',
    'https://www.instagram.com/riceapollos/',
    'https://www.instagram.com/ricealliance/',
    'https://www.instagram.com/p/Cw7qKqgOJE7/',
    'https://www.instagram.com/dfa.rice/',
    'https://www.instagram.com/ricebusinesssociety/',
    'https://www.instagram.com/rbwo_ricebiz/',
    'https://www.instagram.com/rice_business/',
    'https://www.instagram.com/riceblockchainclub/',
    'https://www.instagram.com/rice.bma/',
    'https://www.instagram.com/ricebsa/',
    'https://www.instagram.com/ricecheer/',
    'https://www.instagram.com/ricecsters/',
    'https://www.instagram.com/ricecompsci/',
    'https://www.instagram.com/ricechinesetheatre/',
    'https://www.instagram.com/riceclimatealliance/',
    'https://www.instagram.com/ricecampanile/',
    'https://www.instagram.com/riceclarinets/',
    'https://www.instagram.com/riceeshipclub/',
    'https://www.instagram.com/sushi_samurice/',
    'https://www.instagram.com/rice_clubtennis/',
    'https://www.instagram.com/ricedancetheatre/',
    'https://www.instagram.com/riceu_design/',
    'https://www.instagram.com/ricedems/',
    'https://www.instagram.com/rice_d2klab/',
    'https://www.instagram.com/riceescape/',
    'https://www.instagram.com/riceeclipse/',
    'https://www.instagram.com/rice_events/',
    'https://www.instagram.com/ricewindenergy/',
    'https://www.instagram.com/rice.rev/',
    'https://www.instagram.com/ricerecreation/',
    'https://www.instagram.com/rice_bhaktiyoga/',
    'https://www.instagram.com/frenchclub.rice/',
    'https://www.instagram.com/rice_fencing/',
    'https://www.instagram.com/riceccd/',
    'https://www.instagram.com/ricegolf/',
    'https://www.instagram.com/riceinternationals/',
    'https://www.instagram.com/rice_italianclub/',
    'https://www.instagram.com/ricejapaneseclub/',
    'https://www.instagram.com/ricejewishstudies/',
    'https://www.instagram.com/ricelions/',
    'https://www.instagram.com/ricemusicmds/',
    'https://www.instagram.com/ricemsne/',
    'https://www.instagram.com/riceneuro/',
    'https://www.instagram.com/riceneurotransmitter/',
    'https://www.instagram.com/riceneurosociety/',
    'https://www.instagram.com/riceswimming/',
    'https://www.instagram.com/riceowlsdanceteam/',
    'https://www.instagram.com/riceolyweightlifting/',
    'https://www.instagram.com/rice_rcssa/',
    'https://www.instagram.com/rice.republicans/',
    'https://www.instagram.com/ricetaiwanese/',
    'https://www.instagram.com/rice.theater/',
    'https://www.instagram.com/ricetabletennis/',
    'https://www.instagram.com/rice_mock/',
    'https://www.instagram.com/ricevolleyball/',
    'https://www.instagram.com/ricewomeninbusiness/',
    'https://www.instagram.com/rice_womeninstem/',

    'https://www.instagram.com/ricemusiccollective/',
    'https://www.instagram.com/ricesailing/',
]

if stub.is_inside():
    import openai
    from apify_client import ApifyClient
    import spacy
    from dateutil.parser import parse
    import datetime
    from langchain.schema.document import Document
    from supabase import create_client, Client
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import SupabaseVectorStore
    os.system('python -m spacy download en_core_web_sm')

if modal.is_local():
    from dotenv import load_dotenv

    load_dotenv()
    stub.data_dict = modal.Dict.new({
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "APIFY_API_TOKEN": os.getenv("APIFY_API_TOKEN"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY")
    })


@stub.function(schedule=modal.Period(days=1), image=image)
def get_ins():
    nlp = spacy.load("en_core_web_sm")

    # Initialize the ApifyClient with API token
    client = ApifyClient(stub.app.data_dict["APIFY_API_TOKEN"])

    # Prepare the Actor input
    run_input = {
        "directUrls": club_ins_url,
        "resultsLimit": 1,
        "resultsType": "posts",
        "searchLimit": 1,
        "searchType": "hashtag"
    }

    # Run the Actor and wait for it to finish
    run = client.actor("shu8hvrXbJbY3Eb9W").call(run_input=run_input)

    pages = []
    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        if ('caption' not in item) or ('url' not in item) or ('timestamp' not in item) or ('ownerFullName' not in item):
            continue

        organizer = item['ownerFullName']
        caption = item['caption']
        url = item['url']
        publish_date = item['timestamp']

        # Convert the publish_date to a datetime object
        base_date = parse(publish_date)

        doc = nlp(caption)
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

        if len(dates) > 0:
            # print(f'publish day: {publish_date}')
            # print(f'dates appeared: {dates}')
            # print(f'caption: {caption}')

            parsed_date = None

            # Try to parse and see if any date is valid
            for date in dates:
                try:
                    # Parse extracted date strings with dateutil to get datetime object using base_date as default
                    if parsed_date == None:
                        parsed_date = parse(date, fuzzy=True, default=base_date)

                except Exception as e:
                    pass

            if parsed_date:
                # Check if event is in the future
                if parsed_date.date() > datetime.datetime.now().date():
                    meta_data = {'date': parsed_date.strftime("%Y-%m-%d"), 'source': url}
                    content = f"Event name: {organizer}'s event\nOrganizer: {organizer}\nDescription: {caption}\nTime: {parsed_date.strftime('%Y-%m-%d')}\nWebsite: {url}"
                    page = Document(page_content=content)
                    # page.page_content = content
                    page.metadata = meta_data
                    pages.append(page)
    db_client: Client = create_client(stub.app.data_dict["SUPABASE_URL"], stub.app.data_dict["SUPABASE_SERVICE_KEY"])
    embeddings = OpenAIEmbeddings(openai_api_key=stub.app.data_dict["OPENAI_API_KEY"])
    vector_store = SupabaseVectorStore(client=db_client,
                                       embedding=embeddings,
                                       table_name='events')
    vector_store.add_documents(pages)
