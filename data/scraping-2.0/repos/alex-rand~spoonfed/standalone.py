from openai_api_functions import * 
from anki_connect_functions import * 
from data_cleaning_functions import * 
from narakeet_functions import * 
from sqlite_functions import * 

def main(gpt_model, audio_provider):

    # Load the anki cards from two decks:
    #   - The deck with words or phrases I've learned or know 'known vocab'
    #   - The deck with words or phrases I haven't learned yet (and aren't yet contextualized in a sentence) 'new vocab'

    # Declare 
    known_vocab_deck = "देवनागरी::मैंने सीखा"
    known_card_types_and_fields = {
        "Basic-10b04": ["Front", "Back"],
        "Memrise - ! Hindi Alphabet (Devanagri) (audio) ! - Hindi": ["Hindi", "English"]
    }
    
    new_vocab_deck = "देवनागरी::मैं सीखना चाहता हूँ"
    new_card_types_and_fields = {
        "Basic-10b04": ["Front", "Back"]
    }

    known_vocab = load_vocab_from_deck(known_vocab_deck, known_card_types_and_fields)
    new_vocab = load_vocab_from_deck(new_vocab_deck, new_card_types_and_fields)

    # Make sure there's no overlap between the known and new vocab lists by mistake
    new_vocab = new_vocab[~new_vocab.isin(known_vocab)]

    print(len(known_vocab))
    print(known_vocab)
    #return(0)
    # Ask GPT to generate some new sentences
    # gpt_payload = gpt__generate_new_sentences(known_vocab, new_vocab, 10, "gpt-4")

    ### SCRATCH ###

    # # Export raw payload for debugging
    # with open("test-payload", 'w') as file:
    #     for item in gpt_payload:
    #         file.write(f"{item}\n")
    # 


    # Save a test version so we don't spend money to debug
    #with open('test-payload.txt', 'w') as f:
    #    f.write(repr(gpt_payload))
    import ast
    # Open the file in read mode ('r')
    with open('test-payload.txt', 'r') as f:
        # Read the entire file to a string
        test_payload = f.read()
    gpt_payload = ast.literal_eval(test_payload)

    ### /END SCRATCH ###

    # GPT doesn't do great at meeting the criteria. Call some functions to count the number 
    # of known, want-to-learn, and 'rogue' words in each sentence, so we can filter later.
    gpt_payload_enhanced = evaluate_gpt_response(gpt_payload, known_vocab, new_vocab)

    # Flag sentences that don't meet the N+1 rule, or any other rule you might prefer
    gpt_payload_enhanced = flag_bad_sentences(gpt_payload_enhanced, "n+1 with rogue")
    
    # SCRATCH Output for scratch diagnostics
  #  gpt_payload_enhanced.to_csv('cleaned_response.csv', index=False)

    # Update the database
    save_to_database("database.db", gpt_payload_enhanced, gpt_model, audio_provider) 

    # Subset only the sentences that fit the criteria
    keepers = gpt_payload_enhanced[gpt_payload_enhanced['meets_criteria'] == True]
    
    # Call the audio-generating API, and add the filenames to the 'keepers' table
  #  generate_audio(keepers, language="Hindi", anki_profile_name="Alex")
 
    # Create a new Anki card for each keeper
    keepers.apply(create_new_card, args=(gpt_model, audio_provider), axis=1)
   
   # Remove the keepers from the new cards deck

main("gpt-4", "narakeet")



    # Print the resulting DataFrame
    # print(keepers.columns.tolist())
    # print(keepers_with_audio['audio'])

    # Should I now call Anki and delete any cards in the 'to learn' deck that are included a word that I just generated and that meets criteria?


    # Be sure to use an existing card format I've cleaned in the load_known_vocab() function

