# Note: you need to be using OpenAI Python v0.27.0+ for the code below to work, with a
import openai
import json
import urllib.request
import logging
import os
import ssl
import scraper.tabroom_scrape as tabroom_scrape
from generate_chat_gpt_prompt import generate_chat_gpt_prompt
from generate_list_generation_prompt import generate_list_generation_prompt
from get_debate_or_congress_results import get_debate_or_congress_results
from get_debate_results_from_rounds_only import get_debate_results_from_rounds_only
from get_speech_results_from_final_places import get_speech_results_from_final_places
from get_speech_results_from_rounds_only import get_speech_results_from_rounds_only
from update_global_entry_dictionary import update_global_entry_dictionary
from parse_arguments import parse_arguments
from group_data_by_school import group_data_by_school
from create_data_strings import create_data_strings

if __name__ == "__main__":
    # Get arguments (no pun intended)
    args = parse_arguments()
    school_name = args.school_name
    tournament_id = args.tournament_id
    all_schools = bool(args.all_schools)
    custom_url = args.custom_url
    read_only = bool(args.read_only)
    percentile_minimum = int(args.percentile_minimum)
    max_results_to_pass_to_gpt = int(args.max_results)
    scrape_entry_records_bool = True

    # DOWNLOAD DATA FROM THE TABROOM API - We'll use a combination of this and scraping
    response = urllib.request.urlopen(  # nosec - uses http
        url=f"http://www.tabroom.com/api/download_data.mhtml?tourn_id={tournament_id}",
        context=ssl._create_unverified_context(),  # nosec - data is all public
    )
    html = response.read()
    data = json.loads(html)

    # SCRAPE TABROOM FOR ALL THE GOOD DATA NOT PRESENT IN THE API
    scrape_output = tabroom_scrape.main(
        tournament_id=tournament_id, scrape_entry_records=scrape_entry_records_bool
    )
    scraped_results = scrape_output["results"]
    name_to_school_dict = scrape_output["name_to_school_dict"]
    code_to_name_dict = scrape_output["code_to_name_dict"]
    name_to_full_name_dict = scrape_output["name_to_full_name_dict"]
    entry_counter_by_school = scrape_output["entry_counter_by_school"]
    school_set = scrape_output["school_set"]
    state_set = scrape_output["state_set"]

    # WALK THROUGH EACH EVENT AND PARSE RESULTS
    data_labels = [
        "event_name",
        "event_type",
        "result_set",
        "entry_name",
        "entry_code",
        "school_name",
        "rank",
        "place",
        "percentile",
        "results_by_round",
    ]
    tournament_results = []
    entry_id_to_entry_code_dictionary = {}
    entry_id_to_entry_entry_name_dictionary = {}
    has_speech = False
    has_debate = False
    for category in data["categories"]:
        for event in category["events"]:
            # Create dictionaries to map the entry ID to an Entry Code and Entry Name
            update_global_entry_dictionary(
                sections=event.get("rounds", [{}])[0].get("sections", []),
                code_dictionary=entry_id_to_entry_code_dictionary,
                entry_dictionary=entry_id_to_entry_entry_name_dictionary,
            )

            # Parse results sets
            if event["type"] in ["debate", "congress"]:
                has_debate = True
                if "result_sets" not in event:
                    tournament_results += get_debate_results_from_rounds_only(
                        code_dictionary=entry_id_to_entry_code_dictionary,
                        entry_dictionary=entry_id_to_entry_entry_name_dictionary,
                        entry_to_school_dict=name_to_school_dict,
                        event=event,
                    )
                else:
                    tournament_results += get_debate_or_congress_results(
                        event=event,
                        code_dictionary=entry_id_to_entry_code_dictionary,
                        entry_dictionary=entry_id_to_entry_entry_name_dictionary,
                        entry_to_school_dict=name_to_school_dict,
                        scraped_data=scraped_results,
                        event_type=event["type"],
                    )
                # TODO - add an option to enrich the results via scraped data - perhaps replacing rounds-only?
            elif event["type"] == "speech":
                has_speech = True
                # If Final Places is published as a result set...
                if "Final Places" in [
                    result_set.get("label", "")
                    for result_set in event.get("result_sets", [{}])
                ]:
                    # Then grab that result set and pass it to the designated parsing function
                    final_results_result_set = [
                        result_set
                        for result_set in event.get("result_sets", [{}])
                        if result_set.get("label", "") == "Final Places"
                    ][0]["results"]
                    tournament_results += get_speech_results_from_final_places(
                        final_results_result_set=final_results_result_set,
                        event_name=event["name"],
                        entry_dictionary=entry_id_to_entry_entry_name_dictionary,
                        entry_to_school_dict=name_to_school_dict,
                    )
                else:
                    tournament_results += get_speech_results_from_rounds_only(
                        event=event,
                        code_dictionary=entry_id_to_entry_code_dictionary,
                        entry_dictionary=entry_id_to_entry_entry_name_dictionary,
                        entry_to_school_dict=name_to_school_dict,
                    )
                    # TODO - add an option to enrich the results via scraped data - perhaps replacing rounds-only?

    # Check if a result name has a 'full name' in the full name dictionary (scraped from Tabroom.com)
    # If it exists, replace the short name with the full name
    # Full name can only be ascertained from web scraping
    if scrape_entry_records_bool:
        for result in tournament_results:
            if result["entry_name"] in name_to_full_name_dict:
                result["entry_name"] = name_to_full_name_dict[result["entry_name"]]

    # Select the schools to write up reports on
    if all_schools:
        schools_to_write_up = school_set
        grouped_data = group_data_by_school(
            results=tournament_results, all_schools=all_schools
        )
    else:
        schools_to_write_up = set([school_name])
        grouped_data = group_data_by_school(
            results=tournament_results, school_name=school_name
        )

    # FOR EACH SCHOOL, GENERATE A SUMMARY AND SAVE IT TO DISK
    os.makedirs(f"{data['name']}_summaries", exist_ok=True)
    data_labels.remove("entry_code")  # At this point, code is useless
    for school in schools_to_write_up:
        logging.info(f"Starting results generation for {school}...")
        chat_gpt_payload = generate_chat_gpt_prompt(
            tournament_data=data,
            school_name=school,
            custom_url=custom_url,
            school_count=len(school_set),
            state_count=len(state_set),
            has_speech=has_speech,
            has_debate=has_debate,
            entry_dictionary=name_to_school_dict,
            header_string="|".join(data_labels),
        )
        school_filtered_tournament_results = grouped_data[school]
        if not school_filtered_tournament_results:
            logging.warning(f"No results found for {school}")
            continue
        sorted_school_results = sorted(
            school_filtered_tournament_results,
            key=lambda x: float(x["percentile"]),
            reverse=True,
        )
        # If there is at least one result above the percentile minimum, filter out any results below the percentile minimum
        if int(float(sorted_school_results[0]["percentile"])) > percentile_minimum:
            logging.info(
                "Found a result above the percentile minimum, filtering out results below threshold"
            )
            threshold_school_results = filter(
                lambda x: float(x["percentile"]) > percentile_minimum,
                sorted_school_results,
            )
            sorted_filtered_school_results = list(threshold_school_results)
        else:
            sorted_filtered_school_results = sorted_school_results

        # Filter down to just the top 15 results (based on percentile) to get better results for large schools
        if len(sorted_filtered_school_results) > max_results_to_pass_to_gpt:
            top_sorted_filtered_school_results = sorted_filtered_school_results[
                0 : max_results_to_pass_to_gpt - 1
            ]
        else:
            top_sorted_filtered_school_results = sorted_filtered_school_results
        logging.info(
            f"School specific results without any filtering:\r\n{json.dumps(sorted_school_results, indent=4)}"
        )
        chat_gpt_payload += create_data_strings(
            data_objects=top_sorted_filtered_school_results,
            data_labels=data_labels,
        )
        final_gpt_payload = "\r\n".join(chat_gpt_payload)
        openai.api_key_path = "openAiAuthKey.txt"
        logging.info(f"Generating summary for {school}")
        logging.info(f"GPT Prompt: {final_gpt_payload}")
        if read_only:
            logging.info(
                f"Skipping summary generation for {school} due to read-only mode"
            )
            continue
        else:
            body_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": final_gpt_payload},
                ],
            )["choices"][0]["message"]["content"]
            editor_payload = (
                "You are the editor of a local newspaper. Keep the tone factual and concise. Edit the following article improve its flow and grammar:\r\n"
                + body_response
            )
            editor_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": editor_payload},
                ],
            )["choices"][0]["message"]["content"]
            headline_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a headline for this article. The response should be just a single headline, not in quotes",
                    },
                    {"role": "user", "content": editor_response},
                ],
            )["choices"][0]["message"]["content"]

        sorted_by_event = sorted(
            school_filtered_tournament_results,
            key=lambda x: x["event_name"],
            reverse=False,
        )
        sorted_by_event_without_round_by_round = []
        # Reduce to just the essentials
        for result_for_numbered_list in sorted_by_event:
            # Remove round-by-round results from the numbered list -- not required
            result_for_numbered_list.pop("results_by_round")
            sorted_by_event_without_round_by_round.append(result_for_numbered_list)
        logging.info(f"Generating list of results for {school}")
        list_generation_prompt = generate_list_generation_prompt(headers=data_labels)
        numbered_list_prompt = (
            list_generation_prompt
            + "\r\n"
            + "\r\n".join(
                create_data_strings(
                    data_objects=sorted_by_event_without_round_by_round,
                    data_labels=data_labels,
                )
            )
        )

        logging.info(f"GPT Prompt: {numbered_list_prompt}")
        if read_only:
            logging.info(f"Skipping list generation for {school} due to read-only mode")
            continue
        else:
            numbered_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": numbered_list_prompt},
                ],
            )["choices"][0]["message"]["content"]

            with open(f"{data['name']}_summaries/{school}_summary.txt", "w") as f:
                f.write(
                    headline_response
                    + "\r\n"
                    + editor_response
                    + "\r\n"
                    + "Event-by-Event Results"
                    + "\r\n"
                    + numbered_response
                )
