from mygptapp.models import User, Message
from mygptapp import db, rules

class WebSearch:
    def process_web_search_results(self, original_prompt, query, search_results):
        print("search_results: ", search_results)

        # check for errors
        if "error" in search_results:
            print("error in search_results: ", search_results["error"])
            return search_results

        # bake search_results into a "chat"
        messages = []
        messages.append({"role": "user", "content": rules.get_preamble_text()})
        messages.append({"role": "user", "content": original_prompt})
        search_results_as_message = self.search_results_to_message(query, search_results)
        messages.append({"role": "assistant", "content": search_results_as_message})
        messages.append({
            "role": "system",
            "content": "based on these web results, and remembering that search result snippets are usually stale, please return a relevant scrape_url action to respond to the base prompt"})

        print("calling the model with a followup prompt: ", messages)

        from mygptapp.openai_api import OpenAIAPI
        api = OpenAIAPI()
        followup_response = api.call_model(messages)
        followup_response["search_results_as_message"] = search_results_as_message
        print("followup_response: ", followup_response)

        # save followup response to db in conversation id=2
        bot = User.query.filter_by(username="bot").first()
        #jakedowns = User.query.filter_by(username="jakedowns").first()
        message = Message(
            role="assistant",
            content=followup_response['choices'][0]['message']['content'],
            user_id=bot.id,
            is_inner_thought=False,
            convo_id=1)
        db.session.add(message)
        db.session.commit()
        return followup_response

    def search_results_to_message(self, query, search_results):
        search_results_as_message = "Relevant Search Results for Query: " + query
        search_results_as_message += "\n"
        i = 1
        for result in search_results:
            search_results_as_message += "\n\n"
            search_results_as_message += f"> result {str(i)}"
            search_results_as_message += f"\n> result title: {result['name']}"
            search_results_as_message += f"\n> result url: [{result['url']}]({result['url']})"
            search_results_as_message += "\n> result snippet: " + "\n".join(["> "+line for line in result['snippet'].split('\n')])
            search_results_as_message += "\n"
            search_results_as_message += "\n\n"
            i += 1
        return search_results_as_message