# Processing of sentences using OpenAI

import logging
import re
import uuid

from dna.process_entities import check_if_noun_is_known, get_sentence_entities
from dna.query_openai import access_api, categories, noun_categories, quote_prompt1, quote_prompt2, quote_prompt3, \
    rhetorical_devices, sent_prompt1, sent_prompt2, sent_prompt3, speaker_prompt
from dna.utilities_and_language_specific import empty_string, plural_pronouns

modal_mapping = {'can-present': ':ReadinessAndAbility',              # Ex: I can swim.
                 'can-not-present': ':OpportunityAndPossibility',    # Ex: I can visit tomorrow.
                 'could-past': ':ReadinessAndAbility',               # Ex: I could skate when I was younger.
                 'could-not-past': ':OpportunityAndPossibility',     # Ex: It could rain tomorrow.
                 'have to': ':AdviceAndRecommendation',              # Ex: You have to leave.
                 'may': ':OpportunityAndPossibility',                # Ex: It may rain.
                 'might': ':OpportunityAndPossibility',              # Ex: It might rain.
                 'must': ':RequirementAndDependence',                # Ex: I must leave.
                 'ought to': ':AdviceAndRecommendation',             # Ex: You ought to leave.
                 'shall-second': ':RequirementAndDependence',        # Ex: You shall go to the store.
                 'shall-not-second': ':IntentionAndGoal',            # Ex: They shall go to the store.
                 'should': ':AdviceAndRecommendation',               # Ex: They should go to the store.
                 'will': ':OpportunityAndPossibility',               # Ex: It will rain tomorrow.
                 'would': ':OpportunityAndPossibility'}              # Ex: I would take the train if possible.

previous_sentences_text = 'Here is the text of up to 2 previous sentences from the narrative or article. The ' \
                          'sentences are provided for co-reference resolution. They end with the string "**" which ' \
                          'should be removed. Note that there may be no previous sentences.'

semantic_roles = ("agent", "dative", "experiencer", "force", "instrument",
                  "location", "measure", "patient", "theme")

semantic_role_mapping = {"agent": ":has_active_entity",
                         "dative": ":has_affected_entity",
                         "experiencer": ":has_affected_entity",
                         "force": ":has_active_entity",
                         "instrument": ":has_instrument",
                         "location": ":has_location",
                         "measure": ":has_quantification",
                         "patient": ":has_affected_entity",
                         "theme": ":has_topic"}


def _determine_predicate(noun_text: str, noun_iri: str, noun_category: str, event_category: str,
                         semantic_role: str, event_semantics: str) -> (str, str):
    """
    Logic needed to choose the appropriate predicate for the noun relationships to the sentence's
    verb/semantic.

    :param noun_text: String holding the noun text
    :param noun_iri: An IRI for the noun, if the noun is a proper noun/already known
    :param noun_category: String holding the DNA class of the noun
    :param event_category: String holding the DNA EventAndState class (or subclass) if the noun
                           can be classified as an event or state
    :param semantic_role: The specific semantic role label relating the noun to the verb/semantic

    :return: A tuple of the predicate and object of the triple defining the noun relationship
             to the sentence's semantic.
    """
    triple_object = noun_iri if noun_iri else f'[:text "{noun_text}" ; ' \
                                              f'a {event_category if event_category else noun_category}]'
    # TODO: Further refine mapping rules
    if event_semantics == ':EnvironmentAndCondition':
        if (semantic_role == 'agent' or
                noun_category in (':Person', ':Person, :Collection', ':GovernmentalEntity', ':OrganizationalEntity',
                                  ':EthnicGroup', ':PoliticalGroup', ':ReligiousGroup', ':Animal', ':Plant')):
            return ':has_described_entity', triple_object
        if noun_category == ':Occupation':
            return ':has_aspect', triple_object
    elif event_semantics == ':Affiliation':
        if semantic_role == 'agent':
            return ':has_active_entity', triple_object
        else:
            return ':has_affected_entity', triple_object
    elif semantic_role in semantic_roles:
        if event_category:
            return ':has_topic', triple_object
        else:
            return semantic_role_mapping[semantic_role], triple_object
    return empty_string, empty_string


def get_sentence_details(sent_iri: str, sent_text: str, ttl_list: list, attribution: str, for_quote: bool,
                         entities: list, nouns_dict: dict):
    """
    Retrieve sentence (or quotation) details using the OpenAI API and create the sentence
    (or quotation) level Turtle.

    :param sent_iri: The IRI identifying the current sentence
    :param sent_text: String holding the text of up to the 3 last sentences
    :param ttl_list: The current Turtle definition where the new declarations will be stored
    :param attribution: String identifying the person who stated/communicate a quotation (is empty
                        for normal sentences)
    :param for_quote: Boolean indicating that the processing is for a quotation (if True)
    :param entities: An array of named entities found in the sentence, if any
    :param nouns_dict: A dictionary holding the nouns/named entities encountered in the narrative;
             The dictionary keys are the text for the noun, and its values are a tuple consisting
             of the spaCy entity type and the noun's IRI. An IRI value may be associated with
             more than 1 text. Note that this parameter is only used when processing quotations
             (in order to get the IRI of the speaker).
    :return: N/A (ttl_list is updated with the details from OpenAI)
    """
    # Handle named entities
    if entities:
        entity_iris, entity_ttl = get_sentence_entities(sent_text, entities, nouns_dict)
        ttl_list.extend(entity_ttl)
        for entity_iri in entity_iris:
            ttl_list.append(f'{sent_iri} :mentions {entity_iri} .')
    # Capture a quotation's attribution and whether it is formatted as a question or exclamation
    if for_quote:
        if attribution:
            attrib_type, attrib_iri = check_if_noun_is_known(attribution, 'PERSON', nouns_dict)
            if attrib_iri:
                ttl_list.append(f'{sent_iri} :attributed_to {attrib_iri} .')
        # Also check for punctuation, since this is stored in the quoted text
        if '?' in sent_text:
            ttl_list.append(f'{sent_iri} a :Inquiry .')
        elif '!' in sent_text:
            ttl_list.append(f'{sent_iri} a :ExpressiveAndExclamation .')
    else:
        # Not a quotation but one may be referenced in sentence
        quote_texts = re.findall(r'Quotation[0-9]+', sent_text)
        for quote_text in quote_texts:
            ttl_list.append(f'{sent_iri} :has_component :{quote_text} .')
        if quote_texts:
            # Quotation indicates a speech act and the quote text is analyzed separately
            # TODO: Should the containing sentence be analyzed further?
            return ttl_list
    # Processing the first prompt for details such as sentence person, sentiment, tense, ...
    if for_quote:
        quote_dict = access_api(quote_prompt1.replace("{quote_text}", sent_text))   # Note different prompts
        if quote_dict:   # Might not get reply from OpenAI
            sentiment = quote_dict['sentiment']
            summary = quote_dict['summary']
            ttl_list.extend([f'{sent_iri} :sentiment "{sentiment}" ; :summary "{summary}" .',
                             f'{sent_iri} :grade_level {quote_dict["grade_level"]} .'])
    else:
        sent_dict = access_api(sent_prompt1.replace("{sent_text}", sent_text))
        if sent_dict:   # Might not get reply from OpenAI
            sentiment = sent_dict['sentiment']
            tense = sent_dict['tense']
            voice = sent_dict['voice']
            summary = sent_dict['summary']
            ttl_list.extend([f'{sent_iri} :sentence_person {sent_dict["person"]} ; :sentiment "{sentiment}".',
                             f'{sent_iri} :voice "{voice}" ; :tense "{tense}" ; :summary "{summary}" .',
                             f'{sent_iri} :grade_level {sent_dict["grade_level"]} .'])
            if sent_dict['errors'] == 'true':
                ttl_list.append(f'{sent_iri} :errors true .')
            if 'modal' in sent_dict and sent_dict['modal']['text'] != "none":
                modal_text = sent_dict['modal']['text']
                if modal_text not in modal_mapping:
                    if modal_text == 'can':
                        modal_text = 'can-present' if tense == 'present' else 'can-not-present'
                    elif modal_text == 'could':
                        modal_text = 'could-past' if tense == 'past' else 'could-not-past'
                    elif modal_text == 'shall':
                        modal_text = 'shall-second' if voice == 2 else 'shall-not-second'
                    else:
                        modal_text = empty_string
                if modal_text:
                    modal_negated = sent_dict['modal']['negated']
                    predicate = ':has_semantic {:negated true}' if modal_negated == 'true' else ':has_semantic'
                    ttl_list.append(f'{sent_iri} {predicate} {modal_mapping[modal_text]} .')
    # Processing the second prompt for rhetorical devices
    if for_quote:
        sent_dict = access_api(quote_prompt2.replace("{sent_text}", sent_text))   # Note different prompts
    else:
        sent_dict = access_api(sent_prompt2.replace("{sent_text}", sent_text))
    if sent_dict:   # Might not get reply from OpenAI or there are no rhetorical devices
        if 'rhetorical_devices' in sent_dict and len(sent_dict['rhetorical_devices']) > 0:
            for device_detail in sent_dict['rhetorical_devices']:
                device_numb = device_detail['device_number']
                predicate = ':rhetorical_device {:evidence "' + device_detail['evidence'] + '"} '
                ttl_list.append(f'{sent_iri} {predicate} "{rhetorical_devices[device_numb - 1]}" .')
    # Processing the third prompt for the semantics of events and states, and related nouns
    if for_quote:
        sent_dict = access_api(quote_prompt3.replace("{sent_text}", sent_text))     # Note different prompts
    else:
        sent_dict = access_api(sent_prompt3.replace("{sent_text}", sent_text))
    if sent_dict:       # Needed since might not get reply from OpenAI
        for semantic in sent_dict['semantics']:
            if semantic['negated'] == 'true':
                predicate = ':has_semantic {:negated true}'
            else:
                predicate = ':has_semantic'
            event_iri = f':Event_{str(uuid.uuid4())[:13]}'
            ttl_list.append(f'{sent_iri} {predicate} {event_iri} .')
            trigger_text = semantic['trigger_text']
            event_semantics = categories[semantic["category_number"] - 1]
            ttl_list.append(f'{event_iri} a {event_semantics} ; :text "{trigger_text}" .')
            if 'nouns' in semantic:     # Not returned for quotations
                for noun in semantic['nouns']:
                    noun_text = noun['text']
                    semantic_role = noun['semantic_role']
                    noun_category = noun_categories[noun['noun_type'] - 1]
                    if (noun['singular'] == 'false' or noun_text in plural_pronouns) and \
                            ':Collection' not in noun_category:
                        noun_category += ', :Collection'
                    event_category = categories[noun['semantic_category'] - 1] \
                        if noun['semantic_category'] - 1 > -1 else empty_string
                    entity_type, noun_iri = check_if_noun_is_known(noun_text, empty_string, nouns_dict)
                    predicate, triple_object = _determine_predicate(noun_text, noun_iri, noun_category, event_category,
                                                                    semantic_role, event_semantics)
                    if predicate:     # TODO: Handle no mappings vs just ignoring
                        ttl_list.append(f'{event_iri} {predicate} {triple_object} .')
    return
