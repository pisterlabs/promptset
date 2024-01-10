import json
#import openai
import random
import time

data_path = 'dev.jsonl'
with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f.readlines()]

#with open('../../keys/openai_key', 'r') as f:
#    openai.api_key = f.readline().strip()
#
#with open('../../keys/openai_org_id', 'r') as f:
#    openai.organization = f.readline().strip()

with open('annotated_templates_new.json', 'r') as f:
    templates = json.load(f)

#data = data[:15]

max_hop = 2

def event_relation(doc, event1, event2):
    ### just for offline test ###
    return 'before'
    ### --------------------- ###

    message = 'Document:\n{document}'.format(document=doc)

    e1 = 'Event1:\nTrigger: {t}'.format(t=event1['trigger']['text'])
    for arg in event1['arguments']:
        roles = '{role}: {text}'.format(role=arg['role'], text=arg['text'])
        e1 = '\n'.join([e1, roles])
    message = '\n\n'.join([message, e1])

    e2 = 'Event2:\nTrigger: {t}'.format(t=event1['trigger']['text'])
    for arg in event2['arguments']:
        roles = '{role}: {text}'.format(role=arg['role'], text=arg['text'])
        e2 = '\n'.join([e2, roles])
    message = '\n'.join([message, e2])

    relations = '\n'.join(['Temporal Relations:', 'before', 'after', 'overlap',
                           'equal', 'contain', 'none'])
    message = '\n\n'.join([message, relations])

    question = 'What is the temporal relation between Event1 and Event2?'
    message = '\n\n'.join([message, question])

    pred = call(message)
    relations = ['before', 'after', 'overlap', 'equal', 'contains']
    for r in relations:
        if r in pred:
#            print(json.dumps([message, pred], indent=2), file=f_relations)
#            print(event1['id'], event2['id'], r)
            return r

    return -1

def relation_reverse(relation):
    # 'before', 'after', 'overlap', 'equal', 'contain', 'none'
    if 'before' == relation:
        return 'after'
    if 'after' == relation:
        return 'before'
    if 'contain' == relation:
        return 'none'
    return relation

##### strategy 1 #####
def relation_text_explicit(relation):
    if 'before' == relation:
        return 'which is before'
    elif 'after' == relation:
        return 'which is after'
    elif 'equal' == relation:
        return 'which equals to'
    elif 'contain' == relation:
        return 'which contains'
    elif 'overlap' == relation:
        return 'which overlaps'

def template_event_explicit(doc, event1, event2, relation):
    args1 = event1['arguments']
    args2 = event2['arguments']
    qa_pairs = []
    for a1 in args1[:3]:
        for a2 in args2[:3]:
            text = 'What is the {role1} in the event {relation} another event that {argument2} is {role2}?'.format(
                        role1=a1['role'],
                        relation=relation_text_explicit(relation),
                        role2=a2['role'],
                        argument2=a2['text'],
                    )
            qa_pairs.append({'init_question': text,
                             'expect_answer': a1['text'],
                             'events': (event1, event2),
                             'relation': relation
                             })

    return qa_pairs

def find_template(event):
    event_type = event['event_type']
    ori_type = event_type
    while '' != ori_type:
        try:
            template = templates[event_type]
            break
        except:
            ori_type = ori_type.split('.')[:-1]
            event_type = ori_type
            while 3 > len(event_type):
                event_type.append('Unspecified')
            event_type = '.'.join(event_type)
    return template

def relation_text_implicit(relation):
    if 'before' == relation:
        return 'before'
    elif 'after' == relation:
        return 'after'
    elif 'equal' == relation:
        return 'at the same time'
    elif 'contain' == relation:
        return 'during'
    elif 'overlap' == relation:
        return 'when'
    elif 'none' == relation:
        raise Exception

def template_event_implicit(doc, event1, event2, relation):
    # event 1
    template_1 = find_template(event1)
    verb_1 = template_verbs_extraction(template_1)
    args1 = event1['arguments']

    # event 2
    template_2 = find_template(event2)
    verb_2 = template_verbs_extraction(template_2)
    args2 = event2['arguments']

    qa_pairs = []
    for a1 in args1[:3]:
        for a2 in args2[:3]:
            partial_question = []
            trigger_flag = False
            for idx, trigger in enumerate(template_2['scopes']):
                if 0 < idx and not trigger_flag:
                    trigger_flag = True
                    partial_question.append(verb_2)
                if trigger['role'] == a2['role']:
                    text = trigger['text'].replace('[{role}]'.format(
                            role=trigger['role']),
                            a2['text'])
                    partial_question.append(text)
                if 2 <= len(partial_question):
                    break
            partial_question = ' '.join(partial_question)

            text = 'What {verb1} {relation} {partial_q}'.format(
                    verb1=verb_1,
                    relation=relation_text_implicit(relation),
                    partial_q = partial_question
                    )
            # data format
            qa_pairs.append({'init_question':text,
                             'answer': a1['text'],
                             'events': (event1, event2),
                             'relation': relation})

    return qa_pairs

def template_2(doc, eid, event):
    all_relations = ['before', 'after', 'overlap', 'equal', 'contain']
    args = event['arguments']
    cnt1, cnt2 = 0, 0
    for arg in args:
        cnt2 += 1
        for relation in all_relations:
            text = 'Only output a number: How many events are {relation} the event that {argument} is {role}?'.format(
                        relation=relation,
                        argument=arg['text'],
                        role=arg['role'],
                    )
            message = '\n\n'.join(['Document:\n{doc}'.format(doc=doc), text])
            print(message)
            pred = call(message)

            answer = count_relation(eid, relation)
            if str(answer) in pred:
                cnt1 += 1
            print(json.dumps([text, pred], indent=2), file=f_outputs2)
            print(event['id'], pred)

    return cnt1, cnt2

def template_3(doc, event):
    relations = ['before', 'after', 'overlap', 'equal', 'contains']
    args = event['arguments']
    cnt1, cnt2 = 0, 0
    for arg in args:
        cnt2 += 1
        for relation in relations:
            text = 'Question: List all events are {relation} the event that {argument} is {role}'.format(
                        relation=relation,
                        argument=arg['text'],
                        role=arg['role'],
                    )
            message = '\n\n'.join(['Document:\n{doc}'.format(doc=doc), text])
            pred = call(message)
            print(json.dumps([text, pred], indent=2), file=f_outputs3)
            print(event['id'], pred)

    return cnt1, cnt2

##### strategy 2 #####
event_histories = set([])
def find_next(text_history, event_history, entity_event_dict, hop=-1):
    if hop >= max_hop:
        event_histories.add(tuple([e['id'] for e in event_history] + text_history[:-1]))
        return

    event_list = entity_event_dict[text_history[-1]]
    for event in event_list:
        flag = False
        for h in event_history:
            if event['id'] == h['id']:
                flag = True
                break
        if flag:
            continue

        event_history.append(event)
        for arg in event['arguments']:
            flag = False
            for h in text_history:
                if arg['text'] == h:
                    flag = True
            if flag and (hop + 1 != max_hop):
                continue

            text_history.append(arg['text'])
            find_next(text_history, event_history, entity_event_dict, hop + 1)
            text_history.pop()
        event_history.pop()

def template_verbs_extraction(template):
    verbs = []
    tokens = template['template']
    scopes = template['scopes']
    verbs = tokens[scopes[0]['end'] + 1:scopes[1]['start']]
    return ' '.join(verbs)

def build_init_question(role_options, template):
    init_question = []
    trigger_flag = False
    for idx, trigger in enumerate(template['scopes']):
        if 0 < idx and not trigger_flag:
            trigger_flag = True
            init_question.append(template_verbs_extraction(template))
        if trigger['role'] in role_options:
            init_question.append(trigger['text'])
            role_options.remove(trigger['role'])
    return ' '.join(init_question) + '.'

def template_entity_2hop_implicit(doc, events, text_history):
    event1, event2, event3 = events

    init_question = []

    # event 1
    template = find_template(event1)
    for arg in event1['arguments']:
        if arg['text'] == text_history[0]:
            bridging_arg = arg
            break
    # pick an answer in event 1
    options = []
    for arg in event1['arguments']:
        if arg['text'] != text_history[0] and arg['role'] != bridging_arg['role']:
            options.append(arg)
    answer = random.choice(options)
    role_options = [answer['role'], bridging_arg['role']]
    init_question_1 = build_init_question(role_options, template)
    init_question.append(init_question_1)

    # event 2
    template = find_template(event2)
    bridging_args = []
    for arg in event2['arguments']:
        if arg['text'] in text_history:
            bridging_args.append(arg)
    role_options = [arg['role'] for arg in bridging_args]
    init_question_2 = build_init_question(role_options, template)
    init_question.append(init_question_2)

    # event 3
    template = find_template(event2)
    for arg in event3['arguments']:
        if arg['text'] == text_history[1]:
            bridging_arg = arg
            break
    role_options = [bridging_arg['role']]
    init_question_3 = build_init_question(role_options, template)
    init_question.append(init_question_3)

    init_question = ' '.join(init_question)

    # data format
    qa_pair = {'init_question': init_question,
               'expect_answer': answer['text'],
               'events': (
                   event1,
                   event2,
                   event3,
                ),
               'bridge_entities': (
                   text_history[0],
                   text_history[1]
                )}

    return qa_pair

'''
    for arg in event1['arguments']:
        if arg['text'] != text_history[0]:
            role1 = arg['role']
            answer = arg['text']
            break

    for arg in event1['arguments']:
        template = templates[event_type]
        event_type = event['event_type']
        if arg['text'] == text_history[0]:
            pass

    for arg in event2['arguments']:
        if arg['text'] == text_history[0]:
            role2 = arg['role']
            break

    for arg in event3['arguments']:
        if arg['text'] == text_history[1]:
            role3 = arg['role']
            break

    for arg in event3['arguments']:
        if arg['text'] != text_history[1]:
            role4 = arg['role']
            break

    question = ('What is the {role1} in the event that {trigger1} '
                'the {role2} that {trigger2} '
                'the {role3} that {trigger3} the {role4}?'
                ).format(
                    role1=role1, trigger1=event1['trigger']['text'],
                    role2=role2, trigger2=event2['trigger']['text'],
                    role3=role3, trigger3=event3['trigger']['text'],
                    role4=role4
                    )

    return init_question, answer
'''

##### strategy 3 #####
def count_relation(event, relation, relation_pairs):
    cnt = 0
    events = []
    for (e1, r, e2) in relation_pairs:
        if e1['id'] == event['id'] and r == relation:
            cnt += 1
            events.append(e2)
    return cnt, events

def template_event_list_count(doc, event, relation_pairs):
    relations = ['before', 'after', 'overlap', 'equal', 'contain']
    args = event['arguments']
    qa_pairs = []
    for arg in args:
        for relation in relations:
            text = 'How many events are {relation} the event that {argument} is {role}'.format(
                        relation=relation,
                        argument=arg['text'],
                        role=arg['role'],
                    )
            answer, events_list = count_relation(event, relation, relation_pairs)
            qa_pairs.append({'init_question': text,
                             'expect_answer': answer,
                             'event': event,
                             'relation': relation,
                             'events': events_list})

    return qa_pairs

questions = {}
for d in data:
    doc = d['text']
    events = d['event_mentions']

    # strategy 1 event relations
    # extract relations
    pairs = []
    for i in range(len(events)):
        for j in range(i, len(events)):
            relation = event_relation(doc, events[i], events[j])
            pairs.append([events[i], relation, events[j]])

    double_pairs = []
    for (e1, r, e2) in pairs:
        double_pairs.append([e1, r, e2])
        reverse_r = relation_reverse(r)
        double_pairs.append([e2, reverse_r, e1])
    relation_pairs = double_pairs

    questions_s1 = []
    for (e1, r, e2) in pairs:
        qa_pairs = template_event_implicit(doc, e1, e2, r)
        questions_s1.extend(qa_pairs)
#    print(len(questions_s1))
#    print(questions_s1[:3])
#    input()

    # strategy 2 entity bridging
    # build entity bridging dict
    entity_event_dict = {}
    for event in events:
        arguments = event['arguments']
        if 2 > len(arguments):
            continue
        for arg in arguments:
            if arg['text'] not in entity_event_dict:
                entity_event_dict[arg['text']] = []
            entity_event_dict[arg['text']].append(event)

    # find lists
    event_histories = set([])
    for event in events:
        arguments = event['arguments']
        if 2 > len(arguments):
            continue
        else:
            roles = set([arg['role'] for arg in event['arguments']])
            if 2 > len(roles):
                continue
        for arg in arguments:
            event_history = [event]
            text_history = [arg['text']]
            find_next(text_history, event_history, entity_event_dict, 0)
#    print(event_histories)

    questions_s2 = []
    for histories in event_histories:
        p_list = []
        for i in range(max_hop+1):
            for event in events:
                if histories[i] == event['id']:
                    p_list.append(event)
                    break
        text_history = histories[max_hop+1:]
        qa_pair = template_entity_2hop_implicit(doc, p_list, text_history)
        questions_s2.append(qa_pair)
#        for event in p_list:
#            print(json.dumps(event))
#        print(text_history)
#    print(json.dumps(questions_s2[:1], indent=2))
#    input()

    # strategy 3 listing and counting
    questions_s3 = []
    for event in events:
        qa_pairs = template_event_list_count(doc, event, relation_pairs)
        questions_s3.extend(qa_pairs)

    # strategy 4 comparison
    event_types = set([event['event_type'] for event in events])
    questions_s4 = {}
    for event_type in event_types:
        event_list = []
        for event in events:
            if event_type == event['event_type']:
                event_list.append(event)
        if 1 < len(event_list):
            questions_s4[event_type] = event_list

    questions[d['doc_id']] = {'strategy_1': questions_s1,
                              'strategy_2': questions_s2,
                              'strategy_3': questions_s3,
                              'strategy_4': questions_s4,
                              }
print(questions.keys())
with open('sample_data.json', 'w') as f:
    json.dump(questions, f, indent=2)