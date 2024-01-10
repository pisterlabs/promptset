import os
import struct
import pickle
import numpy as np
import openai

from collections import OrderedDict
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

from quickstart.connection import get_current_user
from quickstart.sqlite_utils import get_insert_query

def build_knn(db, Setting, PriorityList, PriorityItem, PriorityMessage, p_item):
    # ideally we should have 10 nearest neighbors classifiers object fitted on all previous data
    # but for now we can train it right there
    result = PriorityList.query.filter_by(id=p_item.priority_list_id).first()
    platform_id = result.platform_id
    # get all priority_list items except a fresh one
    p_lists = PriorityList.query.filter_by(platform_id=platform_id) \
        .filter(PriorityList.id != p_item.priority_list_id) \
        .all()
    ids = []
    all_vectors = []
    nbrs = None
    for p_list in p_lists:
        # p_methods = PriorityListMethod.query.filter_by(priority_list_id=p_list.id).all()
        p_items = PriorityItem.query.filter_by(priority_list_id=p_list.id).all()
        for p_item in p_items:
            _ = PriorityMessage.query.filter_by(id=p_item.priority_message_id).first()
            ids.append(_.id)
            # emb_vector = struct.unpack('<q', b'\x15\x00\x00\x00\x00\x00\x00\x00')
            emb_vector = np.frombuffer(_.embedding_vector, dtype='<f4')
            emb_vector = np.array(emb_vector, dtype=np.float64)
            all_vectors.append(emb_vector)


    setting = db.session.query(Setting).filter_by(user_id=get_current_user().id).first()

    if all_vectors:
        X = np.array(all_vectors)
        # we want to build NN algorithm for any number of samples present
        # but initially there would not be many
        # let's set this up to 3 for now
        nbrs = NearestNeighbors(n_neighbors=setting.num_neighbors,
                         metric='cosine',
                         algorithm='brute',
                         n_jobs=-1).fit(X)
    return nbrs, ids

def create_priority_list(db, PriorityList, PriorityListMethod, platform_id, session_id):

    # check if no priority_list_methods for this platform
    p_list_methods = PriorityListMethod.query.filter_by(platform_id=platform_id).all()
    if not p_list_methods:
        create_priority_list_methods(db, PriorityListMethod, platform_id)

    timestamp = int(round(datetime.now().timestamp()))
    plist_kwargs = OrderedDict([('session_id', session_id)
        , ('platform_id', platform_id)
        , ('created', timestamp)
        ])
    plist_query = get_insert_query('priority_list', plist_kwargs.keys())
    db.session.execute(plist_query, plist_kwargs)
    db.session.commit()

    p_list = PriorityList.query.filter_by(session_id=session_id) \
        .filter_by(platform_id=platform_id) \
        .first()
    p_list.update_p_a()
    return p_list.id


def create_priority_list_methods(db, PriorityListMethod, platform_id):
    script_path = 'quickstart.priority_method'
    methods = [ (script_path, 'ask_gpt')
        #  (script_path, 'ask_large_bloom')
        # , (script_path, 'toy_keyword_match')
        # , (script_path, 'sentiment_analysis')
        ]

    for python_path, name in methods:
        plist_method_kwargs = OrderedDict([('platform_id', platform_id)
            , ('name', name)
            , ('python_path', python_path)])
        plist_method_query = get_insert_query('priority_list_method', plist_method_kwargs.keys())
        db.session.execute(plist_method_query, plist_method_kwargs)
        db.session.commit()


def update_priority_list_methods(db, PriorityListMethod, platform_id, plist_id):
    pl_methods = PriorityListMethod.query.filter_by(platform_id=platform_id).all()

    for pl_method in pl_methods:
        pl_method.update_p_m_a(plist_id)
    pl_methods = PriorityListMethod.query.filter_by(platform_id=platform_id).all()
    new_values = []
    for pl in pl_methods:
        new_values.append(pl.p_m_a)
    _ = sum(new_values)
    if _ < 1:
        for pl in pl_methods:
            pl.p_m_a *= 1.0 / _
    else:
        for pl in pl_methods:
            pl.p_m_a /= _
    db.session.commit()


# todo: replace with named tuple
def fill_priority_list(db, messages, get_abstract_func, plist_id, \
        PriorityMessage, PriorityList, PriorityItem, PriorityItemMethod, PriorityListMethod \
        , TableName, columns_list, Setting):
    # iterate over records of variable platform
    message_ids = []
    item_ids = []
    result = PriorityList.query.filter_by(id=plist_id).first()
    platform_id = result.platform_id
    method_ids = [ x.id for x in PriorityListMethod.query.filter_by(platform_id=platform_id).all() ]
    method_item_ids = []
    session_id = messages[0].session_id if messages else None
    for message in messages:
        inp_text, m_id = get_abstract_func(message)
        p_message_kwargs = OrderedDict([('message_id', m_id)
            , ('input_text_value', inp_text)
            , ('platform_id', platform_id)
            , ('session_id', message.session_id)])
        p_message_query = get_insert_query('priority_message', p_message_kwargs.keys())
        db.session.execute(p_message_query, p_message_kwargs)
    db.session.commit()

    priority_messages = db.session.query(PriorityMessage).filter_by(session_id=session_id) \
        .filter_by(platform_id=platform_id).all()
    message_ids = [x.id for x in priority_messages]
    sentences = [x.input_text_value for x in priority_messages]

    # local or openai
    # this could be changed to handle get_current_user().setting.embeddings
    # but this would require compatibility of embeddings
    # or re-vectorisation of whole summary history
    # with the support of multiple embeddings stored at the same time
    # for now better to stick to openai ada
    embedding_mode = 'openai'

    if embedding_mode == 'local':
        model_filepath = os.path.join('file_store', '2023-02-22-embedding-model')
        model_pickle = open(model_filepath, 'rb')
        embedding_model = pickle.load(model_pickle)
        embedding_vectors = embedding_model.encode(sentences)
    elif embedding_mode == 'openai':
        openai.api_key = os.getenv("OPEN_AI_KEY")
        embedding_vectors = []
        for sentence in sentences:
            model = 'text-embedding-ada-002'
            text = sentence.replace('\n', ' ')
            vector = openai.Embedding.create(input=text, model=model)['data'][0]['embedding']
            embedding_vectors.append(np.array(vector).tobytes())
        embedding_vectors = np.array(embedding_vectors)
    assert len(embedding_vectors) == len(priority_messages)
    # todo: assert items correspond appropriately, not only by length of arrays but elementwise assertion
    for i in range(len(priority_messages)):
        priority_messages[i].embedding_vector = embedding_vectors[i]
    db.session.commit()


    for message_id in message_ids:
        p_item_kwargs = OrderedDict([('priority_list_id', plist_id)
            , ('priority_message_id', message_id)])
        p_item_query = get_insert_query('priority_item', p_item_kwargs.keys(), returning_id=False)
        db.session.execute(p_item_query, p_item_kwargs)
    db.session.commit()

    priority_items = db.session.query(PriorityItem).filter_by(priority_list_id=plist_id).all()
    item_ids = [x.id for x in priority_items]

    for item_id in item_ids:
        for method_id in method_ids:
            pi_method_kwargs = OrderedDict([('priority_item_id', item_id)
                , ('priority_list_method_id', method_id)])
            pi_method_query = get_insert_query('priority_item_method', pi_method_kwargs.keys())
            db.session.execute(pi_method_query, pi_method_kwargs)

    db.session.commit()

    # todo there is also an obvious optimisation opportunity
    #  chat could also do priority estimation based on text in bulk
    #  however that might degrade accuracy or make ouput unpredictable
    priority_method_items = db.session.query(PriorityItemMethod).join(PriorityItem) \
        .filter(PriorityItem.id.in_(tuple(item_ids))).all()

    for priority_method_item in priority_method_items:
        priority_method_item.calculate_p_b_m_a()


    db.session.commit()
    nbrs_out = {}
    # todo optimise calculate_p_b cause it build the same KNearestNeighbors model each item
    # priority_items = db.session.query(PriorityItem).filter(PriorityItem.id.in_(tuple(item_ids))).all()
    for priority_item in priority_items:
        # todo - this isn't optimized per settings restrictions
        # calculate everything at this stage, and use setting only in representation
        # later design repr so that consequent change in setting would allow different reprs
        nbrs, ids = build_knn(db, Setting, PriorityList, PriorityItem, PriorityMessage, priority_item)

        priority_item.calculate_p_b(nbrs, ids)
        # nbr = priority_item.calculate_p_b(nbrs, ids)
        # nbrs_out[priority_item.priority_message_id] = nbr
        priority_item.calculate_p_b_a()
        priority_item.calculate_p_a_b()
        priority_item.calculate_p_a_c(TableName, columns_list)
        nbr = priority_item.calculate_p_b_c(TableName, columns_list)
        nbrs_out[priority_item.priority_message_id] = nbr
        priority_item.calculate_p_a_b_c()
    db.session.commit()


    if nbrs_out:
        msg_out = {}
        for p_id, nbr_list in nbrs_out.items():
            msg_out[p_id] = []
            for nbr in nbr_list:
                msg = PriorityMessage.query.filter_by(id=nbr).first()
                if msg:
                    msg_out[p_id].append(msg.input_text_value)
        return msg_out
