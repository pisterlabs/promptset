'''
pip install flask flask-cors boto3
'''

from dotenv import load_dotenv
load_dotenv()

import openai  # for generating embeddings
import os
import json
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import boto3
import requests

# ---- embeddings ----

EMBEDDING_MODEL = "text-embedding-ada-002"
openai.api_key = os.environ['OPENAI_KEY']

def map_embedding(strs):
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=strs)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]
    return [e["embedding"] for e in response["data"]]

# ---- token to text ----

cryptos = []
with open('data/cryptos.json', 'r') as fin:
    cmc_cryptos = json.load(fin)
fields = cmc_cryptos['fields']
for row in cmc_cryptos['values']:
    obj = {}
    for idx, name in enumerate(fields):
        obj[name] = row[idx]
    keywords = [obj['name'], obj['symbol'], obj['slug']]
    if obj['address']:
        keywords.append(obj['address'][0])
    keywords = [i.lower() for i in keywords]
    obj['keywords'] = keywords
    cryptos.append(obj)

def desc_token(token):
    token = token.lower()
    found = None
    for crypto in cryptos:
        if token in crypto['keywords']:
            found = crypto
            break
    if not found:
        raise '404 Not Found'
    # return found
    return f'Liked item. ERC20 Token. Project name: {found["name"]}, Symbol: {found["symbol"]}.'

# ---- tag embeddings ----

with open('tmp/data/tag-posts-embeddings.json', 'r') as fin:
    obj = json.load(fin)
    tag_names = [tag for tag, embedding in obj]
    tag_embedding = [embedding for tag, embedding in obj]
tag_db = np.array(tag_embedding)

# ---- calculate ----
from botocore.client import Config
s3 = boto3.client(
    's3', 'us-east-1',
    endpoint_url="https://buckets.chainsafe.io",
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    config=Config(s3={
        'addressing_style': 'path',
    }))



LENS_API = 'https://api.lens.dev/'
LENS_HEADERS = {
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Connection': 'keep-alive',
    'DNT': '1',
    'Origin': 'https://api.lens.dev' 
}

def lens_search(query):
    tpl = '''
    query Search {
      search(request: {
        query: "{{query}}",
        type: PUBLICATION,
        limit: 10
      }) {
        ... on PublicationSearchResult {
          __typename 
          items {
            __typename 
            ... on Comment {
              ...CommentFields
            }
            ... on Post {
              ...PostFields
            }
          }
          pageInfo {
            prev
            totalCount
            next
          }
        }
        ... on ProfileSearchResult {
          __typename 
          items {
            ... on Profile {
              ...ProfileFields
            }
          }
          pageInfo {
            prev
            totalCount
            next
          }
        }
      }
    }

    fragment MediaFields on Media {
      url
      mimeType
    }

    fragment MirrorBaseFields on Mirror {
      id
      profile {
        ...ProfileFields
      }
      stats {
        ...PublicationStatsFields
      }
      metadata {
        ...MetadataOutputFields
      }
      createdAt
      collectModule {
        ...CollectModuleFields
      }
      referenceModule {
        ...ReferenceModuleFields
      }
      appId
    }

    fragment ProfileFields on Profile {
      profileId: id,
      name
      bio
      attributes {
        displayType
        traitType
        key
        value
      }
      isFollowedByMe
      isFollowing(who: null)
      metadataUrl: metadata
      isDefault
      handle
      picture {
        ... on NftImage {
          contractAddress
          tokenId
          uri
          verified
        }
        ... on MediaSet {
          original {
            ...MediaFields
          }
        }
      }
      coverPicture {
        ... on NftImage {
          contractAddress
          tokenId
          uri
          verified
        }
        ... on MediaSet {
          original {
            ...MediaFields
          }
        }
      }
      ownedBy
      dispatcher {
        address
      }
      stats {
        totalFollowers
        totalFollowing
        totalPosts
        totalComments
        totalMirrors
        totalPublications
        totalCollects
      }
      followModule {
        ...FollowModuleFields
      }
    }

    fragment PublicationStatsFields on PublicationStats { 
      totalAmountOfMirrors
      totalAmountOfCollects
      totalAmountOfComments
    }

    fragment MetadataOutputFields on MetadataOutput {
      name
      description
      content
      media {
        original {
          ...MediaFields
        }
      }
      attributes {
        displayType
        traitType
        value
      }
    }

    fragment Erc20Fields on Erc20 {
      name
      symbol
      decimals
      address
    }

    fragment PostFields on Post {
      id
      profile {
        ...ProfileFields
      }
      stats {
        ...PublicationStatsFields
      }
      metadata {
        ...MetadataOutputFields
      }
      createdAt
      collectModule {
        ...CollectModuleFields
      }
      referenceModule {
        ...ReferenceModuleFields
      }
      appId
      hidden
      reaction(request: null)
      mirrors(by: null)
      hasCollectedByMe
    }

    fragment CommentBaseFields on Comment {
      id
      profile {
        ...ProfileFields
      }
      stats {
        ...PublicationStatsFields
      }
      metadata {
        ...MetadataOutputFields
      }
      createdAt
      collectModule {
        ...CollectModuleFields
      }
      referenceModule {
        ...ReferenceModuleFields
      }
      appId
      hidden
      reaction(request: null)
      mirrors(by: null)
      hasCollectedByMe
    }

    fragment CommentFields on Comment {
      ...CommentBaseFields
      mainPost {
        ... on Post {
          ...PostFields
        }
        ... on Mirror {
          ...MirrorBaseFields
          mirrorOf {
            ... on Post {
              ...PostFields          
            }
            ... on Comment {
              ...CommentMirrorOfFields        
            }
          }
        }
      }
    }

    fragment CommentMirrorOfFields on Comment {
      ...CommentBaseFields
      mainPost {
        ... on Post {
          ...PostFields
        }
        ... on Mirror {
          ...MirrorBaseFields
        }
      }
    }

    fragment FollowModuleFields on FollowModule {
      ... on FeeFollowModuleSettings {
        type
        amount {
          asset {
            name
            symbol
            decimals
            address
          }
          value
        }
        recipient
      }
      ... on ProfileFollowModuleSettings {
        type
        contractAddress
      }
      ... on RevertFollowModuleSettings {
        type
        contractAddress
      }
      ... on UnknownFollowModuleSettings {
        type
        contractAddress
        followModuleReturnData
      }
    }

    fragment CollectModuleFields on CollectModule {
      __typename
      ... on FreeCollectModuleSettings {
        type
        followerOnly
        contractAddress
      }
      ... on FeeCollectModuleSettings {
        type
        amount {
          asset {
            ...Erc20Fields
          }
          value
        }
        recipient
        referralFee
      }
      ... on LimitedFeeCollectModuleSettings {
        type
        collectLimit
        amount {
          asset {
            ...Erc20Fields
          }
          value
        }
        recipient
        referralFee
      }
      ... on LimitedTimedFeeCollectModuleSettings {
        type
        collectLimit
        amount {
          asset {
            ...Erc20Fields
          }
          value
        }
        recipient
        referralFee
        endTimestamp
      }
      ... on RevertCollectModuleSettings {
        type
      }
      ... on TimedFeeCollectModuleSettings {
        type
        amount {
          asset {
            ...Erc20Fields
          }
          value
        }
        recipient
        referralFee
        endTimestamp
      }
      ... on UnknownCollectModuleSettings {
        type
        contractAddress
        collectModuleReturnData
      }
    }

    fragment ReferenceModuleFields on ReferenceModule {
      ... on FollowOnlyReferenceModuleSettings {
        type
        contractAddress
      }
      ... on UnknownReferenceModuleSettings {
        type
        contractAddress
        referenceModuleReturnData
      }
      ... on DegreesOfSeparationReferenceModuleSettings {
        type
        contractAddress
        commentsRestricted
        mirrorsRestricted
        degreesOfSeparation
      }
    }
    '''
    q = tpl.replace('{{query}}', query)
    data = json.dumps({"query": q})
    resp = requests.post(LENS_API, headers=LENS_HEADERS, data=data)
    print(resp)
    result = resp.json()
    items = result['data']['search']['items']
    return [(i['id'], i['metadata']['content'], i) for i in items]

def get_user_profile(address):
    resp = s3.get_object(
        Bucket='smartcookiesdemo',
        Key=f'profile/{address}.json')
    data = resp['Body'].read()
    print('retrived from s3:', data)
    return json.loads(data)

def recommend(address):
    print(f'recommend({address})')
    # fetch user likes and create user embedding
    profile = get_user_profile(address)

    descs = [desc_token(item['tokenSymbol']) for item in profile['likes'] if 'tokenSymbol' in item]
    print('desc_tokens:', descs)
    embeddings = np.array(map_embedding(descs))

    # match tags
    profile = np.mean(embeddings, axis=0) # n x 1
    similarity = tag_db.dot(profile).flatten().tolist()
    sorted_list = sorted([(e,i) for i,e in enumerate(similarity)], reverse=True)
    top_tags = [(score, tag_id, tag_names[tag_id]) for score, tag_id in sorted_list[:5]]
    print('top tags:', top_tags)

    # get posts from Lens and filter based on tags
    search_results = {}
    search_response = {}
    related_tags = {}
    for (_, _, tag) in top_tags:
        for pub_id, text, item in lens_search(tag):
            search_results[pub_id] = text
            search_response[pub_id] = item
            if pub_id not in related_tags:
                related_tags[pub_id] = []
            related_tags[pub_id].append(tag)

    strs = [text for id, text in search_results.items()]
    ids = [id for id, text in search_results.items()]
    search_embeddings = np.array(map_embedding(strs))

    similarity = search_embeddings.dot(profile).flatten().tolist()
    sorted_list = sorted([(e,i) for i,e in enumerate(similarity)], reverse=True)

    def map_item(score, text_id):
        resp = search_response[ids[text_id]]
        suggestedBy = related_tags[ids[text_id]]
        pic = None
        try:
            pic = resp['profile']['picture']['original']['url']
        except:
            print('missing picture') # , resp['profile']
        return {
            'score': score, 
            'textId': text_id,
            'text': strs[text_id],
            'pubId': ids[text_id],
            'name': resp['profile']['name'],
            'pic': pic,
            'createdAt': resp['createdAt'],
            'stats': resp['stats'],
            'suggestedBy': suggestedBy,
        }

    result = [map_item(score, text_id) for score, text_id in sorted_list]
    print(result)
    return {
        'userTags': top_tags,
        'feed': result,
    }

# recommend('test-addr-0xAddress')

# ---- token recommendation ----

import json
with open('tmp/data/uniswap-default.json', 'r') as fin:
    uni_default = json.load(fin)['tokens']

for obj in uni_default:
    keywords = [obj['name'], obj['symbol']]
    if obj['address']:
        keywords.append(obj['address'])
    keywords = [i.lower() for i in keywords]
    obj['keywords'] = keywords

def map_uni_token(token):
    token = token.lower()
    found = None
    for crypto in uni_default:
        if token in crypto['keywords']:
            found = crypto
            break
    return found

def uni_recommend(address):
    profile = get_user_profile(address)
    uni_items = [map_uni_token(item['tokenSymbol']) for item in profile['likes'] if 'tokenSymbol' in item]
    uni_items = [i for i in uni_items if i]
    print('uni_recommend:', uni_items)
    return uni_items

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/feed/<addr>")
def feed(addr):
    return jsonify(recommend(addr))

@app.route("/uni/<addr>")
def uni(addr):
    return jsonify(uni_recommend(addr))

if __name__=='__main__':
    app.run(debug=True)