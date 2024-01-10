"""
Cordis API
==========

Extract all Cordis data via the API, by project.
"""

import requests
import pandas as pd
import json
from retrying import retry
from nesta.packages.decorators.ratelimit import ratelimit
from nesta.packages.misc_utils.camel_to_snake import camel_to_snake
from json.decoder import JSONDecodeError
from requests.exceptions import HTTPError

TOP_PREFIX = 'http://cordis.europa.eu/{}'
CSV_URL = TOP_PREFIX.format('data/cordis-{}projects.csv')

INFO_FIELDS = ['rcn', 'acronym', 'startDateCode',
               'endDateCode', 'framework',
               'fundedUnder', 'status', 'title',
               'ecContribution', 'totalCost', 'website']
OBJS_FIELDS = ['fundingScheme', 'objective', 'projectDescription',
               'topics', 'proposalCall']
REPS_FIELDS = ['rcn', 'finalResults', 'workPerformed',
               'teaser', 'summary', 'title']
ORGS_FIELDS = ['activityType', 'address', 'contribution',
               'country', 'name', 'organizationId',
               'type', 'website']

USER_AGENT = ('Mozilla/5.0 (Linux; Android 6.0; '
              'Nexus 5 Build/MRA58N) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/81.0.4044.92 '
              'Mobile Safari/537.36')


def generate_id(text):
    """Deterministically generates an ID from a given text.
    NOT guaranteed to be unique, but the alternative was to
    either drop some data for not having IDs, or 
    generating uniquely on the fly: which is hard to do on
    a batch system.
    
    A negative integer is returned to avoid conflicts 
    with the data which have ids already. 8 digits are
    returned, since 9 are the maximum allowed in the schema.

    Args:
        text (str): Text to convert to a negative 8-digit integer
    Returns:
        _id (int): A negative 8-digit integer.
    """
    # Start from the second digit to allow for 
    _id = str(int.from_bytes(text.encode(), 'big',
                             signed=False))
    end = 9 if len(_id) > 8 else None
    start = 1 if len(_id) > 8 else None
    return -int(_id[start:end])


@retry(stop_max_attempt_number=10)
@ratelimit(max_per_second=10)
def hit_api(api='', rcn=None, content_type=None):
    """
    Hit the Cordis API by project code

    Args:
        api (str): Assumed to support '' (cordis) or 'openaire'.
        rcn (str): RCN id of the project or entity to find.
        content_type (str): contenttype argument for Cordis API
    Returns:
        data (json)
    """
    url = TOP_PREFIX.format('api/details')
    if api is not None:
        url = f'{url}/{api}'
    r = requests.get(url, params={'lang': 'en',
                                  'rcn': rcn,
                                  'paramType': 'rcn',
                                  'contenttype': content_type},
                     headers={'User-Agent':USER_AGENT})
    # Not all projects have data, so this is not an error
    if r.status_code in (404, 500):
        p = r.json()['payload']
        if p['errorType'] == 'ica' or 'does not exist!' in p['message']:
            return None
    r.raise_for_status()
    return r.json()['payload']


def extract_fields(data, fields):
    """
    Extract specific fields and flatten data from Cordis API.

    Args:
        data (dict): A row of data to be processed.
        fields (list): A list of fields to be extracted.
    Returns:
        out_data (dict): Flatter data, with specific fields extracted.
    """
    out_data = {}
    for field in fields:
        if field not in data:
            continue
        value = data[field]
        if type(value) is list:
            value = [{k: _row[k] for k in ['title', 'rcn']}
                     for _row in value]
        snake_field = camel_to_snake(field)
        out_data[snake_field] = value
    return out_data


def get_framework_ids(framework, nrows=None):
    """
    Get all IDs of projects by funding framework.

    Args:
        framework (str): 'fp7' or 'h2020'
    Returns:
        ids (list)
    """
    df = pd.read_csv(CSV_URL.format(framework),
                     nrows=nrows,
                     engine='c',
                     decimal=',', sep=';',
                     error_bad_lines=False,
                     warn_bad_lines=True,
                     encoding='latin')
    col = 'rcn' if 'rcn' in df.columns else 'projectRcn'
    return list(df[col])


def filter_pubs(pubs):
    """Remove publications without links, and merge
    datasets and publications data together. 
    Also deduplicates publications based on pids.
    
    Args:
        pubs (dict): Publication data from OpenAIRE.
    Returns:
        _pubs (list): Flattened list of input data.
    """
    _pubs, pids = [], set()
    for p in pubs['datasets'] + pubs['publications']:
        if 'pid' not in p:
            continue
        already_found = any(id in pids for id in p['pid'])
        pids = pids.union(p['pid'])
        if already_found or len(p['pid']) == 0:
            continue
        _pubs.append(dict(id=p['pid'][0], **p))
    return _pubs


def fetch_data(rcn):
    """
    Fetch all data (project, reports, orgs, publications)
    for a given project id.

    Args:
        rcn (str): Project id.
    Returns:
        data (tuple): project, orgs, reports, pubs
    """
    # Collect project info
    _project = hit_api(rcn=rcn, content_type='project')
    if _project is None:
        return (None,None,None,None)
    info = _project['information']    
    project = {**extract_fields(info, INFO_FIELDS),
               **extract_fields(_project['objective'],
                                OBJS_FIELDS)}
    # Collect organisations
    orgs = []
    oid_field = 'organizationId'
    for _orgs in _project['organizations'].values():
        for org in _orgs:
            no_id_found = (oid_field not in org or 
                           org[oid_field] == '')
            if 'name' not in org and no_id_found:
                continue
            elif no_id_found:
                org[oid_field] = generate_id(org['name'])
            orgs.append(extract_fields(org, ORGS_FIELDS))

    # Collect result reports
    _reports = [hit_api(rcn=report['rcn'], 
                        content_type='result')
                for report in info['relatedResultsReport']]
    reports = []
    if _reports is not None:
        reports = [extract_fields(rep, REPS_FIELDS)
                   for rep in _reports]
    # Collect publications via OpenAIRE
    try:
        pubs = hit_api(api='openaire', rcn=rcn)
        if pubs is None:
            raise HTTPError
    except (HTTPError, JSONDecodeError):
        pubs = []
    else:
        pubs = filter_pubs(pubs)
    return project, orgs, reports, pubs
