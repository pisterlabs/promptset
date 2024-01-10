'''
metadata.py: metadata creation & manipulation

This file is part of https://github.com/caltechlibrary/iga/.

The code in this file constructs a metadata structure in the format expected by
InvenioRDM. It uses data provided in a GitHub release as well as the repository
(and files in the repository), but because GitHub releases and repos don't
directly contain the information needed, we have to resort to looking for and
parsing files in the repo to try to extract the info we want. The most useful
such files are codemeta.json and CITTATION.cff, but we resort to other things
if we can't find them.

CITATION.cff and codemeta.json overlap in their contents, so a natural question
is which one to try to use first. Stephan Druskat wrote the following about
them in https://github.com/citation-file-format/cff-converter-python/issues/4:

* "CodeMeta aims to, generally, provide a minimal schema for metadata
  for research software. It isn't necessarily tailored for citation metadata
  but can be used to provide metadata that can be used for citation."

* "CFF aims to provide a format for the explicit and exclusive provision of
  citation metadata for research software. Things like transformability to
  BibTeX and RIS have been taken into account. As such, CFF is both less
  comprehensive in terms of general metadata (although I will extend it to
  cover the whole key set of CodeMeta at some point), and more "detailed" in
  terms of citation metadata."

Since the use of InvenioRDM is more about archiving repository code than about
citing software, the code below looks for and uses codemeta.json first,
followed by CITATION.cff if a CodeMeta file can't be found.

Copyright (c) 2022-2023 by the California Institute of Technology.  This code
is open-source software released under a BSD-type license.  Please see the
file "LICENSE" for more information.
'''

import arrow
from   commonpy.data_structures import CaseFoldSet, CaseFoldDict
from   commonpy.data_utils import pluralized
from   commonpy.network_utils import scheme as url_scheme
from   itertools import filterfalse
import json5
import os
from   sidetrack import log
import sys
import validators

from iga.data_utils import deduplicated, listified, normalized_url, similar_urls
from iga.exceptions import MissingData
from iga.github import (
    github_account,
    github_file_url,
    github_release,
    github_repo,
    github_repo_contributors,
    github_repo_file,
    github_repo_filenames,
    github_repo_languages,
    GitHubError,
    probable_bot,
)
from iga.id_utils import detected_id, recognized_scheme
from iga.name_utils import split_name, flattened_name
from iga.reference import reference, RECOGNIZED_REFERENCE_SCHEMES
from iga.text_utils import cleaned_text


# Constants.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# It's useful to understand the context of what's going on. The record stored
# in InvenioRDM may have these top-level fields (but might not contain all):
#
# {
#    "$schema": "local://records/record-vX.Y.Z.json",
#    "id": "q5jr8-hny72",
#    "pid": { ... },
#    "pids" : { ... },
#    "parent": { ... },
#    "access" : { ... },
#    "metadata" : { ... },
#    "files" : { ... },
#    "tombstone" : { ... },
#    "created": "...",
#    "updated": "...",
# }
#
# However, what is uploaded to an InvenioRDM server should only contain the
# 'metadata' field, because of the other fields above are added by the system.
# Consequently, IGA only needs to construct the 'metadata' field value. I.e.,
# referring to https://inveniordm.docs.cern.ch/reference/metadata, we are only
# concerned with https://inveniordm.docs.cern.ch/reference/metadata/#metadata
#
# The following is the full set of possible subfields in "metadata".

FIELDS = [
    "additional_descriptions",
    "additional_titles",
    "contributors",
    "creators",
    "dates",
    "description",
    # "formats",            # 2023-03-23 not clear we need this. Skip for now.
    "funding",
    "identifiers",
    "languages",
    "locations",
    "publication_date",
    "publisher",
    "references",
    "related_identifiers",
    "resource_type",
    "rights",
    # "sizes",             # 2023-03-23 not clear we need this. Skip for now.
    "subjects",
    "title",
    "version",
]

# Not all of these need to be provided.  Based on the test cases in
# https://github.com/inveniosoftware/invenio-rdm-records, the minimum set of
# fields that needs to be provided seems to be this:
#
# {
#    "metadata": {
#        "resource_type": { "id": "XYZ", ... },          # note below
#        "title": "ABC",
#        "creators": [
#              {
#                  "person_or_org": {
#                      "family_name": "A",
#                      "given_name": "B",
#                      "type": "C",
#                  }
#              },
#            ],
#        "publication_date": "...date...",
#    }

REQUIRED_FIELDS = [
    "creators",
    "publication_date",
    "resource_type",
    "title"
]

# Vocabularies variable CV gets loaded only if metadata_for_release(...) is
# called. The name mapping is to map the values from caltechdata_api's
# get_vocabularies to something more self-explanatory when used in this file.

CV = {}
CV_NAMES = {'crr'   : 'creator-roles',
            'cor'   : 'contributor-roles',
            'rsrct' : 'resource-types',
            'dty'   : 'description-types',
            'dat'   : 'data-types',
            'rlt'   : 'relation-types',
            'ttyp'  : 'title-types;',
            'idt'   : 'identifier-types'}

# This vocabulary variable is only populated if metadata_for_release(...) is
# called. It's a dict of licenses id's & urls recognized by this Invenio server.

INVENIO_LICENSES = CaseFoldDict()

# URL schemes we accept for URLs in certain situations where a URL type is
# allowed but we don't want to allow some types such as 'data' URLs.
ALLOWED_URL_SCHEMES = ['http', 'https', 'git', 'ftp', 'gopher', 's3', 'svn']


# Exported module functions.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def metadata_for_release(account_name, repo_name, tag, all_metadata):
    '''Return the "metadata" part of an InvenioRDM record.

    Data is gathered from the GitHub release identified by "tag" in the
    repository "repo_name" of the given GitHub "account_name".
    '''
    repo = github_repo(account_name, repo_name)
    release = github_release(account_name, repo_name, tag)

    # We use codemeta.json & CITATION.cff often. Get them now & augment the
    # repo object with them so that field extraction functions can access them.
    repo.codemeta = {}
    repo.cff = {}
    filenames = github_repo_filenames(repo, tag)
    if 'codemeta.json' in filenames:
        codemeta_file = github_repo_file(repo, tag, 'codemeta.json')
        try:
            repo.codemeta = json5.loads(codemeta_file)
        except KeyboardInterrupt:
            raise
        except ValueError:
            from iga.json_utils import partial_json
            log('CodeMeta content has syntactic errors; trying alternate parser')
            repo.codemeta = partial_json(codemeta_file)
        except Exception as ex:         # noqa PIE786
            log('ignoring codemeta.json file because of error: ' + str(ex))
    for name in ['CITATION.cff', 'CITATION.CFF', 'citation.cff']:
        if name in filenames:
            import yaml
            try:
                repo.cff = yaml.safe_load(github_repo_file(repo, tag, name))
            except KeyboardInterrupt:
                raise
            except Exception as ex:     # noqa PIE786
                log(f'ignoring {name} file because of error: ' + str(ex))
            break

    _load_vocabularies()

    # For some fields that contain multiple values, we let the user decide if
    # we should include values from the GitHub repo. The exception is that if
    # there's no CM or CFF file, we always resort to using the repo data.
    include_all = all_metadata or (not repo.codemeta and not repo.cff)

    # The metadata dict is created by iterating over the names in FIELDS and
    # calling each function of that name defined in this (module) file.
    metadata = {}
    this_module = sys.modules[__name__]
    for field in FIELDS:
        log(f'constructing field "{field}"')
        value = getattr(this_module, field)(repo, release, include_all)
        metadata[field] = value
        count = 1 if (value and isinstance(value, (str, dict))) else len(value)
        log(f'finished field "{field}" with {pluralized("item", count, True)}')
    log('done constructing metadata')
    return {"metadata": metadata}


def metadata_from_file(file):
    '''Read a metadata record from the file and apply some basic validation.

    The validation process currently on tests that the record has the
    minimum fields; it does not currently check the field values or types.
    '''
    try:
        log(f'reading metadata provided in file {str(file)}')
        content = file.read().strip()
        metadata = json5.loads(content)
    except KeyboardInterrupt:
        raise
    except Exception as ex:             # noqa PIE786
        log(f'problem trying to read metadata from {str(file)}: ' + str(ex))
        return False

    if 'metadata' not in metadata:
        log('record lacks a "metadata" field')
        return None

    for field in REQUIRED_FIELDS:
        if field not in metadata.get('metadata', {}):
            log(f'metadata structure lacks required field "{field}"')
            return None
    else:
        log(f'metadata in {file} validated to have minimum fields')
        return metadata


# Field value functions.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Summary of the approach: the functions for extracting values from GitHub
# are named after the fields themselves, so that record_from_release(...)
# above can create a record by simply iterating over the names in FIELDS and
# calling the function of that name to get the value for that field.

def additional_descriptions(repo, release, include_all):
    '''Return InvenioRDM "additional descriptions".
    https://inveniordm.docs.cern.ch/reference/metadata/#additional-descriptions-0-n
    '''
    # Description types come from DataCite, and can be: "abstract", "methods",
    # "series-information", "table-of-contents", "technical-info", "other".

    descriptions = []

    # We don't want to add repeated text, so we track what we have seen. Start
    # with the text we put in the InvenioRDM "description" field.
    added = [description(repo, release, include_all, internal_call=True).lower()]

    # This is a helper function used in what follows next. All the fields used
    # below are supposed to be strings or URLs, per the CodeMeta & CFF specs.
    def add(item, role, summary):
        if item is None:
            log(f'not using {summary} because it\'s empty')
            return
        if item and not isinstance(item, str):
            log(f'not using {summary} because it\'s not the expected data type')
            return
        item = item.strip()
        if not item:
            return
        elif item.lower() in added:
            log(f'not using {summary} because it\'s a duplicate of something else')
        else:
            if validators.url(item):
                if url_scheme(item) in ALLOWED_URL_SCHEMES:
                    item = ("Additional information is available at"
                            f" <a href='{item}'>{item}</a>")
                else:
                    log(f'not using {summary} URL {item} due to disallowed scheme')
                    return
            log(f'tentatively adding {summary} as an additional description')
            descriptions.append({'description': item,
                                 'type': {'id': role}})
            added.append(item.lower())

    add(repo.codemeta.get('releaseNotes', ''), 'other', 'CodeMeta "releaseNotes"')
    add(repo.codemeta.get('description', ''), 'other', 'CodeMeta "description"')
    # Note: DataCite defines a type called "abstract", naturally it's tempting
    # to use it for the CFF "abstract" field. IMHO that would be wrong because
    # CFF's definition of the "abstract" field is "a description of the
    # software or dataset" -- i.e., more like the other "description" fields
    # than like an abstract. Thus, we should use the same type ('other') here.
    add(repo.cff.get('abstract', ''), 'other', 'CFF "abstract"')
    if include_all:
        add(repo.description, 'other', 'GitHub repo "description"')

    # CodeMeta's "readme" maps to DataCite's "technical-info". (DataCite's docs
    # say "For software description, this may include a readme.txt ...".)
    add(repo.codemeta.get('readme', ''), 'technical-info', 'CodeMeta "readme"')

    return deduplicated(descriptions)


def additional_titles(repo, release, include_all):
    '''Return InvenioRDM "additional titles".
    https://inveniordm.docs.cern.ch/reference/metadata/#additional-titles-0-n
    '''
    # The main title will use CodeMeta name, CFF title, or repo full_name.
    # Here we add the ones that didn't get used.

    have_cm_name   = bool(repo.codemeta.get('name', ''))
    have_cff_title = bool(repo.cff.get('title', ''))

    # CodeMeta name has a value => we won't have used CFF title yet. Use it now.
    add_cff_title = have_cm_name and have_cff_title
    # No CodeMeta name => we'll have used repo name if have no CFF title.
    add_repo_name = (have_cm_name or have_cff_title) and include_all

    titles = []
    if add_cff_title:
        log('adding CFF "title" as additional title')
        titles.append({'title': cleaned_text(repo.cff.get('title', '')),
                       'type': {'id': 'alternative-title'},
                       'lang': {'id': 'eng'},
                       })
    if add_repo_name:
        log('adding GitHub repo "full_name" as additional title')
        titles.append({'title': cleaned_text(repo.full_name),
                       'type': {'id': 'alternative-title'},
                       'lang': {'id': 'eng'},
                       })

    return deduplicated(titles)


def contributors(repo, release, include_all):
    '''Return InvenioRDM "contributors".
    https://inveniordm.docs.cern.ch/reference/metadata/#contributors-0-n
    '''
    contributors = []

    # CFF's contact field is defined as a single object.
    if contact := repo.cff.get('contact', {}):
        log('adding CFF "contact" value(s) as contributor(s)')
        contributors.append(_entity(contact, role='contactperson'))

    # CodeMeta's "sponsor" is a person or org, but people might use a list.
    for sponsor in listified(repo.codemeta.get('sponsor', {})):
        log('adding CodeMeta "sponsor" value(s) as contributor(s)')
        contributors.append(_entity(sponsor, role='sponsor'))

    # CodeMeta's "producer" is a person or org, but people might use a list.
    for producer in listified(repo.codemeta.get('producer', {})):
        log('adding CodeMeta "producer" value(s) as contributor(s)')
        contributors.append(_entity(producer, role='producer'))

    # CodeMeta's "editor" is a person or org, but people might use a list.
    for editor in listified(repo.codemeta.get('editor', {})):
        log('adding CodeMeta "editor" value(s) as contributor(s)')
        contributors.append(_entity(editor, role='editor'))

    # CodeMeta's "copyrightHolder" is a person or org, but ... oh you know.
    for copyrightHolder in listified(repo.codemeta.get('copyrightHolder', {})):
        log('adding CodeMeta "copyrightHolder" value(s) as contributor(s)')
        contributors.append(_entity(copyrightHolder, role='rightsholder'))

    # For the next bunch, we have a problem. Their roles are all "other", which
    # leads to InvenioRDM to displaying them all in a section titled "Other".
    # If a CodeMeta files includes, say, a maintainer who is also an author,
    # then that name will show up in the "Other" list as well as the creators
    # list, which adds nothing and just looks like a mistake. So the strategy
    # here is avoid adding people we already have added as authors. FIXME: if
    # InvenioRDM ever adds new role definitions for 'maintainer' and 'provider'
    # we can move those up above, and use them without deduplication.

    authors = creators(repo, release, include_all, internal_call=True)

    # CodeMeta's "maintainer" is person or org, but people often use a list.
    # InvenioRDM roles lack an explicit term for maintainer, so we use "other".
    for maintainer in listified(repo.codemeta.get('maintainer', {})):
        entity = _entity(maintainer, role='other')
        if not any(_entity_match(entity, author) for author in authors):
            log('adding CodeMeta "maintainer" value(s) as contributor(s)')
            contributors.append(entity)
        else:
            log(f'skipping "maintainer" {entity} who is already in "authors"')

    # 2023-03-31 I'm not sure "provider" should be counted in contributors.
    # InvenioRDM lacks an explicit term for "provider", so we use "other"
    # for provider in listified(repo.codemeta.get('provider', {})):
    #     entity = _entity(provider, role='other')
    #     if not any(_entity_match(entity, author) for author in authors):
    #         log('adding CodeMeta "provider" value(s) as contributor(s)')
    #         contributors.append(entity)

    # CodeMeta has a list of contributors, but without role information.
    if contribs := listified(repo.codemeta.get('contributor', [])):
        log('adding CodeMeta "contributor" value(s) as contributor(s)')
        for contributor in contribs:
            entity = _entity(contributor, role='other')
            if not any(_entity_match(entity, author) for author in authors):
                contributors.append(entity)
            else:
                log(f'skipping CodeMeta "contributor" {entity} who is in "authors"')
    elif include_all and (repo_contributors := github_repo_contributors(repo)):
        # If CodeMeta doesn't contain contributors, use the repo's, if any.
        # Skip bot accounts.
        for account in filterfalse(probable_bot, repo_contributors):
            entity = _identity_from_github(account, 'other')
            if not any(_entity_match(entity, author) for author in authors):
                log(f'adding GitHub repo contributor {entity} as contributor(s)')
                contributors.append(entity)
            else:
                log(f'skipping GitHub repo contributor {entity} who is in "authors"')

    # We're getting data from multiple sources & we might have duplicates.
    # Deduplicate based on names & roles only.
    if contributors:
        log('deduplicating overall list of contributors -- some may be removed')
    result = []
    seen = []
    for entry in contributors:
        item = entry['person_or_org']
        role = entry['role']['id']
        key = (item, role)
        if key not in seen:
            seen.append(key)
            result.append(entry)
    return result


def creators(repo, release, include_all, internal_call=False):
    '''Return InvenioRDM "creators".
    https://inveniordm.docs.cern.ch/reference/metadata/#creators-1-n
    '''
    # Helper function.
    def log_decision(text):
        if not internal_call:
            log('adding ' + text + ' as creator(s)')

    # CodeMeta & CFF files contain more complete author info than the GitHub
    # release data, so try them 1st.
    if authors := listified(repo.codemeta.get('author', [])):
        log_decision('CodeMeta "author" name(s)')
    elif authors := repo.cff.get('author', []):
        log_decision('CFF "author" name(s)')
    if authors:
        return deduplicated(_entity(x) for x in authors)

    # Couldn't get authors from codemeta.json or CITATION.cff. Try the release
    # author first, followed by the repo owner.
    if identity := _release_author(release):
        log_decision('GitHub release author')
    elif identity := _repo_owner(repo):
        log_decision('GitHub repo owner name')
    if identity:
        return [identity]

    # A release in InvenioRDM can't be made without author data.
    raise MissingData('Unable to extract author info from GitHub release or repo.')


def dates(repo, release, include_all):
    '''Return InvenioRDM "dates".
    https://inveniordm.docs.cern.ch/reference/metadata/#dates-0-n
    '''
    dates = []

    # If we used a different date for the publication_date value than the
    # release date in GitHub, we add release date as another type of date.
    pub_date = publication_date(repo, release, include_all)
    github_date = arrow.get(release.published_at).format('YYYY-MM-DD')
    if pub_date != github_date:
        log('adding the GitHub release "published_at" date as the "available" date')
        dates.append({'date': github_date,
                      'type': {'id': 'available'}})

    # CodeMeta has a "dateCreated" field, which the CodeMeta crosswalk equates
    # to the GitHub repo "created_at" date.
    if created_date := repo.codemeta.get('dateCreated', ''):
        log('adding the CodeMeta "dateCreated" as the "created" date')
    elif include_all and (created_date := repo.created_at):
        log('adding the GitHub repo "created_at" as the "created" date')
    if created_date:
        dates.append({'date': arrow.get(created_date).format('YYYY-MM-DD'),
                      'type': {'id': 'created'}})

    # CodeMeta has a "dateModified" field, which the CodeMeta crosswalk equates
    # to the GitHub repo "updated_at" date.
    if mod_date := repo.codemeta.get('dateModified', ''):
        log('adding the CodeMeta "dateModified" as the "updated" date')
    elif include_all and (mod_date := repo.updated_at):
        log('adding the GitHub repo "updated_at" date as the "updated" date')
    if mod_date:
        dates.append({'date': arrow.get(mod_date).format('YYYY-MM-DD'),
                      'type': {'id': 'updated'}})

    # CodeMeta has a "copyrightYear", but there's no equivalent elsewhere.
    if copyrighted := str(repo.codemeta.get('copyrightYear', '')):
        log('adding the CodeMeta "copyrightYear" date as the "copyrighted" date')
        dates.append({'date': arrow.get(copyrighted).format('YYYY-MM-DD'),
                      'type': {'id': 'copyrighted'}})
    return dates


def description(repo, release, include_all, internal_call=False):
    '''Return InvenioRDM "description".
    https://inveniordm.docs.cern.ch/reference/metadata/#description-0-1
    '''
    from iga.text_utils import html_from_md

    # The description that a user provides for a release in GitHub is stored
    # in the release data as "body". If the user omits the text, GitHub
    # automatically (sometimes? always?  not sure) displays text pulled from
    # commit messages. In those cases, the value of release.body that we get
    # through the API is empty. There doesn't seem to be a way to get the text
    # shown by GitHub in those cases, so we try other alternatives after this.
    if release.body:
        if internal_call:
            return release.body.strip()
        else:
            log('adding GitHub release body text as "description"')
            return html_from_md(release.body.strip())

    # CodeMeta releaseNotes can be either text or a URL. If it's a URL, it
    # often points to a NEWS or ChangeLog or similar file in their repo.
    # Those files often describe every release ever made, and that just
    # doesn't work well for the purposes of an InvenioRDM record description.
    if rel_notes := repo.codemeta.get('releaseNotes', '').strip():
        if not validators.url(rel_notes):
            if internal_call:
                return rel_notes
            else:
                log('adding CodeMeta "releaseNotes" as "description"')
                return html_from_md(rel_notes)
        elif not internal_call:
            log('CodeMeta has releaseNotes in the form of a URL -- skipping')

    # CodeMeta's "description" & CFF's "abstract" (which the CodeMeta crosswalk
    # maps as equivalent) and GitHub's repo "description" field refer to the
    # software or dataset overall, not specifically to the release. Still, if
    # there's nothing else, it seems better to use this instead of leaving an
    # empty description in the record. We do this regardless of include_all.
    value_name = ''
    if text := repo.codemeta.get('description', ''):
        value_name = 'CodeMeta "description"'
    elif text := repo.cff.get('abstract', ''):
        value_name = 'CFF "abstract"'
    elif text := repo.description:
        value_name = 'GitHub repo "description"'
    if text:
        if internal_call:
            return text.strip()
        else:
            log(f'adding {value_name} as "description"')
            return html_from_md(text.strip())

    # Bummer. Invenio won't accept an empty string, so we have to put something.
    log('could not find a usable value for the "description" field')
    return '(No description provided.)'


def formats(repo, release, include_all):
    '''Return InvenioRDM "formats".
    https://inveniordm.docs.cern.ch/reference/metadata/#formats-0-n
    '''
    formats = []
    if release.zipball_url:
        formats.append("application/zip")
    if release.tarball_url:
        formats.append("application/x-tar-gz")
    for asset in release.assets:
        formats.append(asset.content_type)
    return formats


def funding(repo, release, include_all):
    '''Return InvenioRDM "funding references".
    https://inveniordm.docs.cern.ch/reference/metadata/#funding-references-0-n
    '''
    # InvenioRDM funding references must have funder; award info is optional.

    # CITATION.cff doesn't have anything for funding currently, nor does GitHub.
    # codemeta.json has "funding" & "funder": https://codemeta.github.io/terms/.
    # Sometimes people mistakenly put funder info inside the "funding" items.
    funding_field = repo.codemeta.get('funding', '')
    funder_field  = repo.codemeta.get('funder', '')

    # Some people don't actually provide funding info, yet don't leave the
    # fields blank. We don't bother putting these in the InvenioRDM record.
    not_avail = ['n/a', 'not available', 'none']
    for item in listified(funding_field):
        if isinstance(item, str) and any(t in item.lower() for t in not_avail):
            log('CodeMeta "funding" has a value but it says "not available"')
            return []

    # Funder is supposed to be a single item (a dict), but I've seen people use
    # a string and also a list of dict. If a list, things get tricky later.
    funder_tuples = []
    for item in listified(funder_field):
        if isinstance(item, dict):
            # Correct data type.
            funder_tuples.append(_parsed_funder_info(item))
        elif isinstance(item, str):
            # Incorrect data type. Let's take it anyway. Use the whole string.
            funder_tuples.append((item, ''))

    # If we have multiple values for funder, we can only use them if we do NOT
    # also have funding values because if we have multiple funders, we have no
    # way to match funders with funding values. However, Invenio allows a list
    # of funders only, so we return that if we find funders but not funding.
    if len(funder_tuples) > 1:
        log('CodeMeta "funder" has multiple values')
        if not funding_field:
            log('using CodeMeta "funder" values to build "funding" field')
            return [_funding(funder, fid) for (funder, fid) in funder_tuples]
        else:
            log('CodeMeta has multiple funders and funding values – giving up')
            return []

    # If we get here, it means we DON'T have more than one funder/funder_id.
    funder, funder_id = funder_tuples[0] if funder_tuples else ('', '')
    results = []
    for item in listified(funding_field):
        # The correct funding type is text. Sometimes people use lists or dict.
        if isinstance(item, str):
            # InvenioRDM requires EITHER an id drawn from OpenAIRE, OR a grant
            # name + number pair. We can't reliably parse grant numbers & names
            # from strings, and we don't currently have a way to detect OpenAIRE
            # grant id's. All we can do is report the funder. And because if
            # we get here we know we only have one funder, we can also quit now.
            if funder or funder_id:
                log('"funding" value is a string – giving up, returning funder')
                return [_funding(funder, funder_id)]
            else:
                log('"funding" value is a string and we have no funder – giving up')
                return []
        elif isinstance(item, dict):
            # The type used by CodeMeta for funding does not have a separate
            # number field, and we can't reliably parse grant numbers & names
            # out of a plain-text name string, so at best we can only do this:
            #
            #  case 1: if there's a separate identifier, (mis)use that as the
            #          "number" if the funding item also gives a "name".
            #
            #  case 2: if there's a separate funder item within the funding
            #          item, and case #1 doesn't apply, return just the funder.
            award_name = item.get('name', '') or item.get('@name', '')
            award_id = item.get('identifier', '') or item.get('@id', '')
            item_funder = ''
            item_funder_id = ''
            if fun := item.get('funder', {}):
                if isinstance(fun, str):
                    item_funder = fun
                elif isinstance(fun, dict):
                    item_funder, item_funder_id = _parsed_funder_info(fun)
            # If there's an overall funder name in the CodeMeta file & this
            # item also has its own funder name, use this item's value.
            item_funder = item_funder or funder
            item_funder_id = item_funder_id or funder_id
            if not item_funder or item_funder_id:
                continue
            log('using CodeMeta "funding" value')
            # 2023-03-31 We can't know what type of award ID the user is using,
            # and InvenioRDM gives an error if it can't find the award id, so
            # we can't tell if we're supplying an award id that Invenio will
            # like. Disabling this until I can figure out what to do here.
            #
            # if award_id and not award_name:
            #     results.append(_funding(item_funder, item_funder_id,
            #                             award_id=award_id))
            if award_id and award_name:
                results.append(_funding(item_funder, item_funder_id,
                                        award_name=award_name, award_num=award_id))
            else:
                results.append(_funding(item_funder, item_funder_id))
    return deduplicated(results)


def identifiers(repo, release, include_all):
    '''Return InvenioRDM "alternate identifiers".
    https://inveniordm.docs.cern.ch/reference/metadata/#alternate-identifiers-0-n

    This is defined as "persistent identifiers for the resource other than the
    ones registered as system-managed internal or external persistent
    identifiers. This field is compatible with 11. Alternate Identifiers in
    DataCite."
    '''
    identifiers = []
    # CodeMeta's "identifier" can be a URL, text or dict. CFF's "identifiers"
    # can be an array of dict. Make lists out of all of them & iterate.
    if cm_ids := listified(repo.codemeta.get('identifier', [])):
        log('using CodeMeta "identifier" value(s) for "identifiers" field')
    if cff_ids := repo.cff.get('identifiers', []):
        log('using CFF "identifier" value(s) for "identifiers" field')
    for item in cm_ids + cff_ids:
        if isinstance(item, str):
            kind = recognized_scheme(item)
            value = item.strip()
        elif isinstance(item, dict):
            kind = item.get('type', '').lower() or item.get('@type', '').lower()
            value = item.get('value', '').strip()
        else:
            log(f'skipping due to unsupported item data type: {item}')
            continue
        if not value:
            continue
        if kind == 'url' and value not in ALLOWED_URL_SCHEMES:
            log(f'skipping due to unsupported URL type: {value}')
            continue
        if kind != 'url' and validators.url(value):
            # Original value is in URL form, but we recognzie the kind as a
            # particular sort of identifier. Try to extract the identifier.
            value = detected_id(value) or value
        if kind in CV['identifier-types'].values():
            identifiers.append({'identifier': value,
                                'scheme': kind})
        else:
            log(f'skipping due to unsupported identifier type: {value}')
    return deduplicated(identifiers)


def languages(repo, release, include_all):
    '''Return InvenioRDM "languages".
    https://inveniordm.docs.cern.ch/reference/metadata/#languages-0-n
    '''
    # GitHub doesn't provide a way to deal with any other human language.
    log('adding "eng" to "languages"')
    return [{"id": "eng"}]


def locations(repo, release, include_all):
    '''Return InvenioRDM "locations".
    https://inveniordm.docs.cern.ch/reference/metadata/#locations-0-n
    '''
    log('adding empty "locations"')
    return {}


def publication_date(repo, release, include_all):
    '''Return InvenioRDM "publication date".
    https://inveniordm.docs.cern.ch/reference/metadata/#publication-date-1
    '''
    # InvenioRDM's publication_date is the date "when the resource was made
    # available". GitHub's release date is not necessarily the same -- someone
    # might do a release retroactively. So instead, we first try CodeMeta's
    # "datePublished", then CFF's "date-released", and finally the GitHub date.
    if date := repo.codemeta.get('datePublished', ''):
        log('adding CodeMeta "datePublished" as "publication_date"')
    elif date := repo.cff.get('date-released', ''):
        log('adding CFF "date-released" as "publication_date"')
    else:
        date = release.published_at
        log('adding GitHub repo "published_at" as "publication_date"')
    return arrow.get(date).format('YYYY-MM-DD')


def publisher(repo, release, include_all):
    '''Return InvenioRDM "publisher".
    https://inveniordm.docs.cern.ch/reference/metadata/#publisher-0-1
    '''
    if not (name := os.environ.get('INVENIO_SERVER_NAME', '')):
        # It should be set by cli.py during normal operation. During testing or
        # unanticipated situations, let's be careful to have a fallback here.
        from iga.invenio import invenio_server_name
        name = invenio_server_name(os.environ.get('INVENIO_SERVER', ''))
        if name is None:
            name = ''
    return name


def references(repo, release, include_all):
    '''Return InvenioRDM "references".
    https://inveniordm.docs.cern.ch/reference/metadata/#references-0-n
    '''
    # We also add these items as related identifiers (c.f. the corresponding
    # function) b/c InvenioRDM doesn't do much with "references". Still useful
    # b/c it stores free text references & provides compatibility w/ Zenodo.

    # CodeMeta has "referencePublication". CFF has "references" & also
    # "preferred-citation". We collect what we can parse & try to make the list
    # unique. We're hampered by a lack of tools for parsing references from
    # CodeMeta & CFF files (as of Feb. 2023, even cffconvert doesn't handle
    # "references" or "preferred-citation") & the variety of things people put
    # in. For these reasons, we currently only work with things that have
    # recognizable id's. (Otherwise, we'd have to parse multiple bib formats.)

    # For the output, the InvenioRDM format of this field is very limited:
    #     "references": [{ "reference": "Nielsen et al,..",
    #                      "identifier": "10.1234/foo.bar",
    #                      "scheme": "other" }]
    # The list of allowed schemes is so limited that effectively the only one
    # we can use for publications is "other". The tough one is the "reference"
    # value, which is free text and supposed to be a "full reference string".

    if cm_refs := _codemeta_reference_ids(repo):
        log('adding CodeMeta "referencePublication" value(s) to "references"')
    if cff_refs := _cff_reference_ids(repo):
        log('adding CFF "preferred-citation" and/or "references" to "references"')
    return [{'reference': reference(r), 'identifier': r, 'scheme': 'other'}
            for r in (cm_refs | cff_refs)
            if recognized_scheme(r) in RECOGNIZED_REFERENCE_SCHEMES]


def related_identifiers(repo, release, include_all):
    '''Return InvenioRDM "related identifiers/works".
    https://inveniordm.docs.cern.ch/reference/metadata/#related-identifiersworks-0-n
    '''
    # Note about how to interpret the relations below: the direction is
    #   "this release" --> has relationship to --> "related resource identifier"

    def id_dict(url, rel_type, res_type):
        '''Helper function for creating a frequently-used data structure.'''
        return {'identifier': normalized_url(url),
                'relation_type': {'id': rel_type},
                'resource_type': {'id': res_type},
                'scheme': 'url'}

    log('adding GitHub release "html_url" to "related_identifiers"')
    identifiers = [id_dict(release.html_url, 'isidenticalto', 'software')]

    # The GitHub repo is what this release is derived from. Note: you would
    # expect the GitHub repo html_url, the codemeta.json codeRepository, and
    # the CFF repository-code all to be the same value, but we can't be sure,
    # so we have to look at them, and use them in the order of priority.
    if repo_url := repo.codemeta.get('codeRepository', ''):
        log('adding CodeMeta "codeRepository" to "related_identifiers"')
    elif repo_url := repo.cff.get('repository-code', ''):
        log('adding CFF "repository-code" to "related_identifiers"')
    elif include_all and (repo_url := repo.html_url):
        log('adding GitHub repo "html_url" to "related_identifiers"')
    if repo_url:
        identifiers.append(id_dict(repo_url, 'isderivedfrom', 'software'))

    # If releaseNotes is a URL, we will not have used it for either the
    # description or additional descriptions, so add it here.
    relnotes_url = repo.codemeta.get('releaseNotes', '').strip()
    if validators.url(relnotes_url):
        log('adding CodeMeta "releaseNotes" URL to "related_identifiers"')
        identifiers.append(id_dict(relnotes_url, 'isdescribedby', 'other'))

    # A GitHub repo may give a homepage for the software, though users don't
    # always set it. CFF's "url" field is defined as "The URL of a landing
    # page/website for the software or dataset", which is the same concept.
    # CodeMeta's "url" field is more ambiguously defined as "URL of the item",
    # but the CodeMeta crosswalk table equates it to CFF's url field.
    if homepage_url := repo.codemeta.get('url', ''):
        log('adding CodeMeta "url" to "related_identifiers"')
    elif homepage_url := repo.cff.get('url', ''):
        log('adding CFF "url" to "related_identifiers"')
    elif include_all and (homepage_url := repo.homepage):
        log('adding GitHub repo "homepage" to "related_identifiers"')
    if homepage_url:
        identifiers.append(id_dict(homepage_url, 'isdescribedby', 'other'))

    # CodeMeta's "sameAs" = "URL of a reference Web page that unambiguously
    # indicates the item’s identity." Note that relative to a release stored
    # in InvenioRDM, it is not "same as"; rather it's closer to "a version of".
    # There's no equivalent in CFF or the GitHub repo data structure.
    if sameas_url := repo.codemeta.get('sameAs', ''):
        log('adding CodeMeta "sameAs" to "related_identifiers"')
        identifiers.append(id_dict(sameas_url, 'isversionof', 'software'))

    # CodeMeta's "downloadURL" and CFF's "repository-artifact" are equivalent.
    # Watch out that CM defines it as one URL, but some people make it a list.
    if download_url := repo.codemeta.get('downloadUrl', ''):
        log('adding CodeMeta "downloadUrl" to "related_identifiers"')
    elif download_url := repo.cff.get('repository-artifact', ''):
        log('adding CFF "repository-artifact" to "related_identifiers"')
    for url in filter(validators.url, listified(download_url)):
        identifiers.append(id_dict(url, 'isvariantformof', 'software'))

    # CodeMeta "installUrl" is often used to point to (when working w/ Python)
    # the PyPI location for a program. That's basically like downloadUrl.
    if install_url := listified(repo.codemeta.get('installUrl', '')):
        log('adding CodeMeta "installUrl" to "related_identifiers"')
        for url in filter(validators.url, install_url):
            identifiers.append(id_dict(url, 'isvariantformof', 'software'))

    # CodeMeta softwareHelp type is CreativeWork but sometimes people use URLs.
    for help in listified(repo.codemeta.get('softwareHelp', '')):  # noqa A001
        if isinstance(help, str) and validators.url(help):
            url = help
        elif isinstance(help, dict):
            if help_url := help.get('url', ''):
                url = normalized_url(help_url)
            elif validators.url(help.get('@id', '')):
                url = normalized_url(help.get('@id'))
        # Don't add if has been added already as one of the other URLs above.
        if url and not any(url == item['identifier'] for item in identifiers):
            log(f'adding CodeMeta "softwareHelp" {url} to "related_identifiers"')
            identifiers.append(id_dict(url, 'isdocumentedby',
                                       'publication-softwaredocumentation'))

    # The GitHub Pages URL for a repo usually points to documentation or info
    # about the softare, though we can't tell if it's for THIS release.
    if include_all and repo.has_pages:
        url = f'https://{repo.owner.login}.github.io/{repo.name}'
        if not any(url == item['identifier'] for item in identifiers):
            log('adding the repo\'s GitHub Pages URL to "related_identifiers"')
            identifiers.append(id_dict(url, 'isdocumentedby',
                                       'publication-softwaredocumentation'))

    # The issues URL is kind of a supplemental resource.
    if issues_url := repo.codemeta.get('issueTracker', ''):
        log('adding CodeMeta "issueTracker" to "related_identifiers"')
    elif include_all and repo.issues_url:
        log('adding GitHub repo "issues_url" to "related_identifiers"')
        issues_url = f'https://github.com/{repo.full_name}/issues'
    if issues_url:
        identifiers.append(id_dict(issues_url, 'issupplementedby', 'other'))

    # CodeMeta says "relatedLink" value is supposed to be a URL, but most files
    # use a list. The nature of the relationship is more problematic. The
    # CodeMeta spec says this is "A link related to this object, e.g., related
    # web pages"; however, in the codemeta.json file, we get zero info about
    # what the links are meant to be. Worse, relatedLink has no direct
    # equivalent in the relations CV in InvenioRDM. Since the direction is
    # "this release" --> relatedLink --> "something", the closest relationship
    # term seems to be "references", as in "this release references this link".
    if links := listified(repo.codemeta.get('relatedLink')):
        log('adding CodeMeta "relatedLink" URL value(s) to "related_identifiers"')
        for url in filter(validators.url, links):
            url = normalized_url(url)
            # We don't add URLs we've already added (possibly as another type).
            # The list needs to be recreated in the loop b/c we're adding to it.
            added_urls = [item['identifier'] for item in identifiers]
            # We compare URLs loosely b/c people frequently put https in one
            # place and http in another, or add an extraneous trailing slash.
            if any(similar_urls(url, added) for added in added_urls):
                continue
            # There's no good way to know what the resource type actually is.
            identifiers.append(id_dict(url, 'references', 'other'))

    # We add CodeMeta & CFF "references" to InvenioRDM "references" elsewhere
    # (c.f. function references()). Here we add just the identifiers.
    if reference_ids := _codemeta_reference_ids(repo) | _cff_reference_ids(repo):
        log('adding id\'s of CodeMeta & CFF references to "related_identifiers"')
    for id in reference_ids:            # noqa A001
        # Adding id's => must recreate the list in the test on each iteration.
        if id not in [detected_id(item['identifier']) for item in identifiers]:
            identifiers.append({'identifier': id,
                                'relation_type': {'id': 'isreferencedby'},
                                'scheme': recognized_scheme(id)})

    # Final step: remove things that have urls not in our allowed list.
    filtered_identifiers = []
    for item in identifiers:
        if item['scheme'] != 'url':
            filtered_identifiers.append(item)
        elif url_scheme(item['identifier']) not in ALLOWED_URL_SCHEMES:
            log(f'omitting {item["identifier"]} because of disallowed scheme')
            continue
        else:
            filtered_identifiers.append(item)

    return filtered_identifiers


def resource_type(repo, release, include_all):
    '''Return InvenioRDM "resource type".
    https://inveniordm.docs.cern.ch/reference/metadata/#resource-type-1
    '''
    # The only clear source of info about whether this is software or data is
    # the CFF file field "type", so if we can't use that, default to software.
    if repo.cff.get('type') == 'dataset':
        log('using CFF "type" as "resource_type" (and it is "dataset")')
        return {'id': 'dataset'}
    else:
        log('using default value "software" as "resource_type"')
        return {'id': 'software'}


def rights(repo, release, include_all):
    '''Return InvenioRDM "rights (licenses)".
    https://inveniordm.docs.cern.ch/reference/metadata/#rights-licenses-0-n
    '''
    # Strategy: look in CodeMeta and citation first, trying to recognize common
    # licenses. If that fails, we look at GitHub's license info for the repo.
    # If that also fails, we go rummaging in the repo files.

    rights = []

    # CodeMeta's "license" is usually a URL, but sometimes people don't know
    # that and use the name of a license instead. For CFF, "license" is
    # supposed to be a name. CFF also has a separate "license-url" field.
    if value := repo.codemeta.get('license', ''):
        value_name = 'CodeMeta "license"'
    elif value := repo.cff.get('license', ''):
        value_name = 'CFF "license"'
    elif value := repo.cff.get('license-url', ''):
        value_name = 'CFF "license-url"'
    if value:
        license_id = None
        from iga.licenses import LICENSES, LICENSE_URLS
        if value in LICENSES:
            log(f'found {value_name} value in list of known licenses: {value}')
            license_id = value
        elif validators.url(value):
            # Is it a URL for a known license?
            url = normalized_url(value.lower().removesuffix('.html'))
            if url in LICENSE_URLS:
                log(f'found {value_name} value among known license URLs: {url}')
                license_id = LICENSE_URLS[url]
            else:
                log(f'did not recognize {value_name} value {value}')
        else:
            log('{value_name} has a value but we do not recognize it: ' + value)

        if license_id:
            if license_id in INVENIO_LICENSES:
                rights = {'id': license_id.lower()}
            else:
                rights = {'title': {'en': LICENSES[license_id].title},
                          'link' : LICENSES[license_id].url}
                if LICENSES[license_id].description:
                    rights['description'] = {'en': LICENSES[license_id].description}
            return [rights]
    log('continuing to look for any license info we can use')

    # We didn't recognize license info in the CodeMeta or cff files.
    # Look into the GitHub repo data to see if GitHub identified a license.
    if repo.license and repo.license.name != 'Other':
        from iga.licenses import LICENSES
        log('GitHub has provided license info for the repo – using those values')
        spdx_id = repo.license.spdx_id
        if spdx_id in INVENIO_LICENSES:
            rights = {'id': spdx_id.lower()}
        else:
            rights = {'link': repo.license.url,
                      'title': {'en': repo.license.name}}
            if spdx_id in LICENSES and LICENSES[spdx_id].description:
                log(f'adding our own description for license type {spdx_id}')
                rights['description'] = {'en': LICENSES[spdx_id].description}
        return [rights]
    else:
        log('GitHub did not provide license info for this repo')

    # GitHub didn't fill in the license info -- maybe it didn't recognize
    # the license or its format. Try to look for a license file ourselves.
    filenames = github_repo_filenames(repo, release.tag_name)
    for basename in ['LICENSE', 'License', 'license',
                     'LICENCE', 'Licence', 'licence',
                     'COPYING', 'COPYRIGHT', 'Copyright', 'copyright']:
        for ext in ['', '.txt', '.md', '.html']:
            if basename + ext in filenames:
                log('found a license file in the repo: "' + basename + ext + '"')
                # There's no safe way to summarize arbitrary license text,
                # so we can't provide a 'description' field value.
                rights = [{'title': {'en': 'License'},
                           'link': github_file_url(repo, basename + ext)}]
                break
        else:
            continue
        break
    return rights


def sizes(repo, release, include_all):
    '''Return InvenioRDM "sizes".
    https://inveniordm.docs.cern.ch/reference/metadata/#sizes-0-n
    '''
    return []


def subjects(repo, release, include_all):
    '''Return InvenioRDM "subjects".
    https://inveniordm.docs.cern.ch/reference/metadata/#subjects-0-n
    '''
    # Use a case-insensitive set to try to uniquefy the values.
    subjects = CaseFoldSet()

    # Add values from CodeMeta field "keywords". If the whole value is one
    # string, it may contain multiple terms separated by a comma or semicolon.
    if keywords := repo.codemeta.get('keywords', []):
        log('adding CodeMeta "keywords" value(s) to "subjects"')
    if isinstance(keywords, str):
        # Try the ';' first, alone, before trying ',', in case the values
        # separated by semicolons are subject terms containing commas.
        if ';' in keywords:
            subjects.update(keywords.split(';'))
        elif ',' in keywords:
            subjects.update(keywords.split(','))
        else:
            subjects.update(keywords.split())
    else:
        for item in listified(keywords):
            if isinstance(item, str):
                subjects.add(item)
            else:
                log(f'skipping item with unexpected format: {item}')

    # In CFF, people usually write lists, but I've seen mistakes. Be safe here.
    if keywords := listified(repo.cff.get('keywords', [])):
        log('adding CFF "keywords" value(s) to "subjects"')
    for item in keywords:
        if isinstance(item, str):
            subjects.add(item)
        else:
            log(f'skipping item with unexpected format: {item}')

    # Add the languages listed in the CodeMeta file.
    if cm_langs := listified(repo.codemeta.get('programmingLanguage', [])):
        log('adding CodeMeta "programmingLanguage" value(s) to "subjects"')
    for item in cm_langs:
        if isinstance(item, str):
            subjects.add(item)
        elif isinstance(item, dict):
            if lang := (item.get('name', '') or item.get('@name', '')):
                subjects.add(lang)
        else:
            log('found programmingLanguage item with unrecognized format'
                ' in codemeta.json: ' + str(item))
            break

    if include_all:
        log('adding GitHub topics to "subjects"')
        subjects.update(repo.topics)

        # Add repo languages as topics too.
        if languages := github_repo_languages(repo):
            log('adding GitHub repo languages to "subjects"')
        for lang in languages:
            subjects.add(lang)

    # Always add GitHub as a tag.
    subjects.add('GitHub')

    # Always add a tag about IGA. Users are free to edit it out, but it helps
    # repository maintainers to gauge manual vs automated record creation.
    subjects.add('IGA')

    return [{'subject': x} for x in sorted(subjects, key=str.lower)]


def title(repo, release, include_all):
    '''Return InvenioRDM "title".
    https://inveniordm.docs.cern.ch/reference/metadata/#title-1
    '''
    title = ''
    if text := repo.codemeta.get('name', ''):
        title += text
        field = 'CodeMeta "name"'
    elif text := repo.cff.get('title', ''):
        title += text
        field = 'CFF "title"'
    else:
        title += repo.full_name
        field = 'GitHub repo "full_name"'

    # Note: better not to use a colon here. A lot of CodeMeta files use a name
    # like "title: short description", which would then lead to 2 colons here.
    title += ' – '
    if release.name:
        log(f'using {field} + GitHub release name for "title"')
        title += release.name
    else:
        log(f'using {field} + GitHub release "tag_name" for "title"')
        title += release.tag_name
    return cleaned_text(title)


def version(repo, release, include_all):
    '''Return InvenioRDM "version".
    https://inveniordm.docs.cern.ch/reference/metadata/#version-0-1
    '''
    # Note: this is not really the same as a version number. However, there is
    # no version number in the GitHub release data -- there is only the tag.
    # The following does a weak heuristic to try to guess at a version number
    # from certain common tag name patterns, but that's the best we can do.
    log('adding GitHub release "tag_name" as "version" ')
    tag = release.tag_name
    if tag.startswith('v'):
        import re
        tag = re.sub(r'v(er|version)?[ .]? ?', '', tag)
    return tag.strip()


# Miscellaneous helper functions.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# See https://inveniordm.docs.cern.ch/reference/metadata/#creators-1-n for info
# about the fields for authors and contributors in InvenioRDM. Although
# software authors sometimes put more things in (e.g.) the CFF authors fields
# (e.g., email addr), there's no provision in InvenioRDM for that. We have to
# put together a dict that has only the following keys:
#
#   1. person_or_org (required, type dict):
#       a. type (required, string, either "personal" or "organizational")
#       b. given_name + family_name (if person) OR name (if org)
#       c. identifiers (optional, dict)
#          i) scheme (string, taken from a CV)
#          ii) identifier (string)
#
#   2. role (optional, string, taken from a CV)
#
#   3. affiliations (optional, dict, only if type is "personal")
#       a. id OR name
#
# Note that for people, we MUST produce names split into given + family names
# or InvenioRDM will reject the record.

def _entity(data, role=None):
    if isinstance(data, dict):          # Correct data type.
        return _entity_from_dict(data, role)
    elif isinstance(data, str):         # Wrong data type, but we try anyway.
        return _entity_from_string(data, role)
    else:                               # If we get here, it's beyond us.
        log('entity value is neither a string nor a dict -- giving up')
        return {}


def _entity_from_string(data, role):
    # We don't expect strings for this, so everything we do here is heuristics.
    # Possibilities considered:
    #  - an ORCID URL, like "https://orcid.org/0000-0001-9105-5960"
    #  - an ORCID by itself, like "0000-0001-9105-5960"
    #  - an organization's ROR URL, like "https://ror.org/05dxps055"
    #  - an organization's ROR ID by itself, like "05dxps055"
    #  - a GitHub user's name, like "mhucka"
    #  - a GitHub org account name, like "caltechlibrary"
    #  - a person's name, like "Michael Hucka"
    #  - an organization's name, like "California Institute of Technology"

    result = {}
    scheme = recognized_scheme(data)
    if scheme == 'orcid':
        from iga.orcid import name_from_orcid
        orcid = detected_id(data)
        (given, family) = name_from_orcid(orcid)
        if family or given:
            result = {'person_or_org': {'family_name': flattened_name(family),
                                        'given_name': flattened_name(given),
                                        'identifiers': [{'identifier': orcid,
                                                         'scheme': 'orcid'}],
                                        'type': 'personal'}}
    elif scheme == 'ror':
        from iga.ror import name_from_ror
        name = name_from_ror(data)
        if name:
            result = {'person_or_org': {'name': name,
                                        'type': 'organizational'}}
    elif account := _parsed_github_account(data):
        # It's the name of an account in GitHub.
        result = _identity_from_github(account)
    else:
        # We're getting into expensive heuristic guesswork now.
        from iga.name_utils import is_person
        if is_person(data):
            (given, family) = split_name(data)
            if family or given:
                result = {'person_or_org': {'family_name': flattened_name(family),
                                            'given_name': flattened_name(given),
                                            'type': 'personal'}}
            else:
                log(f'guessing "{data}" is a person but failed to split name')
        else:
            result = {'person_or_org': {'name': data,
                                        'type': 'organizational'}}
    if result and role:
        result['role'] = {'id': role}
    return result


def _entity_from_dict(data, role):
    # This handles data coming from CodeMeta and CFF. CodeMeta uses Schema.org
    # Person or Organization, which define many fields, but we can only use a
    # subset anyway because there's no place in Invenio records to put the rest.
    person = {}
    org = {}

    type_ = data.get('@type', '') or data.get('type', '')
    if type_.lower().strip() == 'person':
        # Deal with field name differences between CodeMeta & CFF.
        family = data.get('family-names', '') or data.get('familyName', '')
        given  = data.get('given-names', '') or data.get('givenName', '')

        id = detected_id(data.get('@id', ''))            # noqa A001
        id_type = recognized_scheme(id)

        if not (family or given) and id_type == 'orcid':
            # If we're lucky and the added an orcid, we can try to use that.
            log('no family & given name fields but have ORCID – trying orcid.org')
            from iga.orcid import name_from_orcid
            (given, family) = name_from_orcid(id)

        # If we didn't get family & given names, try another way.
        if not (family or given) and (name := data.get('name', '')):
            # CodeMeta/schema.org allows a single "name" value. Split it.
            log('no family & given name fields; attempting to split "name" value')
            if isinstance(name, list):
                # The name was given as a list. Weird, but let's roll with it.
                name = ' '.join(name)
            (given, family) = split_name(name)

        if family or given:
            person = {'family_name': flattened_name(family),
                      'given_name': flattened_name(given),
                      'type': 'personal'}
        else:
            # We're out of options. This is not great but the best we can do.
            person = {'family_name': flattened_name(name),
                      'given_name': '',
                      'type': 'personal'}

        if id_type in ['orcid', 'isni', 'gnd']:
            person.update({'identifiers': [{'identifier': id,
                                            'scheme': id_type}]})
    else:
        org = _org_from_dict(data, id_field_name='identifier')
        org['type'] = 'organizational'

    result = {}
    if person or org:
        result = {'person_or_org': person or org}
    if person:
        affiliations = []
        for item in listified(data.get('affiliation', '')):
            if isinstance(item, str):
                affiliations.append({'name': flattened_name(item)})
            elif isinstance(item, dict) and (aff := _org_from_dict(item)):
                affiliations.append(aff)
        if affiliations:
            result['affiliations'] = affiliations
    if role:
        result['role'] = {'id': role}
    return result


def _org_from_dict(data, id_field_name='id'):
    # Hopefully it has a name field or id. If it doesn't, we can't do anything
    # more anyway, and will end up with an empty structure.
    org = {}
    # If it has a name, we take it and we're done.
    if name := (data.get('legalName', '') or data.get('name', '')):
        # In CFF the field name is 'legalName'. In CodeMeta it's 'name'.
        org = {'name': flattened_name(name)}
    else:
        # No name field. See if it has an id field of a type that we recognize.
        id = detected_id(data.get('@id', ''))  # noqa A001
        if recognized_scheme(id) == 'ror':
            from iga.ror import name_from_ror
            if name := name_from_ror(id):
                org = {'name': name}
            else:
                # Got a ROR id but a lookup in ROR.org failed to get a name. We
                # return just the identifier instead. Not ideal but acceptable.
                org = {id_field_name: id}
    return org


def _entity_match(first, second):
    # Match based on names only.
    p1 = first['person_or_org']
    p2 = second['person_or_org']

    p1_orcid = None
    p2_orcid = None

    for item in p1.get('identifiers', []):
        if item.get('scheme', '') == 'orcid':
            p1_orcid = item.get('identifier', '')
    for item in p2.get('identifiers', []):
        if item.get('scheme', '') == 'orcid':
            p2_orcid = item.get('identifier', '')

    if p1_orcid and p2_orcid:
        return p1_orcid == p2_orcid
    elif 'name' in p1 and 'name' in p2:
        return p1['name'] == p2['name']
    elif 'family_name' in p1 and 'family_name' in p2:
        return (p1['family_name'] == p2['family_name']
                and p1.get('given_name', '') == p2.get('given_name', ''))
    return False


def _release_author(release):
    # We can call GitHub's user data API, but it returns very little info
    # about a user (e.g.,, it gives a name but that name is not broken out
    # into family & given name), plus sometimes fields are empty.
    account = github_account(release.author.login)
    return _identity_from_github(account) if account.name else None


def _repo_owner(repo):
    account = github_account(repo.owner.login)
    return _identity_from_github(account)


def _identity_from_github(account, role=None):
    if account.type == 'User':
        if account.name:
            (given, family) = split_name(account.name)
            person_or_org = {'given_name': given,
                             'family_name': family,
                             'type': 'personal'}
        else:
            # The GitHub account record has no name, and InvenioRDM won't pass
            # a record without a family name. All we have is the login name.
            person_or_org = {'given_name': '',
                             'family_name': account.login,
                             'type': 'personal'}

    else:
        name = account.name.strip() if account.name else ''
        person_or_org = {'name': name,
                         'type': 'organizational'}
    result = {'person_or_org': person_or_org}
    if account.company and account.type == 'User':
        account.company = account.company.strip()
        if account.company.startswith('@'):
            # Some people write @foo to indicate org account "foo" in GitHub.
            # Grab only the first token after the '@'.
            log(f'company for {account.login} account starts with @')
            try:
                import re
                candidate = re.search(r'\w+', account.company).group()
                org_account = github_account(candidate)
            except GitHubError:
                # No luck. Take it as-is.
                log(f'failed to find {account.company[1:]} as a GitHub account')
                result['affiliations'] = [{'name': account.company}]
            else:
                log(f'using org {candidate} as affiliation for {account.name}')
                result['affiliations'] = [{'name': org_account.name}]
        else:
            result['affiliations'] = [{'name': account.company}]
    if role:
        result['role'] = {'id': role}
    return result


def _parsed_github_account(data):
    if data.startswith('https://github.com'):
        # Might be the URL to an account page on GitHub.
        tail = data.replace('https://github.com/', '')
        if '/' not in tail and (account := github_account(tail)):
            return account
    elif len(data.split()) == 1 and (account := github_account(data)):
        return account
    return None


def _parsed_funder_info(data):
    funder_name = data.get('name', '') or data.get('@name', '')
    funder_id = ''
    if recognized_scheme(data.get('@id', '')) == 'ror':
        funder_id = detected_id(data.get('@id', ''))
    return (funder_name, funder_id)


def _funding(funder_name, funder_id, award_name=None, award_id=None, award_num=None):
    # InvenioRDM funding items are like this: { 'funder': {...}, 'award': {...} }
    #
    # InvenioRDM says funder subfield must have id OR name, and award subfield
    # must have either id or BOTH title and number.
    result = {}

    if funder_name:
        result['funder'] = {'name': funder_name}
    elif funder_id:
        result['funder'] = {'id': funder_id}

    if award_id:
        result['award'] = {'id': award_id}
    elif award_name and award_num:
        result['award'] = {'title': {'en': award_name},
                           'number': award_num}

    return result


def _codemeta_reference_ids(repo):
    # CodeMeta's referencePublication is supposed to be a ScholarlyArticle
    # (dict), but people often make it a list, and moreover, sometimes they
    # make it a list of strings (often URLs) instead of dicts.
    identifiers = CaseFoldSet()
    for item in listified(repo.codemeta.get('referencePublication', [])):
        if isinstance(item, str):
            if recognized_scheme(item):
                identifiers.add(detected_id(item))
            else:
                log('unrecognized scheme in item: ' + str(item))
        elif isinstance(item, dict):
            for field in ['id', '@id', 'identifier', '@identifier']:
                id_field = item.get(field, '')
                if recognized_scheme(id_field):
                    id = detected_id(id_field)  # noqa A001
                    log(f'found id {id} in CodeMeta "referencePublication"')
                    identifiers.add(id)
                    break
            else:
                log(f'cannot use {item} from CodeMeta "referencePublication"')
        else:
            log('unrecognized referencePublication format: ' + str(item))
    return identifiers


def _cff_reference_ids(repo):
    # CFF has "preferred-citation" and "references". The former is one CFF
    # "reference" type object, while the latter is a list of those objects.
    # Annoyingly, a "reference" object itself can have a list of identifiers.
    identifiers = CaseFoldSet()
    for ref in (listified(repo.cff.get('preferred-citation', []))
                + repo.cff.get('references', [])):
        # These are the relevant field names defined in CodeMeta & CFF.
        for field in ['doi', 'pmcid', 'isbn']:
            if value := ref.get(field, ''):
                identifiers.add(value)
                break
        else:
            for item in ref.get('identifiers', []):
                item_type = item.get('type', '')
                if item_type == 'doi':
                    log(f'found id {item["doi"]} in CFF "preferred-citation"')
                    identifiers.add(item['doi'])
                elif item_type == 'other':
                    value = ref.get('value', '')
                    if recognized_scheme(value):
                        log(f'found id {value} in CFF "preferred-citation"')
                        identifiers.add(detected_id(value))
                    else:
                        log(f'cannot use {value} from CFF "preferred-citation"')
            # Tempting to look in the "url" field if none of the other id
            # fields are present. However, people sometimes set "url" to
            # values that aren't the actual reference => can't trust it.
    return identifiers


def _load_vocabularies():
    from caltechdata_api.customize_schema import get_vocabularies
    from iga.invenio import invenio_vocabulary
    log('loading controlled vocabularies using caltechdata_api module')
    for vocab_id, vocab in get_vocabularies().items():
        CV.update({CV_NAMES[vocab_id]: vocab})
    log('asking InvenioRDM server for its list of software & data licenses')
    for item in invenio_vocabulary('licenses'):
        INVENIO_LICENSES[item['id']] = item['props']['url']


def _cv_match(vocab, term):
    from stringdist import levenshtein
    if not term:
        return None
    for entry in CV[vocab]:
        entry_title = entry['title']['en']
        distance = levenshtein(entry_title, term)
        if distance < 2:
            return entry['id']
    return None
