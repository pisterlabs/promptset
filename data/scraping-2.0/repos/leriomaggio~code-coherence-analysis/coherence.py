"""
Caching and Loading the Coherence Dataset
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# Copyright (c) 2015 Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

from os.path import join, abspath, curdir, exists
from operator import or_
from functools import reduce
from itertools import chain

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets.base import Bunch
from scipy.sparse import csr_matrix
import numpy as np

try:
    from coherence_dataset.settings import COHERENT, NOT_COHERENT
    from coherence_dataset.models import Example
    from source_code_analysis.models import SoftwareProject
    from django.db.models import Q
except ImportError:
    pass

import codecs
import pickle
from sklearn.externals import joblib


TARGET_PROJECTS = (
    ('CoffeeMaker', '1.0'),
    ('Jfreechart', '0.6.0'),
    ('Jfreechart', '0.7.1'),
    ('JHotDraw', '7.4.1'),
)

TARGET_FOLDER = join(abspath(curdir), 'data', 'dataset')
TARGET_METADATA_FILENAME = 'coherence_meta.pkz'
TARGET_DATA_FILENAME = 'coherence_data.pkz'
DATASET_DESCRIPTION_FILENAME = 'coherence.rst'


def fetch_coherence_dataset():
    """Fetch the Coherence dataset from the database
    (leveraging on Django ORM settings and connection) and
    returns data ready to be stored locally on cache files.
    Additional *metadata* about the dataset are collected, and
    returned as well.

    Returns
    -------
    data: numpy ndarray of shape (2881, 5642)
        The (dense) array structure containing all the representation of
        methods (i.e. code and comment) in the feature space (VSM).
        Methods are gathered from all the considered software projects and
        are stored in "(class, project)" order - positive ("COHERENT")
        examples first.

    targets: numpy ndarray of shape (2881, )
        The array of targets, stored according to the organization
        of documents/methods in the `data` matrix, namely
        "Coherent" method first, and then "Non Coherent" ones.

    metadata: Bunch  (`sklearn.datasets.base.Bunch`)
        Dictionary-like object containing dataset metadata.
        Stored information (key/value pairs) are:
        'N' : integer
            number of samples (methods)
        'D' : integer
            number of features
        'code' : list
            list of terms extracted from body of methods
        'comments' : list
            list of terms extracted from comments of methods
        'features' : list
            list of feature names (terms) / IR vocabulary
        'target_names' : tuple
            tuple containing the label associated to each class
        'examples_per_classes' : tuple
            tuple containing the numbers of examples in each class
        'projects' : dictionary
            dictionary containing information about the analysed
            software projects (i.e. name, version, offset indices in
            the data matrix, number of examples per target class).
    """
    # --------------------------------------
    # Gather `Example` instances from the Db
    # --------------------------------------
    per_project_filters = list()
    for name, version in TARGET_PROJECTS:
        try:
            project = SoftwareProject.objects.get(name__iexact=name, version__exact=version)
            per_project_filters.append(Q(method__project__id=project.id))
        except SoftwareProject.DoesNotExist:
            continue
    projects_filters = reduce(or_, per_project_filters)
    positive_exs = Example.objects.filter(target=COHERENT).filter(projects_filters)
    negative_exs = Example.objects.filter(target=NOT_COHERENT).filter(projects_filters)
    # Create the Document Collection
    # NOTE: (All) documents of *positive* examples first.
    doc_collection = list()
    for example in chain(positive_exs, negative_exs):
        doc_collection.append(example.method.lexical_info.normalized_comment)
        doc_collection.append(example.method.lexical_info.normalized_code)

    # (META) Textual Data
    comments = doc_collection[::2]  # odd elements correspond to comments
    code = doc_collection[1::2]  # even elements correspond to code

    # Set up the processing pipeline to `fit_transform` data
    pipeline = Pipeline([
        ('vect', CountVectorizer(input='content', lowercase=False)),
        ('tfidf', TfidfTransformer(sublinear_tf=True, norm='l2', use_idf=True)),
    ])
    all_data = pipeline.fit_transform(doc_collection)
    nrows, ncols = all_data.shape

    # Compacting data to (methods, (comment, code)) features
    data = all_data.toarray().reshape(nrows // 2, -1)

    # (META) Textual Feature representation
    # The list of indexed terms
    vectorizer = pipeline.named_steps['vect']
    features = vectorizer.get_feature_names()

    # (META) Count Examples per Classes
    positive_ex_count = positive_exs.count()
    negative_ex_count = negative_exs.count()

    # Create target array, according to examples distribution
    targets = np.hstack((np.ones(positive_ex_count),
                         np.zeros(negative_ex_count)))

    # -----------------------------------
    # Get and Save additional (Meta)data
    # -----------------------------------
    # Per-project information:
    #    - project name
    #    - project version
    #    - Offset of projects entries in dataset
    #    - Examples count (positive and negative)
    #    - "names" : list of keys (name-version) to access
    #                single project data
    projects_info = dict()
    projects_info.setdefault('names', list())

    pos_start = pos_end = 0
    neg_start = neg_end = positive_exs.count()
    for name, version in TARGET_PROJECTS:
        try:
            # 0. Set project key (i.e. `pkey`)
            pkey = '{0}-{1}'.format(name, version)
            projects_info['names'].append(pkey)
            # 1. get project examples
            project = SoftwareProject.objects.get(name__iexact=name, version__exact=version)
            project_examples = Example.objects.filter(method__project__id=project.id)
            # 2. count positive and negative ones,
            # and update the ending coordinates of offsets (positive, and negative)
            pos_count = project_examples.filter(target=COHERENT).count()
            neg_count = project_examples.filter(target=NOT_COHERENT).count()
            pos_end += pos_count
            neg_end += neg_count
            # 3. Store metadata for current project
            projects_info[pkey] = {'name': name, 'version': version,
                                   'positive_examples': (pos_start, pos_end),
                                   'negative_examples': (neg_start, neg_end),
                                   'positive_count': pos_count,
                                   'negative_count': neg_count}
            # 4. Update the starting coordinates for the next offsets count
            pos_start = pos_end
            neg_start = neg_end
        except SoftwareProject.DoesNotExist:
            continue

    # *. Dataset (meta)data descriptors
    #     - N, D --> no. of examples, and no. of features
    #     - Class names and no. of instances per classes
    #     - Project names and versions, along with corresponding offsets
    #     - Target (class) names.
    metadata = Bunch(code=code, comments=comments, features=features,
                     N=nrows, D=ncols, target_names=('COHERENT', 'NOT COHERENT'),
                     examples_per_classes=(positive_ex_count, negative_ex_count),
                     projects=projects_info)
    return data, targets, metadata


def load_coherence_dataset():
    """Load and returns the coherence dataset (classification)

    The Coherence dataset contains information about the
    coherence between the head comment and the implementation
    of a source code methods.

    =================   ==============
    Classes                          2
    Samples per class       (Pos) 1713
                            (Neg) 1168
    Samples total           (Tot) 2881
    Dimensionality                5642
    Unique Terms                  2821
    Features            real, positive
    =================   ==============

    Note:
    -----
    Since methods are gathered from different software projects, to ease data
    analysis (e.g. slicing, splitting or extracting data of a single software project),
    data are stored according to class and project, respectively.
    In particular, data are primarily grouped by classes (all positive instances, first), and then
    further organized per project.

    So far, these are the distribution of examples per software project:
    ======================   ===========================
        Project              Positive | Negative | Total
        CoffeeMaker (1.0)       27    |    20    |   47
        JFreeChart (0.6.0)     406    |    55    |   461
        JFreeChart (0.7.1)     520    |    68    |   588
        JHotDraw (7.4.1)       760    |  1025    |  1785
    ======================   ===========================


    Returns
    -------
    data: Bunch
        Dictionary-like object, the intersting attributes (keys) are:
        'data', the actual data to learn, 'target', 'the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of features (i.e. the `n_features` terms extracted), and
        'DESCR', the full description of the dataset.
    """

    # Setup target (cache) path(s)
    cache_data_path = join(TARGET_FOLDER, TARGET_DATA_FILENAME)
    cache_metadata_path = join(TARGET_FOLDER, TARGET_METADATA_FILENAME)

    if not exists(cache_data_path):
        # get data from db and save files

        data, target, metadata = fetch_coherence_dataset()
        data_sparse = csr_matrix(data, shape=data.shape, dtype=np.float64)

        # Cache Dataset to disk, namely:
        # 1. Dump Data (numpy ndarray file)

        joblib.dump(data, cache_data_path, compress=6)

        # 2. Dump Metadata
        compressed_meta = codecs.encode(pickle.dumps(metadata), 'zlib_codec')

        with open(cache_metadata_path, 'wb') as cache_metadata_file:
            cache_metadata_file.write(compressed_meta)
    else:
        # load data from file
        data = joblib.load(cache_data_path)
        data_sparse = csr_matrix(data, shape=data.shape, dtype=np.float64)

        # load metadata from file
        with open(cache_metadata_path, 'rb') as f:
            compressed_content = f.read()
        uncompressed_content = codecs.decode(
            compressed_content, 'zlib_codec')
        cache_metadata = pickle.loads(uncompressed_content)
        # Store metadata in a Bunch object
        metadata = Bunch()
        metadata.update(cache_metadata)

    # return the Bunch object

    # load Dataset description
    with open(join(TARGET_FOLDER, DATASET_DESCRIPTION_FILENAME)) as rst_file:
        fdescr = rst_file.read()

    # Create target array, according to examples distribution
    positive_ex_count, negative_ex_count = metadata.examples_per_classes
    target = np.hstack((np.ones(positive_ex_count),
                         np.zeros(negative_ex_count)))

    return Bunch(data=data_sparse, target=target,
                 target_names=metadata.target_names,
                 N=metadata.N, D=metadata.D,
                 features=metadata.features, code=metadata.code,
                 comments=metadata.comments, DESCR=fdescr,
                 target_counts = (positive_ex_count, negative_ex_count),
                 projects=metadata.projects)


