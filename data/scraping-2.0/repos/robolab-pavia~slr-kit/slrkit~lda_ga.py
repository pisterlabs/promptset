import csv
import dataclasses
import logging
import pathlib
import random
import shutil
import sys
import time
import uuid
from datetime import datetime
from multiprocessing import Pool, Manager
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, List, Union, ClassVar, Dict, Tuple

import numpy as np
import pandas as pd
import tomlkit
from deap import base, creator, algorithms, tools

# disable warnings if they are not explicitly wanted
if not sys.warnoptions:
    import warnings

    warnings.simplefilter('ignore')

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

from slrkit_utils.argument_parser import ArgParse
from lda import (PHYSICAL_CPUS, MIN_ALPHA_VAL,
                 prepare_topics, output_topics, save_toml_files,
                 load_documents)
from utils import setup_logger

EPSILON = 1e-7

logger = None

# these globals are used by the multiprocess workers used in compute_optimal_model
_corpus: Optional[List[List[str]]] = None
_titles: Optional[List[str]] = None
_seed: Optional[int] = None
_modeldir: Optional[pathlib.Path] = None
_coherences: Optional[Dict[int, Tuple[Optional[float], str]]] = None

creator.create('FitnessMax', base.Fitness, weights=(1.0,))


def to_ignore(_):
    return ['lda*.json', 'lda_info*.txt', '*lda_results/']


class BoundsNotSetError(Exception):
    pass


@dataclasses.dataclass(eq=False)
class LdaIndividual:
    """
    Represents an individual, a set of parameters for the LDA model

    The fitness attribute is used by DEAP for the optimization
    topics_bounds, max_no_below, min_no_above are class attribute and are used
    as bounds for the topics, no_above and no_below parameters.
    They must be set with the set_bounds class method before every operation,
    even the creation of a new instance.
    All the other attribute are protected. To access one value, use the
    corresponding property, or its index.
    The association between index and attribute is given by the order_from_name
    and name_from_order class methods.
    The alpha property is used to retrive the actual alpha value to use from the
    _alpha_val and _alpha_type values.
    The random_individual creates a new individual with random values.
    """
    _topics: int
    _alpha_val: float
    _beta: float
    _no_above: float
    _no_below: int
    _alpha_type: int
    fitness: creator.FitnessMax = dataclasses.field(init=False)
    topics_bounds: ClassVar[tuple] = None
    max_no_below: ClassVar[int] = None
    min_no_above: ClassVar[int] = None

    def __post_init__(self):
        if self.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')
        self.fitness = creator.FitnessMax()
        # these assignments triggers the checks on the values
        self.topics = self._topics
        self.alpha_val = self._alpha_val
        self.beta = self._beta
        self.no_above = self._no_above
        self.no_below = self._no_below
        self.alpha_type = self._alpha_type

    def _to_tuple(self):
        """
        Creates a tuple from the components of the individual

        All the float component are truncated using the EPSILON value.
        This tuple can be used to compare two individuals and to calculate a
        good hash value.
        The order of the elements is: alpha, beta, no_above, no_below, topics.
        The alpha component is a string if the alpha_type of the individual is
        not 0.

        :return: a tuple containing the components of the individual
        """
        a = self.alpha
        a = round(a/EPSILON) * EPSILON if isinstance(a, float) else a
        b = round(self.beta/EPSILON) * EPSILON
        na = round(self.no_above/EPSILON) * EPSILON
        return (a, b, self.no_below, na, self.no_below, self.topics)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        tup_self = self._to_tuple()
        tup_other = other._to_tuple()
        eq = False
        for s, o in zip(tup_self, tup_other):
            if type(s) != type(o):
                return False
            if isinstance(s, (int, str)):
                if s != o:
                    return False
            else:
                if abs(s - o) > EPSILON:
                    return False

        return True

    def __hash__(self):
        tup = self._to_tuple()
        return hash(tup)

    @classmethod
    def set_bounds(cls, min_topics, max_topics, max_no_below, min_no_above):
        """
        Sets the bounds used in properties to check the values

        Must be called before every operation
        :param min_topics: minimum number of topics
        :type min_topics: int
        :param max_topics: maximum number of topics
        :type max_topics: int
        :param max_no_below: maximum value for the no_below parameter
        :type max_no_below: int
        :param min_no_above: minimum value for the no_above parameter
        :type min_no_above: float
        :raise ValueError: if min_no_above is > 1.0
        """
        cls.topics_bounds = (min_topics, max_topics)
        cls.max_no_below = max_no_below
        if min_no_above > 1.0:
            raise ValueError('min_no_above must be less then 1.0')
        cls.min_no_above = min_no_above

    @classmethod
    def index_from_name(cls, name: str) -> int:
        """
        Gives the index of a parameter given its name

        :param name: name of the parameter
        :type name: str
        :return: the index of the parameter
        :rtype: int
        :raise ValueError: if the name is not valid
        """
        for i, f in enumerate(dataclasses.fields(cls)):
            if f.name == '_' + name:
                return i
        else:
            raise ValueError(f'{name!r} is not a valid field name')

    @classmethod
    def name_from_index(cls, index: int) -> str:
        """
        Gives the name of a parameter given its index

        :param index: index of the parameter
        :type index: str
        :return: the name of the parameter
        :rtype: str
        :raise ValueError: if the index is not valid
        """
        try:
            name = dataclasses.fields(cls)[index].name
        except IndexError:
            raise ValueError(f'{index!r} is not a valid index')
        return name.strip('_')

    @classmethod
    def random_individual(cls, prob_no_filters=0.5):
        """
        Creates a random individual

        The prob_no_filters is the probability that the created individual has
        no_below == no_above == 1.
        :param prob_no_filters: probability that the individual has no_below
            and no_above set to 1
        :type prob_no_filters: float
        :return: the new individual
        :rtype: LdaIndividual
        :raise BoundsNotSetError: if the set_bounds method is not called first
        """
        if cls.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')

        no_below = 1
        no_above = 1.0
        if random.random() < 1 - prob_no_filters:
            no_below = random.randint(1, cls.max_no_below)
            no_above = random.uniform(cls.min_no_above, 1.0)

        topics_min = cls.topics_bounds[0]
        topic_max = cls.topics_bounds[1]
        return LdaIndividual(_topics=random.randint(topics_min, topic_max),
                             _alpha_val=random.uniform(MIN_ALPHA_VAL, 1.0),
                             _beta=random.random(),
                             _no_above=no_above,
                             _no_below=no_below,
                             _alpha_type=random.choices([0, 1, -1],
                                                        [0.6, 0.2, 0.2],
                                                        k=1)[0])

    @property
    def topics(self):
        return self._topics

    @topics.setter
    def topics(self, val):
        if self.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')
        self._topics = check_bounds(int(np.round(val)),
                                    self.topics_bounds[0],
                                    self.topics_bounds[1])

    @property
    def alpha_val(self):
        return self._alpha_val

    @alpha_val.setter
    def alpha_val(self, val):
        self._alpha_val = check_bounds(val, 0.0, float('inf'))

    @property
    def beta(self):
        return float(self._beta)

    @beta.setter
    def beta(self, val):
        self._beta = check_bounds(val, 0.0, float('inf'))

    @property
    def no_above(self):
        return float(self._no_above)

    @no_above.setter
    def no_above(self, val):
        if self.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')
        self._no_above = check_bounds(val, self.min_no_above, 1.0)

    @property
    def no_below(self):
        return int(self._no_below)

    @no_below.setter
    def no_below(self, val):
        if self.topics_bounds is None:
            raise BoundsNotSetError('set_bounds must be called first')
        self._no_below = check_bounds(int(np.round(val)), 1, self.max_no_below)

    @property
    def alpha_type(self):
        return self._alpha_type

    @alpha_type.setter
    def alpha_type(self, val):
        v = int(np.round(val))
        if v != 0:
            v = np.sign(v)
        self._alpha_type = v

    @property
    def alpha(self) -> Union[float, str]:
        if self._alpha_type == 0:
            return float(self._alpha_val)
        elif self._alpha_type > 0:
            return 'symmetric'
        else:
            return 'asymmetric'

    def __len__(self):
        return 6

    def __getitem__(self, item):
        return dataclasses.astuple(self, tuple_factory=list)[item]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            indexes = range(*key.indices(len(self)))
        elif isinstance(key, int):
            indexes = [key]
        else:
            msg = f'Invalid type for an index: {type(key).__name__!r}'
            raise TypeError(msg)

        if isinstance(value, (int, float)):
            value = [value]
        elif not isinstance(value, (tuple, list)):
            msg = f'Unsupported type for assignement: {type(value).__name__!r}'
            raise TypeError(msg)

        for val_index, i in enumerate(indexes):
            name = self.name_from_index(i)
            setattr(self, name, value[val_index])


def init_argparser():
    """Initialize the command line parser."""
    epilog = 'The script tests different lda models with different ' \
             'parameters and it tries to find the best model using a GA.'
    parser = ArgParse(description='Performs the LDA on a dataset', epilog=epilog)
    parser.add_argument('postproc_file', action='store', type=Path,
                        help='Path to the postprocess file with the text to '
                             'elaborate.', input=True)
    parser.add_argument('ga_params', action='store', type=Path,
                        help='path to the file with the parameters for the ga.')
    parser.add_argument('outdir', action='store', type=Path, nargs='?',
                        default=Path.cwd(), help='path to the directory where '
                                                 'to save the results.')
    parser.add_argument('--text-column', '-t', action='store', type=str,
                        default='abstract_filtered', dest='target_column',
                        help='Column in postproc_file to process. '
                             'Default: %(default)r.')
    parser.add_argument('--title-column', action='store', type=str,
                        default='title', dest='title',
                        help='Column in postproc_file to use as '
                             'document title. Default: %(default)r.')
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed used for training. Default %(default)r')
    parser.add_argument('--delimiter', action='store', type=str,
                        default='\t', help='Delimiter used in postproc_file. '
                                           'Default %(default)r')
    parser.add_argument('--no_timestamp', action='store_true',
                        help='if set, no timestamp is added to the '
                             'topics file names')
    parser.add_argument('--logfile', default='slr-kit.log',
                        help='log file name. If omitted %(default)r is used',
                        logfile=True)
    return parser


def init_train(corpora, titles, seed, modeldir, coherences):
    global _corpus, _titles, _seed, _modeldir, _coherences
    _corpus = corpora
    _titles = titles
    _seed = seed
    _modeldir = modeldir
    _coherences = coherences


# topics, alpha, beta, no_above, no_below label
def evaluate(ind: LdaIndividual):
    global _corpus, _titles, _seed, _modeldir, _coherences
    logger = logging.getLogger('debug_logger')
    u = str(uuid.uuid4())
    logger.debug(f'{u}: started evaluation')
    # unpack parameter
    n_topics = int(ind.topics)
    alpha = ind.alpha
    beta = ind.beta
    no_above = ind.no_above
    no_below = int(ind.no_below)
    result = {}
    result['uuid'] = u
    result['coherence'] = -float('inf')
    result['seed'] = _seed
    result['num_docs'] = len(_corpus)
    result['num_not_empty'] = 0
    result['topics'] = n_topics
    result['alpha'] = alpha
    result['beta'] = beta
    result['no_above'] = no_above
    result['no_below'] = no_below
    result['saved_model'] = False
    result['same_as'] = ''
    result['time'] = 0
    no_train = False
    # check the cache
    ind_hash = hash(ind)
    try:
        cv = (None, '')
        while cv[0] is None:
            time.sleep(1)
            cv = _coherences[ind_hash]

        result['coherence'] = cv[0]
        result['same_as'] = cv[1]
        no_train = True
    except KeyError:
        _coherences[ind_hash] = (None, '')
    start = timer()
    dictionary = Dictionary(_corpus)
    # Filter out words that occur less than no_above documents, or more than
    # no_below % of the documents.
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    try:
        _ = dictionary[0]  # This is only to "load" the dictionary.
    except KeyError:
        no_train = True

    output_dir = _modeldir / u
    output_dir.mkdir(exist_ok=True)
    if not no_train:
        not_empty_bows = []
        not_empty_docs = []
        not_empty_titles = []
        for i, c in enumerate(_corpus):
            bow = dictionary.doc2bow(c)
            if bow:
                not_empty_bows.append(bow)
                not_empty_docs.append(c)
                not_empty_titles.append(_titles[i])

        result['num_not_empty'] = len(not_empty_bows)
        model = LdaModel(not_empty_bows, num_topics=n_topics,
                         id2word=dictionary, chunksize=len(not_empty_bows),
                         passes=10, random_state=_seed,
                         minimum_probability=0.0, alpha=alpha, eta=beta)
        # computes coherence score for that model
        cv_model = CoherenceModel(model=model, texts=not_empty_docs,
                                  dictionary=dictionary, coherence='c_v',
                                  processes=1)
        result['coherence'] = cv_model.get_coherence()
        stop = timer()
        result['time'] = stop - start
        topics, docs_topics, _ = prepare_topics(model, not_empty_docs,
                                                not_empty_titles, dictionary)
        # check for NaNs
        for t in topics.values():
            if any(np.isnan(p) for p in t['terms_probability'].values()):
                result['coherence'] = -float('inf')

        if not np.isinf(result['coherence']):
            output_topics(topics, docs_topics, output_dir, 'lda',
                          result['uuid'])

        _coherences[ind_hash] = (result['coherence'], result['uuid'])

        model.save(str(output_dir / 'model'))
        dictionary.save(str(output_dir / 'model_dictionary'))
        result['saved_model'] = True
        logger.debug(f"{u}: evaluation completed")

    with open(output_dir / 'results.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    return (result['coherence'],)


def check_bounds(val, min_, max_):
    if val > max_:
        return max_
    elif val < min_:
        return min_
    else:
        return val


def load_ga_params(ga_params):
    """
    Loads the parameter used by the GA from a toml file

    :param ga_params: path to the toml file with the parameters
    :type ga_params: Path
    :return: the parameters dict
    :rtype: dict[str, Any]
    :raise ValueError: if some parameter have the wrong value. The error cause
        is stored in the exception arguments
    """
    try:
        with open(ga_params) as file:
            params = dict(tomlkit.loads(file.read()))
    except FileNotFoundError as err:
        msg = 'Error: file {!r} not found'
        sys.exit(msg.format(err.filename))

    default_params_file = Path(__file__).parent / 'ga_param.toml'
    with open(default_params_file) as file:
        defaults = dict(tomlkit.loads(file.read()))
    for sec in defaults.keys():
        if sec not in params:
            params[sec] = defaults[sec]
            continue
        for k, v in defaults[sec].items():
            if k not in params[sec]:
                params[sec][k] = v
            elif sec == 'mutate':
                if 'mu' not in params[sec][k]:
                    params[sec][k]['mu'] = v['mu']
                if 'sigma' not in params[sec][k]:
                    params[sec][k]['sigma'] = v['sigma']
    # fix types
    params_good = {
        'limits': {
            'min_no_above': float(params['limits']['min_no_above']),
            'max_no_below': float(params['limits']['max_no_below']),
            'min_topics': int(params['limits']['min_topics']),
            'max_topics': int(params['limits']['max_topics']),
        },
        'algorithm': {
            'mu': int(params['algorithm']['mu']),
            'lambda': int(params['algorithm']['lambda']),
            'initial':int(params['algorithm']['initial']),
            'generations':int(params['algorithm']['generations']),
            'tournament_size':int(params['algorithm']['tournament_size']),
        },
        'probabilities': {
            'mutate': float(params['probabilities']['mutate']),
            'component_mutation': float(params['probabilities']['component_mutation']),
            'mate': float(params['probabilities']['mate']),
            'no_filter': float(params['probabilities']['no_filter']),
        },
        'mutate': {}
    }
    for sec in params['mutate']:
        params_good['mutate'][sec] = {'mu': float(params['mutate'][sec]['mu']),
                                 'sigma': float(params['mutate'][sec]['sigma'])}

    params = params_good
    del file, defaults, default_params_file
    if params['limits']['min_topics'] > params['limits']['max_topics']:
        raise ValueError('limits.max_topics must be > limits.min_topics')

    if (params['probabilities']['no_filter'] < 0
            or params['probabilities']['no_filter'] > 1.0):
        raise ValueError('probabilities.no_filter must be a value between'
                         '0 and 1')

    if params['probabilities']['mate'] + params['probabilities']['mutate'] > 1:
        raise ValueError('The sum of the crossover and mutation probabilities '
                         'must be <= 1.0')

    return params


def collect_results(outdir):
    results = []
    for p in outdir.glob('*/results.csv'):
        results.append(pd.read_csv(p))
        p.unlink()

    df = pd.concat(results)
    df.sort_values(by=['saved_model', 'coherence'],
                   ascending=[False, False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def optimization(documents, titles, params, toolbox, args, model_dir):
    """
    Performs the optimization of the LDA model using the GA

    :param documents: corpus of documents to elaborate
    :type documents: list[list[str]]
    :param titles: titles of the documents
    :type titles: list[str]
    :param params: GA parameters
    :type params: dict[str, Any]
    :param toolbox: DEAP toolbox with all the operators set
    :type toolbox: base.Toolbox
    :param args: command line arguments
    :type args: argparse.Namespace
    :param model_dir: path to the directory where to save the models
    :type model_dir: Path
    """
    pop = toolbox.population(n=params['algorithm']['initial'])
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)
    print('Starting GA optimization')
    with Manager() as m:
        coherence_cache = m.dict()
        with Pool(processes=PHYSICAL_CPUS, initializer=init_train,
                  initargs=(documents, titles, args.seed,
                            model_dir, coherence_cache)) as pool:
            toolbox.register('map', pool.map)
            algorithms.eaMuPlusLambda(pop, toolbox,
                                      mu=params['algorithm']['mu'],
                                      lambda_=params['algorithm']['lambda'],
                                      cxpb=params['probabilities']['mate'],
                                      mutpb=params['probabilities']['mutate'],
                                      ngen=params['algorithm']['generations'],
                                      stats=stats, verbose=True)


def prepare_ga_toolbox(max_no_below, params):
    """
    Prepares the toolbox used by the GA

    :param max_no_below: maximum value for the no_below parameter
    :type max_no_below: int
    :param params: parameters of the GA
    :type params: dict[str, Any]
    :return: the initialized toolbox, with the 'mate', 'mutate', 'select',
        'individual' and 'population' attributes set
    :rtype: base.Toolbox
    """
    try:
        LdaIndividual.set_bounds(params['limits']['min_topics'],
                                 params['limits']['max_topics'],
                                 max_no_below, params['limits']['min_no_above'])
    except ValueError as e:
        sys.exit(e.args[0])
    creator.create('Individual', LdaIndividual, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register('individual', LdaIndividual.random_individual,
                     prob_no_filters=params['probabilities']['no_filter'])
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('mate', tools.cxTwoPoint)
    mut_mu = [0.0] * len(params['mutate'])
    mut_sigma = list(mut_mu)
    for f, v in params['mutate'].items():
        i = LdaIndividual.index_from_name(f)
        mut_mu[i] = float(v['mu'])
        mut_sigma[i] = float(v['sigma'])
    toolbox.register('mutate', tools.mutGaussian, mu=mut_mu, sigma=mut_sigma,
                     indpb=params['probabilities']['component_mutation'])
    toolbox.register('select', tools.selTournament,
                     tournsize=params['algorithm']['tournament_size'])
    toolbox.register('evaluate', evaluate)
    return toolbox


def lda_ga_optimization(args):
    global logger
    logger = setup_logger('debug_logger', args.logfile, level=logging.DEBUG)
    logger.info('==== lda_ga_optimization started ====')

    docs, titles = load_documents(args.postproc_file,
                                  args.target_column,
                                  args.title,
                                  args.delimiter)
    try:
        params = load_ga_params(args.ga_params)
    except ValueError as e:
        sys.exit(e.args[0])

    # ga preparation
    num_docs = len(titles)
    # set the bound used by LdaIndividual to check the topics and no_below values
    max_no_below = params['limits']['max_no_below']
    if max_no_below == -1:
        if num_docs >= 10:
            max_no_below = num_docs // 10
        else:
            max_no_below = 1
    elif max_no_below >= num_docs:
        sys.exit('max_no_below cannot be >= of the number of documents')

    if args.seed is not None:
        random.seed(args.seed)

    toolbox = prepare_ga_toolbox(max_no_below, params)

    estimated_trainings = (params['algorithm']['initial']
                           + (params['algorithm']['lambda']
                              * params['algorithm']['generations']))
    print('Estimated trainings:', estimated_trainings)
    logger.info(f'Estimated trainings: {estimated_trainings}')

    # prepare result directories
    now = datetime.now()
    result_dir = args.outdir / f'{now:%Y-%m-%d_%H%M%S}_lda_results'
    result_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(args.ga_params, result_dir / 'ga_params.toml')
    model_dir = result_dir / 'models'
    model_dir.mkdir(exist_ok=True)

    try:
        optimization(docs, titles, params, toolbox, args, model_dir)
    except KeyboardInterrupt:
        pass
    df = collect_results(model_dir)

    best = df.at[0, 'uuid']
    lda_path = model_dir / best
    if Path(lda_path / 'model').is_file():
        model = LdaModel.load(str(lda_path / 'model'))
        if Path(lda_path / 'model_dictionary').is_file():
            dictionary = Dictionary.load(str(lda_path / 'model_dictionary'))
            topics, docs_topics, _ = prepare_topics(model, docs,
                                                    titles, dictionary)
            output_topics(topics, docs_topics, args.outdir, 'lda', best,
                          use_timestamp=not args.no_timestamp)

    save_toml_files(args, df, result_dir)
    df.to_csv(result_dir / 'results.csv', sep='\t', index_label='id')
    with pd.option_context('display.width', 80,
                           'display.float_format', '{:,.3f}'.format):
        print(df)
    logger.info('==== lda_ga_optimization ended ====')


def main():
    args = init_argparser().parse_args()
    lda_ga_optimization(args)


if __name__ == '__main__':
    main()
