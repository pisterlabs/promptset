import os
import sys
from configparser import ConfigParser
import pytest
from functools import reduce
from collections import Counter, OrderedDict

from topic_modeling_toolkit.patm import CoherenceFilesBuilder, TrainerFactory, Experiment, Tuner, PipeHandler, political_spectrum as political_spectrum_manager
# import topic_modeling_toolkit as tmtk
#
# from tmtk.patm import CoherenceFilesBuilder, TrainerFactory, Experiment, Tuner, PipeHandler
#
# from tmtk.patm import political_spectrum as political_spectrum_manager

from topic_modeling_toolkit.processors import Pipeline
# from tmtk.processors import Pipeline
from topic_modeling_toolkit.reporting import GraphMaker, TopicsHandler, DatasetReporter, ResultsHandler
from topic_modeling_toolkit.results import ExperimentalResults


####################3
MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(MODULE_DIR, 'data')

TRAIN_CFG = os.path.join(MODULE_DIR, 'test-train.cfg')
REGS_CFG = os.path.join(MODULE_DIR, 'test-regularizers.cfg')


TEST_COLLECTIONS_ROOT_DIR_NAME = 'unittests-collections'

TEST_COLLECTION = 'unittest-dataset'
MODEL_1_LABEL = 'test-model'
TUNE_LABEL_PREFIX = 'unittest'
#####################


@pytest.fixture(scope='session')
def unittests_data_dir():
    return DATA_DIR


@pytest.fixture(scope='session')
def collections_root_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp(TEST_COLLECTIONS_ROOT_DIR_NAME))

@pytest.fixture(scope='session')
def test_collection_name():
    return TEST_COLLECTION

#
@pytest.fixture(scope='session')
def rq1_cplsa_results_json(unittests_data_dir):
    """These are the results gathered for a cplsa trained model"""
    return os.path.join(unittests_data_dir, 'cplsa100000_0.2_0.json')


@pytest.fixture(scope='session')
def test_collection_dir(collections_root_dir, test_collection_name, tmpdir_factory):
    if not os.path.isdir(os.path.join(collections_root_dir, test_collection_name)):
        os.mkdir(os.path.join(collections_root_dir, test_collection_name))
    return os.path.join(collections_root_dir, test_collection_name)
    # return str(tmpdir_factory.mktemp(os.path.join(collections_root_dir, test_collection_name)))
    # return os.path.join(collections_root_dir, TEST_COLLECTION)

@pytest.fixture(scope='session')
def results_handler(collections_root_dir):
    return ResultsHandler(collections_root_dir, results_dir_name='results')


@pytest.fixture(scope='session')
def pairs_file_nb_lines():  # number of lines in cooc and ppmi files. Dirty code to support python 2 backwards compatibility
    python3 = {True: [1215, 1347],  # second value is expected in travis build with python3
               False: [1255]}
    return python3[2 < sys.version_info[0]]

@pytest.fixture(scope='session')
def pipe_n_quantities(test_collection_dir, pairs_file_nb_lines):
    return {'unittest-pipeline-cfg': os.path.join(MODULE_DIR, 'test-pipeline.cfg'),
            'unittest-collection-dir': test_collection_dir,
            'category': 'posts',
            'sample': 200,
            'resulting-nb-docs': 200,
            'nb-bows': 2765,
            'word-vocabulary-length': 1347,
            'nb-all-modalities-terms': 1348,  # corresponds to the number of lines in the vocabulary file created (must call persist of PipeHandler with add_class_labels_to_vocab=True, which is the default).
            # the above probably will fail in case no second modality is used (only the @default_class is enabled)
            'nb-lines-cooc-n-ppmi-files': pairs_file_nb_lines
            }

@pytest.fixture(scope='session')
def political_spectrum():
    return political_spectrum_manager


#### OPERATIONS ARTIFACTS
@pytest.fixture(scope='session')
def preprocess_phase(pipe_n_quantities):
    pipe_handler = PipeHandler()
    pipe_handler.process(pipe_n_quantities['unittest-pipeline-cfg'], pipe_n_quantities['category'], sample=pipe_n_quantities['sample'])
    return pipe_handler

@pytest.fixture(scope='session')
def test_dataset(preprocess_phase, political_spectrum, test_collection_dir):
    """A dataset ready to be used for topic modeling training. Depends on the input document sample size to take and resulting actual size"""
    text_dataset = preprocess_phase.persist(test_collection_dir, political_spectrum.poster_id2ideology_label, political_spectrum.class_names, add_class_labels_to_vocab=True)
    coh_builder = CoherenceFilesBuilder(test_collection_dir)
    coh_builder.create_files(cooc_window=10, min_tf=0, min_df=0, apply_zero_index=False)
    return text_dataset

# PARSE UNITTEST CFG FILES

def parse_cfg(cfg):
    config = ConfigParser()
    config.read(cfg)
    return {section: dict(config.items(section)) for section in config.sections()}


@pytest.fixture(scope='session')
def train_settings():
    """These settings (learning, reg components, score components, etc) are used to train the model in 'trained_model' fixture. A dictionary of cfg sections mapping to dictionaries with settings names-values pairs."""
    _ = parse_cfg(TRAIN_CFG)
    _['regularizers'] = {k: v for k, v in _['regularizers'].items() if v}
    _['scores'] = {k: v for k, v in _['scores'].items() if v}
    return _


@pytest.fixture(scope='session')
def trainer(collections_root_dir, test_dataset):
    return TrainerFactory().create_trainer(os.path.join(collections_root_dir, test_dataset.name), exploit_ideology_labels=True, force_new_batches=True)


@pytest.fixture(scope='session')
def trained_model_n_experiment(collections_root_dir, test_dataset, trainer):
    experiment = Experiment(os.path.join(collections_root_dir, test_dataset.name))
    topic_model = trainer.model_factory.create_model(MODEL_1_LABEL, TRAIN_CFG, reg_cfg=REGS_CFG, show_progress_bars=False)
    train_specs = trainer.model_factory.create_train_specs()
    trainer.register(experiment)
    experiment.init_empty_trackables(topic_model)
    trainer.train(topic_model, train_specs, effects=False, cache_theta=True)
    experiment.save_experiment(save_phi=True)
    return topic_model, experiment


@pytest.fixture(scope='session')
def loaded_model_n_experiment(collections_root_dir, test_dataset, trainer, trained_model_n_experiment):
    model, experiment = trained_model_n_experiment
    experiment.save_experiment(save_phi=True)
    new_exp_obj = Experiment(os.path.join(collections_root_dir, test_dataset.name))
    trainer.register(new_exp_obj)
    loaded_model = new_exp_obj.load_experiment(model.label)
    return loaded_model, new_exp_obj


@pytest.fixture(scope='session')
def training_params():
    return [
        ('nb-topics', [10, 12]),
        ('collection-passes', 4),
        ('document-passes', 1),
        ('background-topics-pct', 0.2),
        ('ideology-class-weight', 1),
        ('default-class-weight', 1)
        ]


@pytest.fixture(scope='session')
def expected_explorable_params(training_params, regularizers_specs):
    return [(k, v) for k, v in training_params if type(v) == list and len(v) != 1] + [('{}.{}'.format(k, param), value) for k, v in regularizers_specs for param, value in v if type(value) == list and len(value) != 1]

@pytest.fixture(scope='session')
def expected_constant_params(training_params, regularizers_specs):
    return [(k, v) for k, v in training_params if type(v) != list or len(v) == 1] + [('{}.{}'.format(k, param), value) for k, v in regularizers_specs for param, value in v if type(value) != list or len(value) == 1]

@pytest.fixture(scope='session')
def regularizers_specs():
    return [
        ('label-regularization-phi-dom-cls', [('tau', 1e5)]),
        ('decorrelate-phi-dom-def', [('tau', 1e4)])
    ]


@pytest.fixture(scope='session')
def tuning_parameters():
    return dict(prefix_label=TUNE_LABEL_PREFIX,
                # append_explorables=True,
                # append_static=True,
                force_overwrite=True,
                cache_theta=True, verbose=False, interactive=False,
                labeling_params=['nb-topics', 'background-topics-pct', 'collection-passes', 'document-passes', 'ideology-class-weight'],
                preserve_order=False,
                # parameter_set='training|regularization'
                )

@pytest.fixture(scope='session')
def model_names(tuning_parameters, training_params, expected_labeling_parameters):
    """alphabetically sorted expected model names to persist (phi and results)"""
    def _mock_label(labeling, params_data):
        inds = Counter()
        params_data = OrderedDict(params_data)
        for l in labeling:
            if type(params_data[l]) != list or len(params_data[l]) == 1:
                yield params_data[l]
            else:
                inds[l] += 1
                yield params_data[l][inds[l] - 1]
    nb_models = reduce(lambda k,l: k*l, [len(span) if type(span) == list else 1 for _, span in training_params])
    prefix = ''
    if tuning_parameters['prefix_label']:
        prefix = tuning_parameters['prefix_label'] + '_'
    return sorted([prefix + '_'.join(str(x) for x in _mock_label(expected_labeling_parameters, training_params)) for _ in range(nb_models)])


@pytest.fixture(scope='session')
def expected_labeling_parameters(tuning_parameters, training_params, expected_constant_params, expected_explorable_params):
    static_flag = {True: [x[0] for x in expected_constant_params],
                   False: []}
    explorable_flag = {True: [x[0] for x in expected_explorable_params],
                       False: []}
    if tuning_parameters['labeling_params']:
        labeling_params = tuning_parameters['labeling_params']
    else:
        labeling_params = static_flag[tuning_parameters['append_static']] + explorable_flag[tuning_parameters['append_explorables']]
    if tuning_parameters['preserve_order']:
        return [x for x in training_params if x in labeling_params]
    return labeling_params


@pytest.fixture(scope='session')
def tuner_obj(collections_root_dir, test_dataset, training_params, regularizers_specs, tuning_parameters):
    tuner = Tuner(os.path.join(collections_root_dir, test_dataset.name), {
        'perplexity': 'per',
        'sparsity-phi-@dc': 'sppd',
        'sparsity-theta': 'spt',
        'topic-kernel-0.60': 'tk60',
        'topic-kernel-0.80': 'tk80',
        'top-tokens-10': 'top10',
        'top-tokens-100': 'top100',
        'background-tokens-ratio-0.3': 'btr3',
        'background-tokens-ratio-0.2': 'btr2'
    })

    tuner.training_parameters = training_params
    tuner.regularization_specs = regularizers_specs
    tuner.tune(**tuning_parameters)
    return tuner


@pytest.fixture(scope='session')
def dataset_reporter(tuner_obj):
    return DatasetReporter(os.path.dirname(tuner_obj.dataset))


@pytest.fixture(scope='session')
def graphs_parameters():
    return {'selection': 3,
            'metric': 'alphabetical',
            'score_definitions': ['background-tokens-ratio-0.30', 'kernel-coherence-0.80', 'sparsity-theta', 'top-tokens-coherence-10'],
            'tau_trajectories': '',
            # 'tau_trajectories': 'all',
            }


@pytest.fixture(scope='session')
def graphs(exp_res_obj1, trained_model_n_experiment, tuner_obj, graphs_parameters):
    graph_maker = GraphMaker(os.path.dirname(tuner_obj.dataset))
    sparser_regularizers_tau_coefficients_trajectories = False
    selection = graphs_parameters.pop('selection')
    graph_maker.build_graphs_from_collection(os.path.basename(tuner_obj.dataset), selection,  # use a maximal number of 8 models to compare together
                                             **dict({'save': True, 'nb_points': None, 'verbose': False}, **graphs_parameters))
    graphs_parameters['selection'] = selection
    return graph_maker.saved_figures


############################################
@pytest.fixture(scope='session')
def json_path(collections_root_dir, test_collection_name):
    return os.path.join(collections_root_dir, test_collection_name, 'results', 'toy-exp-res.json')


@pytest.fixture(scope='session')
def kernel_data_0():
    return [
        [[1, 2], [3, 4], [5, 6], [120, 100]],
        {'t01': {'coherence': [1, 2, 3],
                 'contrast': [6, 3],
                 'purity': [1, 8]},
         't00': {'coherence': [10, 2, 3],
                 'contrast': [67, 36],
                 'purity': [12, 89]},
         't02': {'coherence': [10, 11],
                 'contrast': [656, 32],
                 'purity': [17, 856]}}
    ]


@pytest.fixture(scope='session')
def kernel_data_1():
    return [[[10,20], [30,40], [50,6], [80, 90]], {'t01': {'coherence': [3, 9],
                                                             'contrast': [96, 3],
                                                             'purity': [1, 98]},
                                                     't00': {'coherence': [19,2,93],
                                                             'contrast': [7, 3],
                                                             'purity': [2, 89]},
                                                     't02': {'coherence': [0,11],
                                                             'contrast': [66, 32],
                                                             'purity': [17, 85]}
                                                     }]


@pytest.fixture(scope='session')
def exp_res_obj1(kernel_data_0, kernel_data_1, json_path, test_collection_dir):

    exp = ExperimentalResults.from_dict({
        'scalars': {
            'dir': 'a-dataset-dir',
            'label': 'toy-exp-res',
            'dataset_iterations': 3,  # LEGACY '_' (underscore) usage
            'nb_topics': 5,  # LEGACY '_' (underscore) usage
            'document_passes': 2,  # LEGACY '_' (underscore) usage
            'background_topics': ['t0', 't1'],  # LEGACY '_' (underscore) usage
            'domain_topics': ['t2', 't3', 't4'],  # LEGACY '_' (underscore) usage
            'modalities': {'dcn': 1, 'icn': 5}
        },
        'tracked': {
            'perplexity': [1, 2, 3],
            'sparsity-phi-@dc': [-2, -4, -6],
            'sparsity-phi-@ic': [-56, -12, -32],
            'sparsity-theta': [2, 4, 6],
            'background-tokens-ratio-0.3': [0.4, 0.3, 0.2],
            'topic-kernel': {
                '0.60': {
                    'avg_coh': kernel_data_0[0][0],
                    'avg_con': kernel_data_0[0][1],
                    'avg_pur': kernel_data_0[0][2],
                    'size': kernel_data_0[0][3],
                    'topics': kernel_data_0[1]
                },
                '0.80': {
                    'avg_coh': kernel_data_1[0][0],
                    'avg_con': kernel_data_1[0][1],
                    'avg_pur': kernel_data_1[0][2],
                    'size': kernel_data_1[0][3],
                    'topics': kernel_data_1[1]
                }
            },
            'top-tokens': {
                '10': {
                    'avg_coh': [5, 6, 7],
                    'topics': {'t01': [12, 22, 3], 't00': [10, 2, 3], 't02': [10, 11]}
                },
                '100': {
                    'avg_coh': [10, 20, 30],
                    'topics': {'t01': [5, 7, 9], 't00': [12, 32, 3], 't02': [11, 1]}
                }
            },
            'tau-trajectories': {'phi': [1, 2, 3], 'theta': [5, 6, 7]},
            'regularization-dynamic-parameters': {'type-a': {'tau': [1, 2, 3]},
                                                  'type-b': {'tau': [-1, -1, -2], 'alpha': [1, 1.2]}},
            'collection-passes': [3]
        },
        'final': {
            'topic-kernel': {
                '0.60': {'t00': ['a', 'b', 'c'],
                         't01': ['d', 'e', 'f'],
                         't02': ['g', 'h', 'i']},
                '0.80': {'t00': ['j', 'k', 'l'],
                         't01': ['m', 'n', 'o'],
                         't02': ['p', 'q', 'r']}
            },
            'top-tokens': {
                '10': {
                    't00': ['s', 't', 'u'],
                    't01': ['v', 'x', 'y'],
                    't02': ['z', 'a1', 'b1']
                },
                '100': {
                    't00': ['c1', 'd1', 'e1'],
                    't01': ['f1', 'g1', 'h1'],
                    't02': ['i1', 'j1', 'k1']
                }
            },
            'background-tokens': ['l1', 'm1', 'n1']
        },
        'regularizers': ['reg1_params_pformat', 'reg2_params_pformat'],
        'reg_defs': {'type-a': 'reg1', 'type-b': 'reg2'},
        'score_defs': {'perplexity': 'prl', 'top-tokens-10': 'top10'}
    })

    if not os.path.isdir(os.path.join(test_collection_dir, 'results')):
        os.mkdir(os.path.join(test_collection_dir, 'results'))
    exp.save_as_json(json_path)
    return exp
