from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.envs.gym import GymEnv
from core.envs.atari_ram import AtariRamEnv
from core.envs.atari import AtariEnv
from core.envs.lab import LabEnv
from core.minisim.minisim_env import MinisimEnv

EnvDict = {"gym":       GymEnv,                 # classic control games from openai w/ low-level   input
           "atari-ram": AtariRamEnv,            # atari integrations from openai, with low-level   input
           "atari":     AtariEnv,               # atari integrations from openai, with pixel-level input
           "lab":       LabEnv,
           "minisim":   MinisimEnv}

from core.models.empty import EmptyModel
from core.models.dqn_mlp import DQNMlpModel
from core.models.dqn_cnn import DQNCnnModel
from core.models.a3c_mlp_con import A3CMlpConModel
from core.models.a3c_cnn_dis import A3CCnnDisModel
from core.models.acer_mlp_dis import ACERMlpDisModel
from core.models.acer_cnn_dis import ACERCnnDisModel
from core.minisim.models.mini_narrowing import A3CMlpNarrowingMinisimModel
from core.minisim.models.mini_two_lstm_separate import A3CMlpDeeperSeparateHiddenMinisimModel
from core.minisim.models.mini_two_lstm_separate_target_to_lstm import A3CMlpDeeper2MinisimModel
from core.minisim.models.mini_two_lstm_shared import A3CMlpDeeperMinisimModel
from core.minisim.models.mini_two_lstm_separate_two_levels import A3CMlpDeeperSeparateHiddenTwoLevelsMinisimModel
from core.minisim.models.mini_wide import A3CMlpMinisimModel
from core.minisim.models.mini_no_lstm import A3CMlpNoLSTMMinisimModel
from core.minisim.models.mini_conv import A3CCnvMinisimModel
from core.minisim.models.mini_target_only import A3CTargetOnlyMinisimModel
from core.minisim.models.mini_conv_lstm import A3CCnvLSTMMinisimModel
ModelDict = {"empty":        EmptyModel,  # contains nothing, only should be used w/ EmptyAgent
             "dqn-mlp":      DQNMlpModel,  # for dqn low-level    input
             "dqn-cnn":      DQNCnnModel,  # for dqn pixel-level  input
             "a3c-mlp-con":  A3CMlpConModel,  # for a3c low-level    input (NOTE: continuous must end in "-con")
             "a3c-cnn-dis":  A3CCnnDisModel,  # for a3c pixel-level  input
             "acer-mlp-dis": ACERMlpDisModel,  # for acer low-level   input
             "acer-cnn-dis": ACERCnnDisModel,  # for acer pixel-level input
             "none":         None,
             "a3c-mlp-minisim": A3CMlpMinisimModel,
             "a3c-mlp-minisim-narrowing": A3CMlpNarrowingMinisimModel,
             "a3c-mlp-deeper": A3CMlpDeeperMinisimModel,
             "a3c-mlp-deeper2": A3CMlpDeeper2MinisimModel,
             "a3c-mlp-deeper-sep-hid": A3CMlpDeeperSeparateHiddenMinisimModel,
             "a3c-mlp-deeper-sep-hid-two": A3CMlpDeeperSeparateHiddenTwoLevelsMinisimModel,
             "a3c-mlp-no-lstm": A3CMlpNoLSTMMinisimModel,
             "a3c-cnv": A3CCnvMinisimModel,
             "a3c-target-only": A3CTargetOnlyMinisimModel,
             "a3c-cnv-lstm": A3CCnvLSTMMinisimModel}

from core.memories.sequential import SequentialMemory
from core.memories.episode_parameter import EpisodeParameterMemory
from core.memories.episodic import EpisodicMemory
MemoryDict = {"sequential":        SequentialMemory,        # off-policy
              "episode-parameter": EpisodeParameterMemory,  # not in use right now
              "episodic":          EpisodicMemory,          # on/off-policy
              "none":              None}                    # on-policy

from core.agents.empty import EmptyAgent
from core.agents.dqn   import DQNAgent
from core.agents.a3c   import A3CAgent
from core.agents.acer  import ACERAgent
AgentDict = {"empty": EmptyAgent,               # to test integration of new envs, contains only the most basic control loop
             "dqn":   DQNAgent,                 # dqn  (w/ double dqn & dueling as options)
             "a3c":   A3CAgent,                 # a3c  (multi-process, pure cpu version)
             "acer":  ACERAgent}                # acer (multi-process, pure cpu version)