import tensorflow as tf

from config import MuZeroConfig, GameConfig, ReplayBufferConfig, MCTSConfig, NetworkConfig, TrainingConfig, ScalarConfig
from utils import KnownBounds
from environment import OpenAIEnvironment
from network import Network, binary_plane_encoder, scalar_to_support_model

# For type annotations
from muzero_types import Value


def make_config() -> MuZeroConfig:
    game_config = GameConfig(name='CartPole',
                             environment_class=OpenAIEnvironment,
                             environment_parameters={'gym_id': 'CartPole-v1'},
                             action_space_size=2,
                             num_players=1,
                             discount=0.99)

    replay_buffer_config = ReplayBufferConfig(window_size=int(1e3),
                                              prefetch_buffer_size=10)

    mcts_config = MCTSConfig(max_moves=500,
                             root_dirichlet_alpha=1.0,
                             root_exploration_fraction=0.25,
                             num_simulations=4,
                             temperature=1.0,
                             freezing_moves=50,
                             default_value=Value(50.0))

    network_config = NetworkConfig(network_class=CartPoleNetwork,
                                   regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   hidden_state_size=128,
                                   hidden_size=128,
                                   support_size=100)

    training_config = TrainingConfig(optimizer=tf.keras.optimizers.Adam(),
                                     batch_size=128,
                                     training_steps=int(2e5),
                                     checkpoint_interval=int(1e3),
                                     replay_buffer_loginterval=int(1e2),
                                     num_unroll_steps=2,
                                     td_steps=100,
                                     steps_per_execution=1)

    reward_config = ScalarConfig(known_bounds=KnownBounds(minv=Value(0.0), maxv=Value(1.0)),
                                 support_size=None,
                                 loss_decay=1.0)

    value_config = ScalarConfig(known_bounds=KnownBounds(minv=Value(0.0), maxv=Value(100.0)),
                                support_size=100,
                                loss_decay=0.1)

    return MuZeroConfig(game_config=game_config,
                        replay_buffer_config=replay_buffer_config,
                        mcts_config=mcts_config,
                        training_config=training_config,
                        network_config=network_config,
                        reward_config=reward_config,
                        value_config=value_config)


class CartPoleNetwork(Network):
    """
    Neural networks for cart-pole game.
    """

    def __init__(self,
                 config: MuZeroConfig,
                 regularizer: tf.keras.regularizers.Regularizer,
                 hidden_state_size: int,
                 hidden_size: int,
                 support_size: int
                 ) -> None:
        """
        Representation input (observation batch):       (batch_size, 4, support_size+1).
        Representation output (hidden state batch):     (batch_size, 1, hidden_state_size)

        Encoded action batch:                           (batch_size, 1+1, hidden_state_size)

        Dynamics input:                                 (batch_size, 2, hidden_state_size)
        Dynamics outputs:
            - hidden_state:                             (batch_size, 1, hidden_state_size)
            - reward:                                   (batch_size, )

        Prediction input:                               (batch_size, 1, hidden_state_size)
        Prediction outputs:
            - policy_logits:                            (batch_size, action_space_size=2)
            - value:                                    (batch_size, )
        """

        cartpole_state_preprocessing: tf.keras.Model = scalar_to_support_model(input_shape=(4,),
                                                                               scalar_min=tf.constant([-4.8, -5., -0.418, -5.]),
                                                                               scalar_max=tf.constant([4.8, 5., 0.418, 5.]),
                                                                               support_size=support_size)

        cartpole_representation: tf.keras.Model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=hidden_size, kernel_size=4, padding='valid', activation='relu',
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                   input_shape=(4, support_size+1)),
            tf.keras.layers.Dense(units=hidden_state_size, activation='relu',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)
        ], name=config.network_config.REPRESENTATION)

        encoded_state_action = tf.keras.Input(shape=(2, hidden_state_size))
        x = tf.keras.layers.Conv1D(filters=hidden_size, kernel_size=2, padding='valid', activation='relu',
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(encoded_state_action)
        hidden_state = tf.keras.layers.Dense(units=hidden_state_size, activation='relu',
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        x = tf.keras.layers.Flatten()(hidden_state)
        x = tf.keras.layers.Dense(units=hidden_size, activation='relu',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        reward_output = tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=regularizer,
                                              bias_regularizer=regularizer, name='reward')(x)
        cartpole_dynamics: tf.keras.Model = tf.keras.Model(inputs=encoded_state_action,
                                                           outputs=[hidden_state, reward_output],
                                                           name=config.network_config.DYNAMICS)

        hidden_state = tf.keras.Input(shape=(1, hidden_state_size))
        x = tf.keras.layers.Flatten()(hidden_state)
        x = tf.keras.layers.Dense(units=hidden_size, activation='relu',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        x = tf.keras.layers.Dense(units=hidden_size, activation='relu',
                                  kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        value_output = tf.keras.layers.Dense(units=support_size+1, activation='softmax', name='value',
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        policy_logits_output = tf.keras.layers.Dense(units=2, name='policy_logits',
                                                     kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

        cartpole_prediction: tf.keras.Model = tf.keras.Model(inputs=hidden_state,
                                                             outputs=[value_output, policy_logits_output],
                                                             name=config.network_config.PREDICTION)

        super().__init__(config=config,
                         representation=cartpole_representation,
                         dynamics=cartpole_dynamics,
                         prediction=cartpole_prediction,
                         state_action_encoder=binary_plane_encoder(state_shape=(1, hidden_state_size), axis=0),
                         state_preprocessing=cartpole_state_preprocessing)
