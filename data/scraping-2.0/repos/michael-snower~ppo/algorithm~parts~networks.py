# analytical tools
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

class NetworkBuilder(object):
    '''
    Creates PPO networks from command line arguments. Orthogonal weight initialization is
    often used because this is done by OpenAI Baselines. Also, in the case of recurrent nets,
    orthogonal weight initialization is suggested by
    Henaff, Szlam, and LeCun.
    https://pdfs.semanticscholar.org/f824/04ab4b6e789ed4faa15433d18c7bca0bd698.pdf
    '''
    def __init__(self):
        network_tools = NetworkTools()
        self._fc = network_tools.fc
        self._conv2d = network_tools.conv2d
        self._batch2seq = network_tools.batch2seq
        self._seq2batch = network_tools.seq2batch
        self._lstm_cell = network_tools.lstm_cell

    def cnn(self, X):
        '''
        Convolutional neural net with the Architecture described in Deepmind's
        "Human-level control through deep reinforcement learning"
        Call with 'cnn'
        '''
        X = tf.cast(X, tf.float32) / 255.
        h1 = self._conv2d(X, filters=32, kernel=8, activ='relu', stride=4,
                          wi='ortho', bi=0.0, wi_scale=np.sqrt(2))
        h2 = self._conv2d(h1, filters=64, kernel=4, activ='relu', stride=2,
                          wi='ortho', bi=0.0, wi_scale=np.sqrt(2))
        h3 = self._conv2d(h2, filters=64, kernel=3, activ='relu', stride=1,
                          wi='ortho', bi=0.0, wi_scale=np.sqrt(2))
        h3_flat = layers.flatten(h3)
        h4 = self._fc(h3_flat, size=512, activ='relu', wi='ortho', bi=0.0,
                      wi_scale=np.sqrt(2))
        return h4

    def cnn_lstm(self, X, steps, lstm_size=128):
        '''
        Adds an LSTM cell to the cnn above.
        Call with 'cnn_lstm'
        '''
        # feed forward through conv and fully connected layers
        rec_masks = tf.placeholder(tf.float32, [steps], name='pl_rm')
        rec_state = tf.placeholder(tf.float32, [1, lstm_size * 2], name='pl_rst')
        X = tf.cast(X, tf.float32) / 255.
        h1 = self._conv2d(X, filters=32, kernel=8, activ='relu', stride=4,
                          wi='ortho', bi=0.0, wi_scale=np.sqrt(2))
        h2 = self._conv2d(h1, filters=64, kernel=4, activ='relu', stride=2,
                          wi='ortho', bi=0.0, wi_scale=np.sqrt(2))
        h3 = self._conv2d(h2, filters=64, kernel=3, activ='relu', stride=1,
                          wi='ortho', bi=0.0, wi_scale=np.sqrt(2))
        h3_flat = layers.flatten(h3)
        h4 = self._fc(h3_flat, size=512, activ='relu', wi='ortho', bi=0.0,
                      wi_scale=np.sqrt(2))
        # shape data and pass through lstm cell
        xs = self._batch2seq(h4, steps)
        ms = self._batch2seq(rec_masks, steps)
        ms = [tf.expand_dims(m, 1) for m in ms]
        lstm_out, new_rec_state = self._lstm_cell(xs, ms, rec_state, lstm_size)
        h5 = self._seq2batch(lstm_out)
        return h5, {'rec_masks':rec_masks, 'rec_state':rec_state,
                   'cur_rec_state':new_rec_state}

    def fc2_small(self, X, size=64):
        '''
        Two fc layers.
        Call with 'fc2_small'. 
        '''
        h1 = self._fc(X, size=size, activ='relu', wi='ortho', bi=0.0,
                      wi_scale=np.sqrt(2))
        h2 = self._fc(h1, size=size, activ='relu', wi='ortho', bi=0.0,
                      wi_scale=np.sqrt(2))
        return h2

    def fc3(self, X, size=256):
        '''
        Two fc layers.
        Call with 'fc3'. 
        '''
        h1 = self._fc(X, size=size, activ='relu', wi='ortho', bi=0.0,
                      wi_scale=np.sqrt(2))
        h2 = self._fc(h1, size=size, activ='relu', wi='ortho', bi=0.0,
                      wi_scale=np.sqrt(2))
        h3 = self._fc(h2, size=size//2, activ='relu', wi='ortho', bi=0.0,
                      wi_scale=np.sqrt(2))
        return h3

    def actor_fc1(self, X, num_actions):
        '''
        Actor with 1 fully connected layer.
        Call with 'actor_fc1'
        '''
        h1 = self._fc(X, size=num_actions, activ=None, wi='ortho', bi=0.0,
                      wi_scale=0.01)
        return h1

    def critic_fc1(self, X):
        '''
        Critic (value function) with 1 fully connected layer.
        Call with 'critic_fc1'
        '''
        h1 = self._fc(X, size=1, activ=None, wi='ortho', bi=0.0,
                      wi_scale=1.0)
        return tf.squeeze(h1)

class NetworkTools(object):
    '''
    Tools for building deep networks.
    '''
    def fc(self, X, size, activ, wi, bi, wi_scale=None, scope=None):
        '''
        Builds a fully connected layer.
        '''
        return tf.contrib.layers.fully_connected(
            inputs=X,
            num_outputs=size,
            activation_fn=self._get_activ(activ),
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=self._get_wi(wi, wi_scale),
            weights_regularizer=None,
            biases_initializer=tf.constant_initializer(bi),
            biases_regularizer=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            scope=scope)

    def conv2d(self, X, filters, kernel, activ, stride, wi, bi, wi_scale=None,
               scope=None):
        '''
        Builds a 2d convolution layer.
        '''
        return tf.contrib.layers.conv2d(
            inputs=X,
            num_outputs=filters,
            kernel_size=kernel,
            stride=stride,
            padding='SAME',
            data_format=None,
            rate=1,
            activation_fn=self._get_activ(activ),
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=self._get_wi(wi, wi_scale),
            weights_regularizer=None,
            biases_initializer=tf.constant_initializer(bi),
            biases_regularizer=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            scope=scope)

    def _get_activ(self, a):
        if a == 'relu':
            return tf.nn.relu
        elif a == 'leaky_relu':
            return tf.nn.leaky_relu
        elif a == None:
            return None
        else:
            raise NotImplementedError

    def _get_wi(self, wi, wi_scale):
        if wi == 'xavier':
            return tf.initializers.xavier_initializer()
        elif wi == 'vs':
            return tf.variance_scaling_initializer()
        elif wi == 'ortho':
            return self._ortho_init(wi_scale)
        else:
            raise NotImplementedError

    def batch2seq(self, batch, steps):
        '''
        Converts a batch to a sequence.
        '''
        seq = tf.dynamic_partition(data=batch, partitions=tf.range(steps),
                                   num_partitions=steps)
        return seq

    def seq2batch(self, h, flat=False):
        '''
        Converts a sequence to a batch. From OpenAI Baselines.
        https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
        '''
        shape = h[0].get_shape().as_list()
        if not flat:
            assert(len(shape) > 1)
            nh = h[0].get_shape()[-1].value
            return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
        else:
            return tf.reshape(tf.stack(values=h, axis=1), [-1])

    def lstm_cell(self, xs, ms, s, nh):
        '''
        Creates an LSTM cell that takes "done masks" as an input in addition to the game 
        states (observations) and lstm state. Assigns weights to these dones to make it
        easier for the network to differentiate between states from games.
        From OpenAI Baselines : https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
        '''
        _, nx = [v.value for v in xs[0].get_shape()]
        with tf.variable_scope('lstm'):
            wx = tf.get_variable('wx', [nx, nh*4], initializer=self._ortho_init(1.0))
            wh = tf.get_variable('wh', [nh, nh*4], initializer=self._ortho_init(1.0))
            b = tf.get_variable('b', [nh*4], initializer=tf.constant_initializer(0.0))

            c, h = tf.split(axis=1, num_or_size_splits=2, value=s)

            for idx, (x, m) in enumerate(zip(xs, ms)):
                c = c*(1-m)
                h = h*(1-m)
                z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
                i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                i = tf.nn.sigmoid(i)
                f = tf.nn.sigmoid(f)
                o = tf.nn.sigmoid(o)
                u = tf.tanh(u)
                c = f*c + i*u
                h = o*tf.tanh(c)
                xs[idx] = h
            s = tf.concat(axis=1, values=[c, h])
            return xs, s

    def _ortho_init(self, scale):
        '''
        From OpenAI Baselines.
        https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
        '''
        def _weight_fn(shape, dtype=None, partition_info=None):
            #lasagne ortho init for tf
            shape = tuple(shape)
            if len(shape) == 2:
                flat_shape = shape
            elif len(shape) == 4: # assumes NHWC
                flat_shape = (np.prod(shape[:-1]), shape[-1])
            else:
                raise NotImplementedError
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v # pick the one with the correct shape
            q = q.reshape(shape)
            return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
        return _weight_fn
