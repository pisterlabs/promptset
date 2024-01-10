""" Space class - pieces of this code are borrowed from openai gym repository. """
from __future__ import absolute_import
from collections import OrderedDict
import numpy as np

from bc_gym_planning_env.envs.base.action import Action
from bc_gym_planning_env.utilities.serialize import Serializable

SPACE_LOCAL_RANDOM_STATE = np.random.RandomState()
SPACE_LOCAL_RANDOM_STATE.seed(0)


class Space(Serializable):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """
    VERSION = 1

    def __init__(self, shape=None, dtype=None):
        """
        Initialize Space
        :param shape: shape of the space
        :param dtype: dtype of the space
        """
        self.shape = None if shape is None else tuple(shape)
        self.dtype = dtype

    @staticmethod
    def _get_type_mapping():
        """ Get a dict mapping from name of space to type of the space.
        :return Dict[str, type]: a dict mapping from name of space to type of the space.
        """
        return {
            Box.SPACE_NAME: Box,
            Dict.SPACE_NAME: Dict,
            Discrete.SPACE_NAME: Discrete
        }

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        :param x: anything - is it in the space?
        """
        raise NotImplementedError

    __contains__ = contains

    @staticmethod
    def to_jsonable(sample_n):
        """
        Convert a batch of samples from this space to a JSONable data type.
         By default, assume identity is JSONable

        :param sample_n object: actually sort of whatever
        :return: also whatever
        """
        return sample_n

    @staticmethod
    def from_jsonable(sample_n):
        """
        Convert a JSONable data type to a batch of samples from this space.
        By default, assume identity is JSONable

        :param sample_n object: actually sort of whatever
        :return: also whatever
        """
        return sample_n

    def serialize(self):
        raise NotImplementedError("Should be implemented by children action spaces")

    @classmethod
    def deserialize(cls, state):
        name = state.pop()
        subclass = cls._get_type_mapping()[name]
        return subclass(**state)


class Box(Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.

    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    SPACE_NAME = 'Box'

    def __init__(self, low=None, high=None, shape=None, dtype=None):
        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape

        :param low:  lower bounds
        :param high:  higher bounds
        :param shape: shape of the box space
        :param dtype: dtype of the space
        """
        if shape is None:
            assert low.shape == high.shape
            shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            low = low + np.zeros(shape)
            high = high + np.zeros(shape)
        if dtype is None:  # Autodetect type
            if (high == 255).all():
                dtype = 'uint8'
            else:
                dtype = 'float32'
        self.low = low.astype(dtype)
        self.high = high.astype(dtype)
        Space.__init__(self, shape, dtype)

    def serialize(self):
        return dict(
            name=Box.SPACE_NAME,
            low=self.low,
            high=self.high,
            shape=self.shape,
            dtype=self.dtype
        )

    def sample(self):
        v, w = SPACE_LOCAL_RANDOM_STATE.uniform(
            low=self.low,
            high=self.high + (0 if np.dtype(self.dtype).kind == 'f' else 1),
            size=self.low.shape
        ).astype(self.dtype)

        return Action(command=np.array([v, w]))

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    __contains__ = contains

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


class Dict(Space):
    """
    A dictionary of simpler spaces.
    Example usage:
    self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
    Example usage [nested]:
    self.nested_observation_space = spaces.Dict({
        'sensors':  spaces.Dict({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete([ [0,4], [0,1], [0,1] ]),
        'inner_state':spaces.Dict({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.Dict({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """
    SPACE_NAME = 'Dict'

    def __init__(self, spaces=None, **spaces_kwargs):
        """ Initialize Dict space
        :param spaces: input spaces
        :param spaces_kwargs: kwargs that are going to be passed to subspaces, see below
        """
        assert (spaces is None) or (not spaces_kwargs), 'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces = spaces
        Space.__init__(self, None, None)    # None for shape and dtype, since it'll require special handling

    def sample(self):
        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def serialize(self):
        raise NotImplementedError

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    __contains__ = contains

    def __repr__(self):
        return "Dict(" + ", ". join([k + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as dict-repr of vectors
        return {key: space.to_jsonable([sample[key] for sample in sample_n])
                for key, space in self.spaces.items()}

    def from_jsonable(self, sample_n):
        dict_of_list = {}
        key = None
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        if key is not None:
            for i, _ in enumerate(dict_of_list[key]):
                entry = {}
                for key, value in dict_of_list.items():
                    entry[key] = value[i]
                ret.append(entry)
        return ret

    def __eq__(self, other):
        return self.spaces == other.spaces


class Discrete(Space):
    """
    {0,1,...,n-1}

    Example usage:
    self.observation_space = spaces.Discrete(2)
    """
    SPACE_NAME = 'Discrete'

    def __init__(self, n):
        """
        Initialize Space
        :param n int: number of discrete choices
        """
        self.n = n
        Space.__init__(self, (), np.int64)

    def serialize(self):
        return dict(
            name=Discrete.SPACE_NAME,
            n=self.n,
            shape=None,
            dtype='uint8'
        )

    def sample(self):
        return np.random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    __contains__ = contains

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
