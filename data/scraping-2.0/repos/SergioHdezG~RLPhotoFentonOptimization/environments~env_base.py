from abc import ABCMeta, abstractmethod

class EnvInterface(object, metaclass=ABCMeta):
    """
    This class is an interface for building custom environments. It is based on gym (from OpenAI) environment interfaces
    but using this interface you can avoid create a custom gym environment.
    """
    def __init__(self):
        self.action_space = None  # ActionSpaceInterface object.
        self.observation_space = None  # numpy nd array of observation shape

    @abstractmethod
    def reset(self):
        """
        Reset the environment to an initial state
        :return: (numpy nd array) observation.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Take an action and executes it on the environment.
        :param action: (int if actions are discrete or numpy array of floats if actions are continuous). Action to
            execute.
        :return: (numpy nd array) observation, (float) reward, (bool) done, (dict or None) additional info
        """
        pass

    @abstractmethod
    def render(self):
        """
        Render the environment.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close rendering window.
        """
        pass

    def deepcopy(self):
        """
        Make a new copy of the environment.
        When using very complex environment copy.deepcopy method may not work properly and you must define how to copy
        your environment when using agents with environment parallelization (A3C or parallel PPO).
        """
        import copy
        return copy.deepcopy(self)

class ActionSpaceInterface(object):
    """
    This class defines the ActionSpaceInterface type used in EnvInterface.
    """
    def __init__(self):
        self.n = None  # Number of actions.
        self.actions = None  # List of actions.

        # For continuous action spaces. Higher and lower bounds.
        self.low = None
        self.high = None
