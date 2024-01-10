import numpy as np
import sympy

from ..signal_processing.filters import Coherence


class CompositeSignal:
    """
    Class for composite signal
    """
    def __init__(self, signals, expression, name, ind, fs):
        """
        Constructor
        :param signals: list of all signals
        :param expression: str expression
        """
        self.ind = ind
        self.name = name
        self.signals = signals
        self.coh_filter = None
        if 'coh' in expression.lower():
            names = ''.join([ch if ch.isalnum() else ' ' for ch in expression]).split()[1:]
            self.signals_idx = [j for j, signal in enumerate(self.signals) if signal.name in names]
            self.signals = [self.signals[j] for j in self.signals_idx]
            self.expression_lambda = self.coherence
            self.coh_filter = Coherence(500, fs, (8, 12))
        elif expression == '':
            self.expression_lambda = self.push_zeros
        else:
            self._signals_names = [signal.name for signal in self.signals]
            self.expression = sympy.sympify(expression)
            self.expression_lambda = sympy.lambdify(self._signals_names, self.expression, modules="numpy")
            self.signals_idx = list(range(len(signals)))
        self.current_sample = 0
        self.current_chunk = None
        # signal statistics
        self.scaling_flag = False
        self.mean = np.nan
        self.std = np.nan

    def update(self, chunk):
        self.current_sample = self.expression_lambda(*[signal.current_chunk for signal in self.signals])
        if self.scaling_flag and self.std>0:
            self.current_sample = (self.current_sample - self.mean) / self.std
        self.current_chunk = self.current_sample*np.ones(len(chunk))
        pass

    def coherence(self, x1, x2):
        X = np.vstack([x1, x2]).T
        return self.coh_filter.apply(X)[-1]

    def push_zeros(self, *args):
        return np.zeros(len(args[0]))

    def update_statistics(self, updated_derived_signals_recorder=None, stats_type='meanstd'):
        signals_data = updated_derived_signals_recorder.copy()
        if self.coh_filter is None:
            if signals_data.shape[1] > 1:
                signal_recordings = self.expression_lambda(*signals_data.T)
            else:
                signal_recordings = np.apply_along_axis(self.expression_lambda, 0, signals_data)
            if stats_type == 'meanstd':
                self.mean = signal_recordings.mean()
                self.std = signal_recordings.std()
            elif stats_type == 'max':
                self.std = signal_recordings.max()
                self.std = 1 if self.std == 0 else self.std
                self.mean = 0
        else:
            self.coh_filter.buffer *= 0
            self.mean, self.std = (0, 1)
        self.enable_scaling()

    def enable_scaling(self):
        self.scaling_flag = True

    def descale_recording(self, data):
        return data * self.std + self.mean if self.scaling_flag else data
