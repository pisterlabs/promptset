"""Parts of this code adapted and borrowed from OpenAI's Spinningup Repository:

https://spinningup.openai.com.
"""
import json
import shutil
import numpy as np
import tensorflow as tf
import os.path as osp
import time
import atexit
import os
import reflo.mpi_utils as mpi_utils

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logging:
    """
    Modified from OpenAI SpinningUp code.
    """
    def __init__(self, log_dir=None, output_fname=None):
        if mpi_utils.process_id() == 0:
            self.log_dir = log_dir or '/tmp/experiments/%i' % int(time.time())
            if osp.exists(self.log_dir):
                print(colorize('Warning: Log dir %s already exists! Storing info there anyway.' %
                               self.log_dir, 'red', bold=True))
            else:
                os.makedirs(self.log_dir)
            if output_fname is not None:
                self.log_file = open(osp.join(self.log_dir, output_fname), 'w')
                atexit.register(self.log_file.close)
                print(colorize('Logging data to %s' %
                               self.log_file.name, 'blue', bold=True))
            else:
                self.log_file = None
                print(colorize('No log file given. Output will not be saved to .txt file', 'blue', bold=True))
        else:
            self.log_dir = None
            self.log_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

    def log_msg(self, msg, color='green'):
        if mpi_utils.process_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, value):
        if self.first_row:
            self.log_headers.append(key)

        else:
            assert key in self.log_headers, "Trying to include new key {} that wasn't in first iteration".format(
                key)

        assert key not in self.log_current_row, '{} was already set this iteration. Need to call dump_tabular() before setting again'.format(key)
        self.log_current_row[key] = value

    def dump_tabular(self):
        """Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if mpi_utils.process_id() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%'+'%d' % max_key_len
            fmt = '| ' + keystr + 's | %15s |'
            n_slashes = 22 + max_key_len
            print('-'*n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, '')
                valstr = '%8.3g' % val if hasattr(val, '__float__') else val
                print(fmt % (key, valstr))
                vals.append(val)
            print('-'*n_slashes)
            if self.log_file is not None:
                if self.first_row:
                    self.log_file.write('\t'.join(self.log_headers)+'\n')
                self.log_file.write('\t'.join(map(str, vals))+'\n')
                self.log_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logging):
    """Pulled directly from OpenAI's SpinningUp code.

    A variant of Logger tailored for tracking average values over
    epochs. Typical use case: there is some quantity which is calculated
    many times throughout an epoch, and at the end of the epoch, you
    would like to report the average / std / min / max value of that
    quantity. With an EpochLogger, each time the quantity is calculated,
    you would use .. code-block:: python
    epoch_logger.store(NameOfQuantity=quantity_value) to load it into
    the EpochLogger's state. Then at the end of the epoch, you would use
    .. code-block:: python     epoch_logger.log_tabular(NameOfQuantity,
    **options) to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(
                v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = mpi_utils.mpi_statistics_scalar(
                vals, with_min_and_max=with_min_and_max)
            super().log_tabular(
                key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])
                super().log_tabular('Min'+key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """Lets an algorithm ask the logger for mean/std/min/max of a
        diagnostic."""
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(
            v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return mpi_utils.mpi_statistics_scalar(vals)
