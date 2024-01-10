# The following code is modified from openai/evolution-strategies-starter
# (https://github.com/openai/evolution-strategies-starter)
# under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.


import numpy as np
import logging

logger = logging.getLogger(__name__)

debug = False


class SharedNoiseTable(object):
    def __init__(self):
        import ctypes
        seed = 42
        # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        count = 250000000 if not debug else 1000000
        logger.info('Sampling {} random numbers with seed {}'.format(count, seed))
        
        #####################################################################
        # MODIFICATION: Noise table is an in-memory variable in the process instead of
        # being put in shared memory using multiprocessing.Array.
        # In Fiber or multiprocessing, a worker executes multiple processes, so
        # it is useful to put the noise table in shared memory so that all processes
        # withtin the worker can access it. However, in Lithops only one process is
        # executed per worker, so it is unnecessary to put it in shared memory.
        # In fact, AWS Lambda does not mount /dev/shm to the lambda function runtime,
        # so this would actually raise an exception.

        generator = np.random.Generator(np.random.PCG64(seed))
        self.noise = generator.random(size=count, dtype=np.float32)

        # import multiprocessing
        # self.noise = np.random.RandomState(seed).randn(count).astype(np.float32)  # 64-bit to 32-bit conversion here
        # self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        # self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        
        #####################################################################
        
        assert self.noise.dtype == np.float32
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)
