from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(name='ddpg_curiosity_mc_her',
      packages=[package for package in find_packages()
                if package.startswith('ddpg_curiosity_mc_her')],
      install_requires=[
          'gym[mujoco]',
          'scipy',
          'tqdm',
          'joblib',
          'dill',
          'progressbar2',
          'mpi4py',
          'cloudpickle',
          'click',
          'opencv-python',
          'numpy',
          'plotly',
          'matplotlib'
      ],
      description='DDPG with Curiosity and Multi-Criteria HER, modified from OpenAI baselines.',
      author='',
      url='https://github.com/openai/baselines',
      author_email='',
      version='0.1.5')


# ensure there is some tensorflow build with version above 1.4
try:
    from distutils.version import StrictVersion
    import tensorflow
    assert StrictVersion(tensorflow.__version__) >= StrictVersion('1.4.0')
except ImportError:
    assert False, "TensorFlow needed, of version above 1.4"
