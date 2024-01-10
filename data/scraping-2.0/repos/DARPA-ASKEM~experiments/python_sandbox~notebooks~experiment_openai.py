# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Experiment with OpenAI Code Completion
#
# Idea: use OpenAI's code completion API endpoint to attempt to generate Python code snippets
# that one can use as blocks in a scientific workflow.
#
# 

# %%
import os
import openai
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt


# %%
# Get API key from env variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# %%
# Codex Best Practices:
# * set temperature to be 0 or near 0
# * set max_tokens > 256
# * set finish_reason == "stop"
# * resample 3-5 times

# %%
# Test

prompt = """
# Python 3
\"\"\"
Get OpenAI model list and convert it into a Python pandas table with id as a column name
\"\"\"
import pandas as pd
import openai
models = openai.Model.list()['data']
df = pd.DataFrame(
"""

suffix = ")"

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  suffix = suffix,
  max_tokens = 256,
  temperature = 0.1,
  stop = '#'
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# """
# Get OpenAI model list and convert it into a Python pandas table with id as a column name
# """
# import pandas as pd
# import openai
# models = openai.Model.list()['data']
# df = pd.DataFrame(
#     [model['id'] for model in models],
#     columns=['id']
# )
# df.to_csv('models.csv', index=False)

# %%
models = openai.Model.list()['data']

df = pd.DataFrame(
    [model['id'] for model in models],
    columns = ['id']
)

df
# 	id
# 0	babbage
# 1	ada
# 2	davinci
# 3	text-embedding-ada-002
# 4	babbage-code-search-code
# ...	...
# 61	davinci-instruct-beta:2.0.0
# 62	text-ada:001
# 63	text-davinci:001
# 64	text-curie:001
# 65	text-babbage:001

# %%
# Load Paul's data cube
cube = xr.open_dataset('../data/Dedri Queries/cube.netcdf')

# %%
# Select cohort & scenario & timesteps using the Xarray cube

prompt = """
# Python 3
import xarray as xr
ds = xr.open_dataset('data.nc')
def count_cohort_size(ds):
  \"\"\"
  ds is an Xarray dataset with dimensions (scenarios, replicates, times) and data variables (health, location, age, sex, beta, gamma, masked, mask_mandate)
  filter the location variable where sex = 0 and health = 1 and age is in [0, 1]
  slice the location variable along the scenarios dimension where the scenarios value = 0, and along the times dimension where the times value is between 10 and 40
  return the number of rows without nan values in the slice
  \"\"\"
"""

suffix = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  stop = "#",
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import xarray as xr
# ds = xr.open_dataset('data.nc')
# def count_cohort_size(ds):
#   """
#   ds is an Xarray dataset with dimensions (scenarios, replicates, times) and data variables (health, location, age, sex, beta, gamma, masked, mask_mandate)
#   filter the location variable where sex = 0 and health = 1 and age is in [0, 1]
#   slice the location variable along the scenarios dimension where the scenarios value = 0, and along the times dimension where the times value is between 10 and 40
#   return the number of rows without nan values in the slice
#   """
#   return ds.location.where((ds.sex == 0) & (ds.health == 1) & (ds.age.isin([0, 1]))).sel(scenarios = 0, times = slice(10, 40)).dropna(dim = 'times').shape[0]


# %%[markdown]
# Note:
# * Incorrect answer
# * Correct answer has `.dropna(dim = 'replicates')`
# * Requires knowledge about the concept of dimension and how it works in Xarray
# * Slight changes in the prompt can change the response enough to cause more or less errors

# %%
def count_cohort_size(ds):
  """
  ds is an Xarray dataset with dimensions (scenarios, replicates, times) and data variables (health, location, age, sex, beta, gamma, masked, mask_mandate)
  filter the location variable where sex = 0 and health = 1 and age is in [0, 1]
  slice the location variable along the scenarios dimension where the scenarios value = 0, and along the times dimension where the times value is between 10 and 40
  return the number of rows without nan values in the slice
  """
  return ds.location.where((ds.sex == 0) & (ds.health == 1) & (ds.age.isin([0, 1]))).sel(scenarios = 0, times = slice(10, 40)).dropna(dim = 'replicates').shape[0]

count_cohort_size(cube)
# 126

# %%
# Convert to a Pandas DataFrame 
# with cols = scenario, replicate, timestep, attribute0, attribute1, ... 

if False:

  timesteps = np.arange(len(cube['times']))
  num_timesteps = len(timesteps)

  attributes = list(cube.data_vars.keys())
  num_attributes = len(attributes)
  # attributes = ['health', 'location', 'age', 'sex', 'beta', 'gamma', 'masked', 'mask_mandate']

  num_records = len(cube['scenarios']) * len(cube['replicates']) * num_timesteps * num_attributes

  array = np.empty((num_records, 3 + num_attributes), dtype = int)

  n = 0
  for i, scenario in enumerate(tqdm(cube['scenarios'])):
    for j, replicate in enumerate(cube['replicates']):

        array[n:(n + num_timesteps), 0] = i
        array[n:(n + num_timesteps), 1] = j
        array[n:(n + num_timesteps), 2] = timesteps

        for k, attr in enumerate(attributes):
          k += 3
          array[n:(n + num_timesteps), k] = cube[attr][i, j, :].to_numpy()

        n += 1 * num_timesteps

  df = pd.DataFrame(array, columns = ['scenario', 'replicate', 'timestep'] + attributes)

  # Save DataFrame
  df.to_parquet('../data/Dedri Queries/cube.gzip', compression = 'gzip')

# %%
# Load DataFrame
if False:
  df = pd.read_parquet('../data/Dedri Queries/cube.gzip')

# %%
# Select cohort & scenario & timesteps using a Pandas DataFrame

prompt = """
# Python 3
import pandas as pd
df = pd.read_csv('data.csv')
def count_cohort_size(df):
  \"\"\"
  Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [0, 1] and sex = 0 and health = 1
  Return the number of rows with unique values in the replicate column
  \"\"\"
"""

suffix = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  stop = "#",
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import pandas as pd
# df = pd.read_csv('data.csv')
# def count_cohort_size(df):
#   """
#   Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [0, 1] and sex = 0 and health = 1
#   Return the number of rows with unique values in the replicate column
#   """
#   return df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (df['age'].isin([0, 1])) & (df['sex'] == 0) & (df['health'] == 1)]['replicate'].nunique()

# def count_cohort_infected(df):
#   """
#   Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [0, 1] and sex = 0 and health = 1
#   Return the number of rows with unique values in the replicate column
#   """
#   return df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (df['age'].isin([0, 1])) & (df['sex'] == 0) & (df['health'] == 2)]['replicate'].nunique()

# def count_cohort_rec


# %%[markdown]
# Note:
# * Human intervention needed to at least select the lines of code that are relevant in the response.
# * Interesting that some of the extended response contains answers to somewhat similar queries (e.g. `count_cohort_size_by_mask_mandate`).
# * Could use "def" or "#" as the `stop` condition but questionable
# * Best practices also suggest requesting multiple responses and having a human select the best one.

# %%
def count_cohort_size(df):
  """
  Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [0, 1] and sex = 0 and health = 1
  Return the number of rows with unique values in the replicate column
  """
  return df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (df['age'].isin([0, 1])) & (df['sex'] == 0) & (df['health'] == 1)]['replicate'].nunique()

count_cohort_size(df)
# 1275

# %%
# Select Sample 1

prompt = """
# Python 3
import pandas as pd
df = pd.read_csv('data.csv')
def select_sample_1(df):
  \"\"\"
  Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [0, 1] and sex = 0 and health = 1
  Return the filtered dataframe
  \"\"\"
"""

suffix = ""
stop = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  # stop = stop,
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import pandas as pd
# df = pd.read_csv('data.csv')
# def select_sample_1(df):
#   """
#   Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [0, 1] and sex = 0 and health = 1
#   Return the filtered dataframe
#   """
#   # YOUR CODE HERE
#   return df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (df['age'].isin([0, 1])) & (df['sex'] == 0) & (df['health'] == 1)]

# def select_sample_2(df):
#   """
#   Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [0, 1] and sex = 0 and health = 1
#   Return the filtered dataframe
#   """
#   # YOUR CODE HERE
#   return df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (df['age'].isin([0, 1])) & (df['sex'] == 0) & (df['health'] == 1)]

# def select_sample_3(df):
#   """
#   Pandas dataframe with columns = scenario, replicate


# %%
def select_sample_1(df):
  """
  Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [0, 1] and sex = 0 and health = 1
  Return the filtered dataframe
  """
  return df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (df['age'].isin([0, 1])) & (df['sex'] == 0) & (df['health'] == 1)]

df_1 = select_sample_1(df)

# %%
# Select Sample 2

prompt = """
# Python 3
import pandas as pd
df = pd.read_csv('data.csv')
def select_sample_2(df):
  \"\"\"
  Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [2, 3] and sex = 0 and health = 1
  Return the filtered dataframe
  \"\"\"
"""

suffix = ""
stop = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  # stop = stop,
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import pandas as pd
# df = pd.read_csv('data.csv')
# def select_sample_2(df):
#   """
#   Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [2, 3] and sex = 0 and health = 1
#   Return the filtered dataframe
#   """
#   # YOUR CODE HERE
#   df = df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (df['age'].isin([2, 3])) & (df['sex'] == 0) & (df['health'] == 1)]
#   return df

# df_sample_2 = select_sample_2(df)
# df_sample_2.head()

# # Python 3
# import pandas as pd
# df = pd.read_csv('data.csv')
# def select_sample_3(df):
#   """
#   Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [2, 3] and sex = 0 and health = 1
#   Return the filtered dataframe
#   """
#   # YOUR CODE HERE
#   df = df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (

# %%
def select_sample_2(df):
  """
  Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Filter rows with scenario = 0 and timestep is in range(10, 40) and age is in [2, 3] and sex = 0 and health = 1
  Return the filtered dataframe
  """
  # YOUR CODE HERE
  df = df[(df['scenario'] == 0) & (df['timestep'].between(10, 40)) & (df['age'].isin([2, 3])) & (df['sex'] == 0) & (df['health'] == 1)]
  return df

df_2 = select_sample_2(df)

# %%
# Compare Sample 1 and Sample 2 as time series

prompt = """
# Python 3
import pandas as pd
import matplotlib.pyplot as plt
def compare_dfs_over_time(df_1, df_2):
  \"\"\"
  df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Group and count df_1 and df_2 by the timestep column values
  Join together as a single dataframe
  Plot using plt
  \"\"\"
"""

suffix = ""
stop = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  # stop = stop,
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import pandas as pd
# import matplotlib.pyplot as plt
# def compare_dfs_over_time(df_1, df_2):
#   """
#   df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Group and count df_1 and df_2 by the timestep column values
#   Join together as a single dataframe
#   Plot using plt
#   """
#   df_1_grouped = df_1.groupby(['timestep']).count()
#   df_2_grouped = df_2.groupby(['timestep']).count()
#   df_1_grouped.rename(columns={'scenario': 'scenario_1'}, inplace=True)
#   df_2_grouped.rename(columns={'scenario': 'scenario_2'}, inplace=True)
#   df_1_grouped = df_1_grouped.drop(columns=['replicate', 'health', 'location', 'age', 'sex', 'beta', 'gamma', 'masked', 'mask_mandate'])
#   df_2_grouped = df_2_grouped.drop(columns=['replicate', 'health', 'location', 'age', 'sex', 'beta', 'gamma', 'masked', 'mask_mandate'])
#   df_joined = df_1_grouped.join(df_2_grouped)
#   df_joined.plot()
#   plt.show

# %%
def compare_dfs_over_time(df_1, df_2):
  """
  df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Group and count df_1 and df_2 by the timestep column values
  Join together as a single dataframe
  Plot using plt
  """
  df_1_grouped = df_1.groupby(['timestep']).count()
  df_2_grouped = df_2.groupby(['timestep']).count()
  df_1_grouped.rename(columns={'scenario': 'scenario_1'}, inplace=True)
  df_2_grouped.rename(columns={'scenario': 'scenario_2'}, inplace=True)
  df_1_grouped = df_1_grouped.drop(columns=['replicate', 'health', 'location', 'age', 'sex', 'beta', 'gamma', 'masked', 'mask_mandate'])
  df_2_grouped = df_2_grouped.drop(columns=['replicate', 'health', 'location', 'age', 'sex', 'beta', 'gamma', 'masked', 'mask_mandate'])
  df_joined = df_1_grouped.join(df_2_grouped)
  df_joined.plot()
  plt.show

compare_dfs_over_time(df_1, df_2)

# %%[markdown]
# Note:
# * This response works
# * Tediousness could be alleviated by automatically populating the prompt with the inputs' schema

# %%
# Describe the derived quantity "count" in a given sample

prompt = """
# Python 3
import pandas as pd
import matplotlib.pyplot as plt
def describe_sample_count(df):
  \"\"\"
  df: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Group and count df by the timestep column values
  Use Pandas to describe the count column
  \"\"\"
"""

suffix = ""
stop = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  # stop = stop,
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import pandas as pd
# import matplotlib.pyplot as plt
# def describe_sample_count(df):
#   """
#   df: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Group and count df by the timestep column values
#   Use Pandas to describe the count column
#   """
#   df_count = df.groupby(['timestep']).count()
#   df_count = df_count.rename(columns={'scenario': 'count'})
#   df_count = df_count.drop(columns=['replicate', 'health', 'location', 'age', 'sex', 'beta', 'gamma', 'masked', 'mask_mandate'])
#   return df_count.describe()
# def describe_sample_mean(df):
#   """
#   df: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Group and mean df by the timestep column values
#   Use Pandas to describe the mean column
#   """
#   df_mean = df.groupby(['timestep']).mean()
#   df_mean = df_mean.rename(columns={'scenario': 'mean'})
#   df_mean = df_mean.drop(columns=['replicate', 'health', 'location', 'age', 'sex', 'beta', '

# %%
def describe_sample_count(df):
  """
  df: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Group and count df by the timestep column values
  Use Pandas to describe the count column
  """
  df_count = df.groupby(['timestep']).count()
  df_count = df_count.rename(columns={'scenario': 'count'})
  df_count = df_count.drop(columns=['replicate', 'health', 'location', 'age', 'sex', 'beta', 'gamma', 'masked', 'mask_mandate'])
  return df_count.describe()

describe_sample_count(df_1)
# 	    count
# count	31.000000
# mean	610.870968
# std	145.465401
# min	313.000000
# 25%	505.000000
# 50%	668.000000
# 75%	731.000000
# max	754.000000

# %%[markdown]
# Note: Works exactly as desired

# %%
# Compute the Pearson correlation coefficient between the counts of the two sample dataframes

prompt = """
# Python 3
import pandas as pd
import matplotlib.pyplot as plt
def compute_pearson_correlation(df_1, df_2):
  \"\"\"
  df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Group and count df_1 and df_2 by the timestep column values
  Return the Pearson correlation coefficient between df_1 and df_2
  \"\"\"
"""

suffix = ""
stop = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  # stop = stop,
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import pandas as pd
# import matplotlib.pyplot as plt
# def compute_pearson_correlation(df_1, df_2):
#   """
#   df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Group and count df_1 and df_2 by the timestep column values
#   Return the Pearson correlation coefficient between df_1 and df_2
#   """
#   df_1_grouped = df_1.groupby(['timestep']).count()
#   df_2_grouped = df_2.groupby(['timestep']).count()
#   return df_1_grouped.corrwith(df_2_grouped, axis=0, method='pearson')
# 
# def plot_pearson_correlation(df_1, df_2, title):
#   """
#   df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   title: String
#   Plot the Pearson correlation coefficient between df_1 and df_2
#   """
#   df_1_grouped = df_1.groupby(['timestep']).count()
#   df_2_grouped = df_2.groupby(['timestep']).count()
#   df_1

# %%
def compute_pearson_correlation(df_1, df_2):
  """
  df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Group and count df_1 and df_2 by the timestep column values
  Return the Pearson correlation coefficient between df_1 and df_2
  """
  df_1_grouped = df_1.groupby(['timestep']).count()
  df_2_grouped = df_2.groupby(['timestep']).count()
  return df_1_grouped.corrwith(df_2_grouped, axis=0, method='pearson')

compute_pearson_correlation(df_1, df_2)

# %%
# Do a one-way analysis of variance (ANOVA) between the counts of the two sample dataframes
# (compare the means of two or more samples to check if they come from the different distributions)

prompt = """
# Python 3
import pandas as pd
import matplotlib.pyplot as plt
def do_oneway_anova(df_1, df_2):
  \"\"\"
  df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Group and count df_1 and df_2 by the timestep column values
  Return one-way ANOVA between df_1 and df_2
  \"\"\"
"""

suffix = ""
stop = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  # stop = stop,
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import pandas as pd
# import matplotlib.pyplot as plt
# def do_oneway_anova(df_1, df_2):
#   """
#   df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Group and count df_1 and df_2 by the timestep column values
#   Return one-way ANOVA between df_1 and df_2
#   """
#   df_1_grouped = df_1.groupby(['timestep']).count()
#   df_2_grouped = df_2.groupby(['timestep']).count()
#   return stats.f_oneway(df_1_grouped['scenario'], df_2_grouped['scenario'])

# def do_oneway_anova_by_age(df_1, df_2):
#   """
#   df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Group and count df_1 and df_2 by the timestep column values
#   Return one-way ANOVA between df_1 and df_2
#   """
#   df_1_grouped = df_1.groupby(['timestep', 'age']).count()
#   df_2_grouped

# %%
def do_oneway_anova(df_1, df_2):
  """
  df_1: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  df_2: Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Group and count df_1 and df_2 by the timestep column values
  Return one-way ANOVA between df_1 and df_2
  """
  df_1_grouped = df_1.groupby(['timestep']).count()
  df_2_grouped = df_2.groupby(['timestep']).count()
  return sp.stats.f_oneway(df_1_grouped['scenario'], df_2_grouped['scenario'])

x = do_oneway_anova(df_1, df_2)
print(x)
# F_onewayResult(statistic=22.44033782150607, pvalue=1.370453444717472e-05)

# %%[markdown]
# Note: Correct response but used `stats` without stating the need to do `import scipy.stats as stats`

# %%
# Select a given scenario-cohort-timerange sample
# Count the number of timesteps during which an unique replicate is infected (health = 1)
# Average across all replicates in the sample

# Select
prompt = """
# Python 3
import pandas as pd
df = pd.read_csv('data.csv')
def count_sample_timesteps(df, scenario_of_interest, age_of_interest, health_of_interest):
  \"\"\"
  Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Filter rows with scenario = scenario_of_interest and age is in age_of_interest and health = health_of_interest
  Only keep the replicate and timestep columns
  Group by replicate values and count
  Return the mean
  \"\"\"
"""

suffix = ""
stop = ""

response = openai.Completion.create(
  model = 'code-davinci-002',
  prompt = prompt,
  # suffix = suffix,
  # stop = stop,
  max_tokens = 256,
  temperature = 0.1,
)

print(prompt + response['choices'][0]['text'] + suffix)
# # Python 3
# import pandas as pd
# df = pd.read_csv('data.csv')
# def select_sample_3(df, scenario_of_interest, age_of_interest, health_of_interest):
#   """
#   Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Filter rows with scenario = scenario_of_interest and age is in age_of_interest and health = health_of_interest
#   Only keep the replicate and timestep columns
#   Group by replicate values and count
#   Return the mean
#   """
#   return df[(df['scenario'] == scenario_of_interest) & (df['age'].isin(age_of_interest)) & (df['health'] == health_of_interest)][['replicate', 'timestep']].groupby('replicate').count().mean()[0]

# def count_sample_timesteps_by_location(df, scenario_of_interest, age_of_interest, health_of_interest, location_of_interest):
#   """
#   Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
#   Filter rows with scenario = scenario_of_interest and age is in age_of_interest and health = health_of_interest and location = location_of_interest
#   Only keep the replicate and timestep columns
#   Group by replicate values and count
#   Return the mean
#   """
#   return df[(df['scenario'] == scenario_of_interest) & (df['age'].isin(age_of_interest)) & (df['health'] == health_of

# %%
def count_sample_timesteps(df, scenario_of_interest, age_of_interest, health_of_interest):
  """
  Pandas dataframe with columns = scenario, replicate, timestep, health, location, age, sex, beta, gamma, masked, mask_mandate
  Filter rows with scenario = scenario_of_interest and age is in age_of_interest and health = health_of_interest
  Only keep the replicate and timestep columns
  Group by replicate values and count
  Return the mean
  """
  return df[(df['scenario'] == scenario_of_interest) & (df['age'].isin(age_of_interest)) & (df['health'] == health_of_interest)][['replicate', 'timestep']].groupby('replicate').count().mean()[0]


x = count_sample_timesteps(df, 0, [0, 1], 1)
print(x)

x = count_sample_timesteps(df, 0, [2, 3], 1)
print(x)

# %%
