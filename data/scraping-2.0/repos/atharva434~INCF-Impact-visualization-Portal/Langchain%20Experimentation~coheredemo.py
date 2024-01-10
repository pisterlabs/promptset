import cohere
from langchain.llms import Cohere
co = Cohere(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi")

text="""There are several modeling studies using brain network models which incorporate biologically
realistic macroscopic connectivity (the so-called connectome) to understand the global dynamics
observed in the healthy and diseased brain measured by different neuroimaging modalities such as
fMRI, EEG and MEG.
For this particular modelling approach in Computational Neuroscience, open source frameworks
enabling the collaboration between researchers with different backgrounds are not widely available.
The Virtual Brain is, so far, the only neuroinformatics project filling that place.
All projects below can be tailored for a 12-week time window, both Full-time and part-time, as the
features/pages can be built incrementally.
TVB has a demo dataset currently published on Zenodo.
We use it by manually downloading it, unzip then use inside tvb code and web GUI.
We intend to use Zenodo API instead.
The task is mainly for part-time, if only the above feature will be used, but it can be extended if
needed with other external datasources for Full-time applicants.
More details here or here.
Expected results: A set of classes , with demo Jupyter notebook, and unit tests"""

prompt = f"Tell me the impact of {text} in brief"
# response = co.generate(
#     model='command-nightly',
#     prompt = prompt)

# print(response)


with open('response.txt', 'w') as f:
    f.write(co(prompt))
# print(intro_paragraph)