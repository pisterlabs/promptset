from langchain.prompts import PromptTemplate

template = '''
You extract details from scientific articles.

Methods:

To examine the effects of microplastic pollution on coral reef ecosystems, we conducted a comprehensive field study across several sites in the Indo-Pacific region. These sites were selected based on varying levels of microplastic pollution, ranging from low to high concentrations. At each site, we collected samples of seawater, sediment, coral tissue, and reef-associated organisms to investigate the distribution and accumulation of microplastics.

Seawater samples were collected at different depths and filtered through a 0.45 μm mesh to isolate microplastic particles. Sediment samples were collected using a sediment core sampler and analyzed for microplastic content. Coral tissue samples were obtained by carefully removing small fragments of coral and subsequently examining them for microplastics under a microscope. Reef-associated organisms, including fish and invertebrates, were sampled to assess potential trophic transfer of microplastics through the food web.

Results:

Our results indicate that microplastic pollution is pervasive across all study sites, with varying concentrations in seawater, sediment, coral tissue, and reef-associated organisms. The highest concentrations of microplastics were found at sites with the greatest anthropogenic influence, such as those near urban centers and shipping lanes.

Coral tissue samples exhibited microplastic accumulation, with higher levels observed in corals from highly polluted sites. Additionally, our analysis of reef-associated organisms revealed that microplastics were present in the gut contents of several species, indicating trophic transfer within the food web.

The presence of microplastics in coral tissue was associated with increased stress markers, such as reactive oxygen species production and upregulation of stress-related genes. This suggests that microplastic pollution may compromise the health and resilience of coral reefs by inducing physiological stress responses.
You would produce the following output:
- Field study conducted across several sites in the Indo-Pacific region.
- Sites selected based on varying levels of microplastic pollution.
- Samples collected: seawater, sediment, coral tissue, and reef-associated organisms.
- Seawater samples filtered to isolate microplastic particles.
- Sediment samples analyzed for microplastic content.
- Coral tissue samples examined for microplastics under a microscope.
- Reef-associated organisms sampled to assess trophic transfer of microplastics.
- Microplastic pollution pervasive across all study sites.
- Highest concentrations found near urban centers and shipping lanes.
- Coral tissue samples exhibited microplastic accumulation.
- Microplastics present in the gut contents of several reef-associated species.
- Presence of microplastics in coral tissue associated with increased stress markers.
- Microplastic pollution may compromise coral health and resilience

Here is the text chunk you should summarize:
{text_chunk}
'''

extractor_template = PromptTemplate(template = template,
                                        input_variables = ['text_chunk'])

template_answerer = '''
you will be given some retrieved details that may be relevant to a question

answer the question using those details

Details:
{details}

Question:
{question}
'''

answerer_template = PromptTemplate(template = template_answerer, input_variables=['details','question'])
