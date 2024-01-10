import data.questions_const
import tiktoken
from config import OPEN_AI_KEY
import openai

GPT_3_T_4 = 'gpt-3.5-turbo'
GPT_3_T_16 = 'gpt-3.5-turbo-16k'
GPT_4 = 'gpt-4'

def openai_count_tokens(input_text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_count = len(encoding.encode(input_text))
    return token_count


def post_paper_prompt(p_prompts, fake=False):
    from prompt_engineering.prompt_engineering import PaperPrompt
    if not isinstance(p_prompts, PaperPrompt):
        raise "argument must be of type PaperPrompt"
    res = ''
    for i in range(p_prompts.number_of_api_calls):
        res += access_chat_gpt_3(p_prompts.contents[i],
                                 fake=fake)
    return res


def access_chat_gpt_3(prompts_text: [],
                      fake=False):
    openai.api_key = OPEN_AI_KEY
    prompts = [dict(role="user",
                    content=txt) for txt in prompts_text]
    chat_completion = openai.ChatCompletion.create(
        model=GPT_3_T_16,
        messages=prompts)
    res = chat_completion.choices[0].message.content
    # check the chat_completion.usage to check for overloads and actual
    # tokens size
    return res


dummy_response = "DOI: 10.1016/j.jacc.2019.11.056\n" \
                 "PMID: 31918846\n" \
                 "Number of patients: 1,000\n" \
                 "Patients age: 50-60\n"

mock_response = \
    """Ref DOI number: 10.1002/adem.201900288
Ref Lead author: Manisha Panda
Ref Publication date: 2019-XX-XX (publication year is provided in the paper, but specific month and day are not given)

Cell Stack sequence: PET-ITO | PEN-ITO | Perovskite

Cell Area Measured: 0.1 cm^2

Cell Number of cells per substrate: Not provided in the paper

Cell Architecture: nip (inverted)

Cell Flexible: Yes

Cell Semitransparent: No

Cell Semitransparent Average visible transmittance Wavelength range: Not mentioned in the paper

Module: No module information provided in the paper

Substrate Stack sequence: PET | ITO

ETL Stack sequence: TiO2-c | TiO2-mp

ETL Deposition Procedure: Spin-coating

Perovskite Single crystal: No

Perovskite Dimension 0D (Quantum dot): No

Perovskite Dimension 2D: No

Perovskite Dimension 2D/3D mixture: No

Perovskite Dimension 3D: Yes

Perovskite Dimension List of layers: 3

Perovskite Composition Perovskite inspired structure: No

Perovskite Composition A-ions: Cs; FA; MA

Perovskite Composition A-ions Coefficients: 0.05; 0.79; 0.16

Perovskite Composition B-ions: Pb

Perovskite Composition B-ions Coefficients: 1

Perovskite Composition C-ions: Cs; FA; MA

Perovskite Composition C-ions Coefficients: 0.49; 2.51

Perovskite Composition Inorganic perovskite: No

Perovskite Composition Lead free: No

Perovskite Band gap Graded: No

Perovskite Deposition Number of deposition steps: Not provided in the paper

Perovskite Deposition Procedure: Spin-coating

Perovskite Deposition Aggregation state of reactants (Liquid/Gas/Solid): Liquid

Perovskite Deposition Solvents: DMF; DMSO

Perovskite Deposition Quenching induced crystallisation: Yes

Perovskite Deposition Solvent annealing: Yes

HTL Stack sequence: Spiro-MeOTAD

Backcontact Stack sequence: Au

Backcontact Deposition Procedure: Evaporation

Add Lay Front: Yes

Add Lay Back: No

Encapsulation: Yes

JV Measured: Yes

JV Average over N number of cells: Yes

JV Certified values: No

JV Light Intensity: 100 mW/cm^2

JV Light Wavelength range: Not mentioned in the paper

JV Light Masked cell: No

Stabilised performance Measured: Yes

EQE Measured: Yes

Stability Measured: Yes

Stability Average over N number of cells: Yes

Stability Light Wavelength range: Not mentioned in the paper

Stability Light UV filter: No

Stability Potential bias Range: Not mentioned in the paper

Stability Temperature Range: Not mentioned in the paper

Stability Relative humidity Range: Not mentioned in the paper

Stability Periodic JV measurements: Not mentioned in the paper

Stability PCE Burn in observed: Not mentioned in the paper

Outdoor Tested: Not mentioned in the paper

Outdoor Average over N number of cells: Not mentioned in the paper

Outdoor Location Coordinates: Not mentioned in the paper

Outdoor Installation Number of solar tracking axis: Not mentioned in the paper

Outdoor Time Start: Not mentioned in the paper

Outdoor Time End: Not mentioned in the paper

Outdoor Potential bias Range: Not mentioned in the paper

Outdoor Temperature Range: Not mentioned in the paper

Outdoor Periodic JV measurements: Not mentioned in the paper

Outdoor PCE Burn in observed: Not mentioned in the paper

Outdoor Detailed weather data available: Not mentioned in the paper

Outdoor Spectral data available: Not mentioned in the paper

Outdoor Irradiance measured: Not mentioned in the paper"""

if __name__ == '__main__':
    access_chat_gpt_3()
