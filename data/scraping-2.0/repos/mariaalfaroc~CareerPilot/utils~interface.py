import io

import openai
import numpy as np
import gradio as gr
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from utils.model import load_model
from utils.data import transform_data, clean_sector

def plot_months(input, model):
    """Plot the probability of finding a job in the next six months, starting from the current month.
    input = {'sexo': str,
            'edad': int,
            'nhijos': int,
            'espanol': str,
            'minusv': str,
            'educa_cat': str,       
            'expe': str,
            'expe_perma': str,
            'expe_tcompleto': str,
            'paro': str,
            'mes_actual': str,
            'expe_sector: list}
    This is the input format of the function transform_data in utils/data.py"""
    months = ['Enero', 'Febrero', 'Marzo' ,  'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    # Iterate over months
    df_months = pd.DataFrame()
    mes_actual = input['mes_actual']
    for id, m in enumerate(months, start=1):
      input['mes_actual'] = m
      alt_df = transform_data(input)

      # Obtain predictions from binary model
      probs = model.predict_proba(alt_df)[0, 1]*100
      new_row = pd.DataFrame({'mes': [m], 'prob': [probs], 'mes_num': [id]}, columns=['mes', 'prob', 'mes_num'])
      df_months = pd.concat([df_months, new_row], ignore_index=True)

    # Keep only the first six months after the current month
    mes_actual_num = df_months[df_months.mes == mes_actual].mes_num.iloc[0]
    df_months.loc[df_months['mes_num'] < mes_actual_num, 'mes_num'] = df_months.loc[df_months['mes_num'] < mes_actual_num, 'mes_num'] + 12
    df_months[['mes_num']] = df_months[['mes_num']] - mes_actual_num + 1
    df_months = df_months.drop(df_months[df_months['mes_num'] > 6].index)    
    df_months = df_months.sort_values(by='mes_num', ascending=True)

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.bar(df_months.mes, df_months.prob)

    # Set plot title and axis labels
    ax.set_title('Probabilidad de encontrar empleo el próximo mes')
    ax.set_xlabel('')
    ax.set_ylabel('%')
    ax.set_xticks(range(len(df_months.mes)), df_months.mes.values, rotation=45)

    # Adjust the layout
    fig.tight_layout()

    # Save the plot to a file
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Open the saved file using PIL
    pil_image = Image.open(buffer)

    # Close the plot
    plt.close(fig)

    # Return the PIL Image object
    return pil_image

def plot_sector(input, model, model_classes):
    """Plot the probability of finding a job in the next six months, starting from the current month.
    The plot is grouped by sector.
    input = {'sexo': str,
            'edad': int,
            'nhijos': int,
            'espanol': str,
            'minusv': str,
            'educa_cat': str,       
            'expe': str,
            'expe_perma': str,
            'expe_tcompleto': str,
            'paro': str,
            'mes_actual': str,
            'expe_sector: list}
    This is the input format of the function transform_data in utils/data.py"""
    user_df = transform_data(input)

    # Obtain predictions from multinominal model
    probabilities = model.predict_proba(user_df).reshape(15,)
    proba_df = pd.DataFrame({'classes': model_classes, 'probabilities': probabilities})
    proba_df = proba_df.drop(proba_df[proba_df['classes'] == 'Paro'].index)
    proba_df = clean_sector(proba_df)
    
    desc_plot = proba_df[['probabilities', 'sector_clean']]
    desc_plot.loc[:, 'probabilities'] = desc_plot['probabilities'] * 100 
    desc_plot = desc_plot.sort_values(by='probabilities', ascending=False)

    # Set the data 
    fig, ax = plt.subplots() 
    y_pos = np.arange(len(desc_plot.sector_clean))

    ax.barh(y_pos, desc_plot.probabilities, align='center')
    ax.set_yticks(y_pos, labels=desc_plot.sector_clean)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('%')
    ax.set_title('Probabilidad de estar empleado el próximo mes, por sector')

    # Adjust the layout
    fig.tight_layout()

    # Save the plot to a file
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Open the saved file using PIL
    pil_image = Image.open(buffer)

    # Close the plot
    plt.close(fig)

    # Return the PIL Image object
    return pil_image

##############################################################################################################

def profiling(sexo, edad, nijos, espanol, minusv, educa_cat, expe, expe_perma, expe_tcompleto, paro, mes_actual, expe_sector, api):
    # Transform user input into a dataframe
    user = {'sexo': sexo,
            'edad': edad,
            'nhijos': nijos,
            'espanol': espanol,
            'minusv': minusv,
            'educa_cat': educa_cat,       
            'expe': expe,
            'expe_perma': expe_perma,
            'expe_tcompleto': expe_tcompleto,
            'paro': paro,
            'mes_actual': mes_actual,
            'expe_sector': expe_sector}
    user_df = transform_data(user)

    binary = load_model('gridSearch/final_results/models/binary_model.json')
    multiclass = load_model('gridSearch/final_results/models/multiclass_model.json')
    multiclass_classes = np.load('gridSearch/final_results/label_encoder.npy', allow_pickle=True)

    # Binary model - how likely is to be employed
    probabilities = binary.predict_proba(user_df)
    user_proba = probabilities[0, 1]*100
    print_user_proba = f'Probabilidad de estar empleado el próximo mes:\n{user_proba:.0f}%'

    # Multiclass model - how likely is to be employed at a certain sector
    probabilities = multiclass.predict_proba(user_df).reshape(15,)
    proba_df = pd.DataFrame({'classes': multiclass_classes, 'probabilities': probabilities})
    proba_df = proba_df.drop(proba_df[proba_df['classes'] == 'Paro'].index)
    proba_df = clean_sector(proba_df)

    best_model = proba_df.loc[proba_df['probabilities'].idxmax()]
    sector = best_model[2]
    prob = best_model[1]*100
    print_sector_proba = f'Sector económico con la mayor probabilidad:\n{sector} ({prob:.0f}%)'

    # Clean API Key
    try:
        api = api.strip()
    except:
        pass

    if api is not None and api != '':
        # OpenAI API
        openai.api_key = api
        # GPT prompt
        prompt = f'Esto es el resultado de un modelo de empleabilidad donde el usuario introduce sus características y se le presenta, en viñetas: la probabilidad de estar empleado al mes siguiente, el sector económico con la probabilidad más alta, una interpretación de los resultados y recomendaciones. Génerame la interpretación de los resultados (sin decir que son buenos o malos) y las recomendaciones personalizadas sobre formación deseada, páginas para encontrar empleo y ocupaciones con mayor probabilidad de encontrarlo. Ejemplo de input: {user} Ejemplo de output del modelo: {print_user_proba}. {print_sector_proba}.'
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}]
        )
        print_gpt = response.choices[0].message.content
        return f'{print_user_proba}\n\n{print_sector_proba}\n\nDescripción de los resultados:\n\n{print_gpt}', plot_sector(user, multiclass, multiclass_classes), plot_months(user, binary)
    else:
        # No GPT prompt
        return f'{print_user_proba}\n\n{print_sector_proba}', plot_sector(user, multiclass, multiclass_classes), plot_months(user, binary)

def get_interface():
    return gr.Interface(profiling,
                        inputs=[gr.Dropdown(['Hombre', 'Mujer'], label='Sexo'),
                                gr.Slider(16, 65, value=24, step = 1, label='Edad actual'),
                                gr.Slider(0, 10, value=0, step = 1, label='Número de hijos'),
                                gr.Dropdown(['Sí', 'No'], label='Nacionalidad española'),
                                gr.Dropdown(['Sí', 'No'], label='Minusvalía superior al 33%'),
                                gr.Dropdown(['Sin estudios', 'Primarios', 'Secundarios', 'FP o medio', 'Universitarios'], label='Nivel educativo'),
                                gr.Dropdown(['Sin experiencia', 'Menos de 1 año', 'Entre 1-2 años', 'Más de 2 años'], label='Experiencia laboral'),
                                gr.Dropdown(['Sin experiencia', 'Menos de 1 año', 'Entre 1-2 años', 'Más de 2 años'], label='Experiencia laboral como indefinido'),
                                gr.Dropdown(['Sin experiencia', 'Menos de 1 año', 'Entre 1-2 años', 'Más de 2 años'], label='Experiencia laboral a tiempo completo'),
                                gr.Dropdown(['Entre 0-3 meses', 'Entre 4-6 meses', 'Entre 7-12 meses', 'Entre 1-2 años', 'Más de 2 años'], label='Tiempo en paro'),
                                gr.Dropdown(['Enero', 'Febrero', 'Marzo' ,  'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'], label='Mes actual'),
                                gr.CheckboxGroup(['Construcción', 'Comercio', 'Hostelería', 'Administrativas y servicios auxiliares'], label='Experiencia laboral', info='¿Has trabajado en estos sectores durante los últimos 6 meses?'),
                                gr.Textbox(placeholder='OpenAI API Key (opcional)', info='Introduce tu clave de la API de OpenAI si tienes una. Esto hará que ChatGPT interprete los resultados obtenidos por CareerPilot.', label='OpenAI API Key (opcional)', type='password')],
                        title='CareerPilot: una aplicación para entender tu perfil laboral',
                        description='Descubre tu destino laboral con CareerPilot: ¡la aplicación que predice tu probabilidad de empleo en el próximo mes! Obtén información detallada sobre el sector con más oportunidades, recomendaciones de ocupaciones y formaciones clave, además de dónde buscarlas. ¡Impulsa tu carrera con Careerpilot y toma decisiones informadas para alcanzar el éxito profesional!',
                        outputs=[gr.Textbox(label='Resultados'), gr.Image(label='Imagen'), gr.Image(label='Imagen')],
                        allow_flagging='never')