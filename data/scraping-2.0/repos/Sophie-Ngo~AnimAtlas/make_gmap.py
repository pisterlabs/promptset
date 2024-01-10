from bokeh.io import output_notebook, show, output_file, curdoc
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, WMTSTileSource, ColumnDataSource, Circle, TextInput, GMapOptions, Button, CustomJS, Styles, RangeSlider, CDSView, BooleanFilter, CustomJSFilter, OpenURL, HoverTool, TapTool, Div, GlobalImportedStyleSheet, InlineStyleSheet
from bokeh.plotting import figure, show, output_file, gmap
from bokeh.layouts import gridplot, column, row, layout
from bokeh.embed import components, json_item
from bokeh.events import Tap
from pyrolite.util.time import Timescale # used to determine geological time periods

from openai import OpenAI

import apikeys

import pandas as pd
import numpy as np

global source
global bool_filter
global time_slider

time_slider = RangeSlider(value=(100, 200), start=0, end=4000, step=1, direction='rtl', title="Time Range (Ma)", sizing_mode='stretch_width')

data = {'0': [0], # placeholder data
        '0': [0]}
source = ColumnDataSource(data=data)
starting_records = 20
 # placeholder data, will be replaced with real data when map initializes
bool_filter = BooleanFilter([True]*starting_records) # placeholder data, will be replaced with real data when map initializes
starting_data = 'https://paleobiodb.org/data1.2/occs/list.csv?base_name=Dinosauria&pgm=gplates,scotese,seton&show=full,classext,genus,subgenus,acconly,ident,img,etbasis,strat,lith,env,timebins,timecompare,resgroup,ref,ent,entname,crmod&limit=' + str(starting_records)

# output_file("index.html")

# convert lat/long to web mercator format
def wgs84_to_web_mercator(df, lon="lng", lat="lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi/180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
    return df

# FETCH DATA
# num_records = 20
# global df
# df = pd.read_csv('https://paleobiodb.org/data1.2/occs/list.csv?base_name=Dinosauria&pgm=gplates,scotese,seton&show=full,classext,genus,subgenus,acconly,ident,img,etbasis,strat,lith,env,timebins,timecompare,resgroup,ref,ent,entname,crmod&limit=' + str(num_records), on_bad_lines='skip')

# # df = pd.read_csv("pbdb_data_dino_50.csv", on_bad_lines='skip')

# wgs84_to_web_mercator(df) 

# #used to display on the map
# global source
# source = ColumnDataSource(df)

# global p

map_options = GMapOptions(lat=0, lng=0, map_type="satellite", zoom=3)
# p = gmap(apikeys.googlekey, map_options, tools="tap, pan, wheel_zoom", active_scroll="wheel_zoom", sizing_mode="stretch_width", toolbar_location=None, height=600)
# p.xaxis.visible = False
# p.yaxis.visible = False

# global ids
# ids = df['occurrence_no']

def updateMap(api_url):
    global df
    global source
    global time_slider
    df = pd.read_csv(api_url, on_bad_lines='skip')

    wgs84_to_web_mercator(df) 

    #used to display on the map
    source = ColumnDataSource(df)

    p = gmap(apikeys.googlekey, map_options, tools="tap, pan, wheel_zoom", active_scroll="wheel_zoom", toolbar_location=None, sizing_mode="stretch_width", height=900)
    p.xaxis.visible = False
    p.yaxis.visible = False

    bool_filter = BooleanFilter([True]*len(df)) # true for everything, so shows everything initially
    # create a view that allows the points to only appear for the set time period (filter by time slider)
    # views do not change underlying data, they only filter what is shown
    view = CDSView(filter = bool_filter) 

    r = p.circle(x='lng', y='lat', fill_color='orange', line_color="firebrick", selection_color="blue", size=10, source=source, view=view)

    p.on_event(Tap, callbackShowData)

    selected_circle = Circle(fill_color="blue", size=20)
    nonselected_circle = Circle(fill_color="orange", line_color="firebrick", size=10)

    r.selection_glyph = selected_circle
    r.nonselection_glyph = nonselected_circle

    time_slider = RangeSlider(value=(df['min_ma'].min(), df['max_ma'].max()), start=df['min_ma'].min(), end=df['max_ma'].max(), step=1, direction='rtl', title="Time Range (Million Years Ago)", sizing_mode='stretch_width')
    callbackUpdateTimerange = CustomJS(args=dict(source=source, filter=bool_filter, time_slider=time_slider),
    code="""

    var data = source.data;

    const start = time_slider.value[0]
    const end = time_slider.value[1]
    
    // min/start means the oldest the fossil can be, max/end means the youngest the fossil can be
    for (let i = 0; i < filter.booleans.length; i++)
        filter.booleans[i] = data['min_ma'][i] >= start && data['max_ma'][i] <= end;

    console.log(filter.booleans)
    source.change.emit();
    """)
    time_slider.js_on_change('value', callbackUpdateTimerange)
    time_slider.on_change('value', callbackUpdateTimePeriod)

    callbackUpdateTimePeriod(None, None, time_slider.value)

    final_layout.children[1].children[0] = p # update map

    print("Map updated!")
    print("New number of rows:", len(df))

def getTimePeriod(min, max):
    min_ma = min
    max_ma = max

    ts = Timescale()

    max = ts.named_age(max_ma, level='Period')
    min = ts.named_age(min_ma, level='Period')

    if min == max:
        return min
    else:
        return str(max) + " - " + str(min)

def callbackUpdateTimePeriod(attr, old, new):
    label = getTimePeriod(new[0], new[1])

    time_slider_label = Div(text="<h1>Period: " + label + "</h1>", margin=(0,50,0,50))
    time_slider_row = row(time_slider, time_slider_label, sizing_mode='stretch_width')

    final_layout.children[1].children[1] = time_slider_row # update time slider

def callbackShowData(event):

    print("Getting details...")

    # if no data point is selected, do not change the existing div, but update the sidebar to show data
    if len(source.selected.indices) == 0:
        updateSidebar(nav, default_content)
        return

    global df

    indexActive = source.selected.indices[0]

    prompt = "What is " + str(df['accepted_name'][indexActive]) + "?"

    gpt_response = getGPTResponse(prompt)

    ref = getReference(df['occurrence_no'][indexActive])
    record = df.iloc[indexActive]

    data_div = Div(text=  
    "<h1>" + str(getAttributeFromList(record, 'accepted_name')).capitalize() + 
    "</h1><br><p>Time period: " + str(getTimePeriod(getAttributeFromList(record, 'min_ma'), getAttributeFromList(record, 'max_ma'))) + 
    "<br> (" + str(getAttributeFromList(record, 'max_ma')) + " million years ago - " + str(getAttributeFromList(record, 'min_ma')) + " million years ago) </p>" +
    "<p>Diet: " + str(getAttributeFromList(record, 'diet')).capitalize() + "</p>" +
    "<p> Rank: " + str(getAttributeFromList(record, 'accepted_rank')).capitalize() + "</p>" +
    "<h4>What is it?</h4><p>  " + str(gpt_response) + "</p>" +
    "<h4>Attribution</h4> <p> Title: " + str(getAttributeFromList(ref, 'reftitle')) + "</p>" +  # reftitle = reference title in ref csv
    "<p> Published in: " + str(getAttributeFromList(ref, 'pubyr')) + "</p>" +
    "<p> DOI: " + str(getAttributeFromList(ref, 'doi')) + "</p>" ,  

    margin=(50,0,0,0), stylesheets=[stylesheet])
    
    # prompt_box = TextInput(placeholder="", title="Have a question?")
    # prompt_button = Button(label="Ask", button_type="default")
    # prompt_button.on_click()

    # prompt_area = row(prompt_box, prompt_button)
    # gpt_answer = Div(text="<p>" + gpt_response + "</p>", stylesheets=[stylesheet])

    # content = column([data_div, prompt_area, gpt_answer], margin=(50,0,0,0), stylesheets=[stylesheet])

    updateSidebar(nav, data_div)

    print("Details updated!")

# work in progress
def askQuestion(prompt):
    answer = getGPTResponse(prompt)

    # update sidebar with answer



def getAttributeFromList(list, attr):
    if not isinstance(list, pd.Series):
        return 'dataframe is undefined.'

    value = list[attr]

    if pd.isnull(value):
    
        return ('No ' + str(attr) + ' listed.')

    return value

def getReference(occ_id):
    row = df[df["occurrence_no"] == occ_id]
    ref_id = row["reference_no"].iloc[0]
    link = "https://paleobiodb.org/data1.2/refs/single.csv?id=" + str(ref_id) + "&show=both"

    print("Fetching reference from URL:", link)

    ref = pd.read_csv(link, on_bad_lines='skip')

    try :
        ref = ref.iloc[0] # get first and only row
    except:
        print("Error fetching reference from URL:", link)
        return pd.DataFrame()
    
    return ref

def getGPTResponse(prompt):
    print("Asking: " + prompt)
    client = OpenAI(
        api_key=apikeys.openaikey,
    )

    chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
    )

    # copilit gave me this code, idk what it does yet
    # chat_completion = client.chat.completions.create(
    #     model="davinci",
    #     prompt=text,
    #     max_tokens=100,
    #     temperature=0.9,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0.6,
    #     stop=["\n", " Human:", " AI:"]
    # )

    return chat_completion.choices[0].message.content

# updates sidebar
def updateSidebar(nav, content):

    sidebar = column(nav, content, width=sidebar_width)

    final_layout.children[0] = sidebar

    return sidebar

def callbackSearchSubmit(event):

    base_link = "https://paleobiodb.org/data1.2/occs/list.csv?show=full&"
    # show=full to get all bolded attributes except paleolocation. contains lng and lat

    parameters = []

    if taxonomy.value != "":
        parameters.append("base_name=" + taxonomy.value)

    if record_limit.value != "":
        parameters.append("limit=" + record_limit.value)

    if ma_range_max.value != "":
        parameters.append("max_ma=" + ma_range_max.value)

    if ma_range_min.value != "":
        parameters.append("min_ma=" + ma_range_min.value)

    final_link = base_link + "&".join(parameters)

    print("Refreshing map with data from URL:", final_link)

    updateMap(final_link)

# show the search page
def callbackSearch(event):

    content = column(search_fields, margin=(50,0,0,0), stylesheets=[stylesheet])

    updateSidebar(nav, content) 

def introPage(event):

    div = Div(text="<h1>What is AnimAtlas?</h1><br><p> AnimAtlas allows you to explore fossil discoveries that have been recorded in the <a href='https://paleobiodb.org/'>Paleobiology Database.</a> </p>" + 
              "<p>Each dot on the map represents a fossil occurrence. What is a fossil occurence, you may ask? </p>" + 
              "<p>A <b>fossil occurrence</b> represents a taxon (for example, a species or genus) whose fossils are recorded at a specific place and a specific geological time. It may consist of " + 
              "more than one fossil. Each occurrence has been described in a publication, " +
              "which helps us gain information about it. Besides time range, researchers also store information such as the condition of the fossils, the habits of the organism, " + 
              "and data about the surrounding geography. </p> <h4>Things to do</h4> <list><li>You can click on any of the dots on the map to learn more about each fossil occurrence. </li>" + 
              "<li>Use the search bar to specify what appears on the map.</li> <li>Play with the time range slider to see what organisms lived during different time periods!</li></list>" + 
              "<br> <p> To return to this page at any time, click the 'Help' button at the top of the page. </p> ")

    content = column([div], margin=(50,0,0,0), stylesheets=[stylesheet])

    updateSidebar(nav, content)

def trexPage(event):

    div = Div(text="<h1>Tyrannosaurus Rex</h1><br><p>You probably know of Tyrannosaurus Rex--a fearsome creature, likely to be the largest land predator to have ever existed. " + 
              "Luckily, many Tyrannosaurus fossils have been recorded in this database. Most T. Rex fossils have been found in North America, which was their original range. " + 
              "We can use the map to visualize where these fearsome beasts have once roamed the planet. </p>" +
              "<p>Want to explore Tyrannosaurus fossil occurrences on your own?  " +
              "At the top of the page, click the 'Search' button. Then, in the 'Taxonomy' box, type 'Tyrannosaurus.' Then, click the 'Submit' button. </p> <br> " + 
              "<center><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Tyrannosaurus.jpg/1024px-Tyrannosaurus.jpg' width=300, height=300>" +
              "<p>'Black Beauty' at the Museum of Natural History (Stockholm, Sweden) <br> Image credit: <a href='https://commons.wikimedia.org/wiki/File:Tyrannosaurus.jpg'>Wikimedia Commons</a></p></center>")

    content = column([div], margin=(50,0,0,0), stylesheets=[stylesheet])

    updateSidebar(nav, content)

# search fields
taxonomy = TextInput(placeholder="e.g. Dinosauria", title="Taxonomy:")
record_limit = TextInput(placeholder="(optional) if too slow, limit records", title="Limit Number:")
ma_range_max = TextInput(placeholder="Oldest", title="Max Ma:")
ma_range_min = TextInput(placeholder="Youngest", title="Min Ma:")

submit_button = Button(label="Submit", button_type="default")
submit_button.on_click(callbackSearchSubmit)

search_fields = [taxonomy, record_limit, ma_range_max, ma_range_min, submit_button]

# button that, when clicked, directs user to fossil info page
detailButton = Button(label="Detail", button_type="default")
detailButton.on_click(callbackShowData)

# button that, when clicked, directs user to search page
searchButton = Button(label="Search", button_type="default")
searchButton.on_click(callbackSearch)

help_button = Button(label="Help", button_type="default")
help_button.on_click(introPage)

nav = row([detailButton, searchButton, help_button])

sidebar_width = 400

stylesheet = InlineStyleSheet(css=":host{color:black;height:100%;width:" + str(sidebar_width) + "px;position:fixed;z-index:1;top:0;left:0;background-color:white;overflow-x:hidden;padding:20px;font-size:15px;}:host p{padding:6px 8px 6px 16px;text-decoration:none;color:black;display:block}:host a:hover{color:blue}:host h1{padding:6px 8px 6px 16px;text-decoration:none;font-size:25px;color:black;display:block}")

# create a view that allows the points to only appear for the set time period (filter by time slider)
# views do not change underlying data, they only filter what is shown
# view = CDSView(filter = bool_filter) 


# p.hex_tile(df['lng'], df['lat'],size=0.5, hover_color="pink", hover_alpha=0.8)

# p.hexbin(df['lng'].values, df['lat'].values,size=0.5, hover_color="pink", hover_alpha=0.8)


# color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=4000)

# color_bar = ColorBar(color_mapper)

basic_div = Div(text=""" <p>Select a data point to learn about fossil <button type='button' onclick="console.log('test')"></button>occurences! </p> <p>Don't know where to start? Explore these pages below. </p> """,
          margin=(50,0,-200,0), stylesheets=[stylesheet]) 

intropage_button = Button(label="What is AnimAtlas?", button_type="default")
intropage_button.on_click(introPage)
trex_button = Button(label="Tyrannosaurus Rex", button_type="default")
trex_button.on_click(trexPage)
# homosapiens_button = Button(label="Homo Sapiens", button_type="default")

explore_buttons = column(intropage_button, trex_button, margin=(200,0,0,0), stylesheets=[InlineStyleSheet(css=":host{z-index:1;padding:20px;}")])

default_content = column([basic_div, explore_buttons])

# --------------------------------------------- #
# Final Layout Stuff 
# --------------------------------------------- #
p = Div(text="") #placeholder

final_layout = column(p, time_slider, margin=(0,0,0,0), sizing_mode='stretch_width')
final_layout = row(default_content, final_layout, sizing_mode='stretch_width')

curdoc().add_root(final_layout)

updateSidebar(nav, default_content)

updateMap(starting_data)



