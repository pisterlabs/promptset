import sys
import os
from time import sleep
from constants import  get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY, MAX_TOKENS
import project
import component_lister
import declare_or_use_comp_classifier
import get_if_service_is_used
import get_if_service_is_singleton
# import list_component_expansions
import list_how_service_describes_components
import resolve_component_imports
import component_descriptions_exact
import get_interface_parts
import get_interface_parts_usage
import primary_component
import list_component_props
import json
import result_loader

import openai
import tiktoken

ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """Act as a full-stack ai software developer.
It is your task to write all the code for the component '{0}'

use the following development stack:
{1}
"""
user_prompt = """The component '{0}' is described as follows:
{1}
{2}
globally declared features:
{3}

imports (only include the imports that are used in the code):
{4}"""
term_prompt = """
Use small functions.
When the user text contains references to other components, use the component, do not write the functionality inline.
A file always contains the definition for 1 component, service or object, no more.
Add documentation to your code.
only write valid code
do not include any intro or explanation, only write code
add styling names"""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('render_component', key)
    
    def reportTokens(prompt):
        encoding = tiktoken.encoding_for_model(model)
        # print number of tokens in light gray, with first 10 characters of prompt in green
        token_len = len(encoding.encode(prompt))
        print(
            "\033[37m"
            + str(token_len)
            + " tokens\033[0m"
            + " in prompt: "
            + "\033[92m"
            + prompt
            + "\033[0m"
        )
        return token_len

    # Set up your OpenAI API credentials
    openai.api_key = OPENAI_API_KEY

    messages = []
    prompt = system_prompt.format(params['component'], params['dev_stack']) + term_prompt
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['component'], params['feature_description'].strip(), params['interface_parts'], params['global_features'], params['imports'].strip() )
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    
    total_tokens = (total_tokens * 2) + DEFAULT_MAX_TOKENS # code needs max allowed
    if total_tokens > MAX_TOKENS:
        total_tokens = MAX_TOKENS
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": total_tokens,
        "temperature": 0,
    }

    # Send the API request
    keep_trying = True
    response = None
    while keep_trying:
        try:
            response = openai.ChatCompletion.create(**params)
            keep_trying = False
        except Exception as e:
            # e.g. when the API is too busy, we don't want to fail everything
            print("Failed to generate response (retrying in 30 sec). Error: ", e)
            sleep(30)
            print("Retrying...")

    # Get the reply from the API response
    if response:
        reply = response.choices[0]["message"]["content"] # type: ignore
        print("response: ", reply)
        return reply
    return None


def collect_file_list(title, file_names, writer):
    names_str = json.dumps(file_names)
    writer.write(f'# {title}\n')
    writer.write(f'{names_str}\n')
    writer.flush()


def get_file_path(title, root_path):
    file_name = title.replace(" > ", "_")
    file_path = os.path.join(root_path, file_name.replace(" ", "_") + ".js")
    return file_path

def collect_response(title, response, root_path):
    file_name = title.replace(" > ", "_").replace(" ", "_")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    file_path = os.path.join(root_path, file_name + ".js")
    with open(file_path, "w") as writer:
        writer.write(response)
    return file_path


def get_import_service_line(import_def, cur_path, root_path):
    service = import_def['service']
    service_path = os.path.join(root_path, import_def['path'])
    service_path = os.path.relpath(service_path, cur_path)
    is_global = get_if_service_is_singleton.is_singleton(import_def['service_loc'], service)
    if is_global:
        service_txt = "global object"
        service = service.lower()
    else:
        service_txt = "service"
    return f"The {service_txt} {service} from {service_path} (exported as default)\n"


def get_all_imports(component, full_title, cur_path, root_path):
    imports_txt = ''
    cur_path = cur_path.strip()
    imports = resolve_component_imports.get_data(full_title)
    primary = primary_component.get_primary(full_title) # only the primary component imports other components declared in the same fragment. this is to prevent confusion from gpt
    for comp, items in imports.items():
        if comp == component:
            for import_def in items:
                imports_txt += get_import_service_line(import_def, cur_path, root_path)
        else:
            is_declare = declare_or_use_comp_classifier.get_is_declared(full_title, comp)
            if is_declare:
                # only import components from the same folder if we are rendering the primary component, which uses the children
                if primary == component: # this is to prevent confusion from gpt, otherwise it starts using the parent component in the children
                    # another component declared in the same fragment, so import from local path
                    imports_txt += f"The component {comp}  from ./{comp}\n"
            else:
                items_path = os.path.join(root_path, items.strip())
                rel_path = os.path.relpath(items_path, cur_path)
                rel_path = os.path.join('.', rel_path)
                imports_txt += f"The component {comp} from {rel_path}\n"
    return imports_txt  


def get_to_render_and_used(title, components):
    to_render = []
    used = []
    for component in components:
        is_declare = declare_or_use_comp_classifier.get_is_declared(title, component)
        if is_declare:
            to_render.append(component)
        else:
            used.append(component)
    return to_render, used


def get_global_features(full_title, component):
    """
    returns the descriptions of all the features of the services that are declared to be globally used.
    This list of services is normally declared globally: each service says that it is to be used globally or not.
    Sometimes however, a component doesn't use a global service (ex: all components must use the dialogService to 
    log errors, but what if a component doesn't log errors?)
    The developer can overwrite the global behavior through the imports that are declared for the component: if a global
    service is not included in the list of imports, don't include it in the description either.
    """
    to_search = full_title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    result = ''
    services_to_include = []
    imports = resolve_component_imports.get_data(full_title)
    if imports:
        imports_for_comp = imports.get(component)
        if imports_for_comp:
            for import_def in imports_for_comp:
                service = import_def['service']
                services_to_include.append(service)
    for fragment in list_how_service_describes_components.text_fragments:
        if fragment.data:
            if result:
                result += '\n'
            for service, features in fragment.data.items():
                if service in services_to_include:
                    result += f'- {service}\n{features}'
    return result


def extract_service_interface_parts(code, fragment, rendered_comp):
    """
    extract the interface parts from the code and save them in the interface parts file
    """
    imports = resolve_component_imports.get_data(fragment.full_title)
    items = imports.get(rendered_comp)
    if items:
        for import_def in items:
            service = import_def['service']
            service_loc = import_def['service_loc']
            get_interface_parts.extract_interface_parts_for(service, service_loc, code, fragment.full_title)

def extract_used_comp_interface_parts(code, fragment, rendered_comp, used_comps):
    for comp in used_comps:
        comp_loc = declare_or_use_comp_classifier.get_declared_in(fragment.full_title, comp)
        get_interface_parts.extract_interface_parts_for(comp, comp_loc, code, fragment.full_title, True)


def get_description_and_interface_parts(fragment, component, to_render):
    if len(to_render) > 1:
        feature_description = component_descriptions_exact.get_description(fragment.full_title, component)
    else:
        feature_description = fragment.content
    interface_parts = get_interface_parts.get_interface_parts(fragment.full_title, component) # things this component needs to implement
    if interface_parts:
        to_join = []
        for key, value in interface_parts.items():
            to_join.append(f'{key}: {value}')
        interface_parts = f'\nMake certain that {component} has:\n- ' + '\n- '.join(to_join) + '\n'
    if not interface_parts:
        interface_parts = ''
    return feature_description, interface_parts


def format_interface(interface_def):
    result = ''
    for key, value in interface_def.items():
        result += f'- {key}: {value}\n'
    return result

def get_interface_parts_of_others(fragment, component):
    """
    goes over all the services that are imported by the component and builds up the interface from them so that 
    the entire application uses the same interface for the same service
    also goes over all the components that are used by the component and builds up the interface from them as well
    """
    result = ''
    imports = resolve_component_imports.get_data(fragment.full_title)
    if not imports:
        raise Exception(f'no imports found for {fragment.full_title}: need to run resolve_component_imports first')
    items = imports.get(component)
    if items:
        component_desc = component_descriptions_exact.get_description(fragment.full_title, component)
        for import_def in items:
            service = import_def['service']
            service_loc = import_def['service_loc']
            interface_def = get_interface_parts.get_interface_parts(service_loc, service)
            if interface_def:
                comp_and_global_service_desc = component_desc
                service_features_fragment = list_how_service_describes_components.get_data(service_loc)
                if service_features_fragment:
                    comp_and_global_service_desc += '\n'
                    for key, value in service_features_fragment.items():
                        comp_and_global_service_desc += f'\n{value}'

                        
                interface_def = format_interface(interface_def)
                interface = get_interface_parts_usage.list_used_interface_parts(service, service_loc, interface_def, comp_and_global_service_desc, fragment.full_title)
                if interface:
                    interface_list = []
                    interface_parts_def = get_interface_parts.get_interface_parts(service_loc, service)
                    for key, value in interface.items():
                        if key in interface_parts_def: # prefer the description that is extracted with get_interface_parts, gives better results (less chatter)
                            description = interface_parts_def[key]
                        else:
                            description = value
                        interface_list.append(f'{key}: {description}')
                    result += f'\n{service} has the following interface:\n- ' + '\n- '.join(interface_list) + '\n'
    
    for child_comp, path in imports.items():
        if child_comp != component:
            child_comp_loc = None
            is_declare = declare_or_use_comp_classifier.get_is_declared(fragment.full_title, child_comp)
            if is_declare:
                child_comp_loc = fragment.full_title
            else:
                child_comp_loc = declare_or_use_comp_classifier.get_declared_in(fragment.full_title, child_comp)
            props_def = list_component_props.get_props(child_comp_loc, child_comp)
            if props_def:
                props_lst = []
                for prop_name, prop_desc in props_def.items():
                    props_lst.append(f'{prop_name}: {prop_desc}')
                if props_lst:
                    result += f'\n{child_comp} has the following props:\n- ' + '\n- '.join(props_lst) + '\n'
    return result


def render_component(component, fragment, to_render, root_path, file_names):
    # calculate the path to the files we will generate cause we need it to get the import paths of the locally declared components
    title_to_path = fragment.full_title.replace(":", "").replace('?', '').replace('!', '')
    path_items = title_to_path.split(" > ")
    path_items = [part.replace(" ", "_") for part in path_items]
    path_items[0] = 'src' # the first item is the project name, we need to replace it with src so that the code gets rendered nicely
    path_section = os.path.join(root_path, *path_items)

    feature_description, interface_parts = get_description_and_interface_parts(fragment, component, to_render)
    others_interface_parts = get_interface_parts_of_others(fragment, component)
    global_features = get_global_features(fragment.full_title, component)

    if interface_parts and others_interface_parts:
        interface_parts += '\n'
    interface_parts += others_interface_parts

    params = {
        'component': component,
        'feature_title': fragment.title,
        'feature_description': feature_description,
        'dev_stack': project.fragments[1].content,
        'imports': get_all_imports(component, fragment.full_title, path_section, root_path),
        'interface_parts': interface_parts,
        'global_features': global_features,
    }
    response = generate_response(params, fragment.full_title)
    if response:
        # remove the code block markdown, the 3.5 version wants to add it itself
        response = response.strip() # need to remove the newline at the end
        if response.startswith("```javascript"):
            response = response[len("```javascript"):]
        if response.endswith("```"):
            response = response[:-len("```")]
        file_name = collect_response(component, response, path_section)
        file_names.append(file_name)
        return response
    

def process_data(root_path, writer): 
    for fragment in project.fragments:
        if ONLY_MISSING and has_fragment(fragment.full_title):
            continue
        components = component_lister.get_components(fragment.full_title)
        if len(components) > 0:
            file_names = [] # keep track of the file names generated for this fragment, so we can save it in the markdown file
            to_render, used = get_to_render_and_used(fragment.full_title, components)
            primary = primary_component.get_primary(fragment.full_title)
            if not primary:
                raise Exception('no primary found for ', fragment.full_title)
            # we presume that only the primary component is currently using the sub components and non of the subs use each other.
            # to save some cost & time. this means don't put too complex stuff in a single fragment
            primary_code = render_component(primary, fragment, to_render, root_path, file_names)
            extract_service_interface_parts(primary_code, fragment, primary)
            extract_used_comp_interface_parts(primary_code, fragment, primary, used)
            
            non_primary = [c for c in to_render if c != primary]
            for component in non_primary:
                get_interface_parts.extract_interface_parts_for(component, fragment.full_title, primary_code, fragment.full_title)
                code = render_component(component, fragment, to_render, root_path, file_names)
                extract_service_interface_parts(code, fragment, component)
            if file_names:
                collect_file_list(fragment.full_title, file_names, writer)
                    


def main(prompt, components_list, declare_or_use_list, comp_features_from_service, is_service_used, 
         is_service_singleton, imports, comp_desc, interface_parts, interface_parts_usage, primary_components, comp_props, root_path=None, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    component_lister.load_results(components_list)
    declare_or_use_comp_classifier.load_results(declare_or_use_list)
    # list_component_expansions.load_results(expansions)
    list_how_service_describes_components.load_results(comp_features_from_service)
    get_if_service_is_used.load_results(is_service_used)
    get_if_service_is_singleton.load_results(is_service_singleton)
    resolve_component_imports.load_results(imports)
    component_descriptions_exact.load_results(comp_desc)
    get_interface_parts.load_results(interface_parts)
    get_interface_parts_usage.load_results(interface_parts_usage)
    primary_component.load_results(primary_components)
    list_component_props.load_results(comp_props)

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
        
    open_mode = 'w'
    if ONLY_MISSING:
        load_results(file + "_component_files.md")
        open_mode = 'a'

    print("rendering results")

    # save there result to a file while rendering.
    if root_path is None:
        root_path = './'
    
    
    with open(file + "_component_files.md", open_mode) as writer:
        process_data(root_path, writer)
    
    print("done! check out the output file for the results!")


text_fragments = []  # the list of text fragments representing all the results that were rendered.

def load_results(filename, overwrite_file_name=None, overwrite=True):
    if not overwrite_file_name and overwrite:
        # modify the filename so that the filename without extension ends on _overwrite
        overwrite_file_name = filename.split('.')[0] + '_overwrite.' + filename.split('.')[1]
    result_loader.load(filename, text_fragments, True, overwrite_file_name)
        

def has_fragment(title):
    '''returns true if the title is in the list of fragments'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return True
    return False


def get_data(title):
    '''returns the list of components for the given title'''
    to_search = title.lower().strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title.lower() == to_search:
            return fragment.data or []
    return []    



if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 15:
        print("Please provide a prompt and a file containing the components to check")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        components_list = sys.argv[2]
        declare_or_use_list = sys.argv[3]
        comp_features_from_service = sys.argv[4]
        is_service_used = sys.argv[5]
        is_service_singleton = sys.argv[6]
        imports = sys.argv[7]
        comp_desc = sys.argv[8] 
        interface_parts = sys.argv[9]
        interface_parts_usage = sys.argv[10]
        primary_components = sys.argv[11]
        comp_props = sys.argv[12]  

    # Pull everything else as normal
    folder = sys.argv[13] if len(sys.argv) > 13 else None
    file = sys.argv[14] if len(sys.argv) > 14 else None

    # Run the main function
    main(prompt, components_list, declare_or_use_list, comp_features_from_service, is_service_used, 
         is_service_singleton, imports, comp_desc, interface_parts, interface_parts_usage, primary_components, comp_props, folder, file)
