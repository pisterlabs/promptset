import langchain
import inspect

def list_functions(root, obj):
    # Get all attributes of the object
    for name in dir(obj):
        # Skip special attributes
        if name.startswith('__') and name.endswith('__'):
            continue

        try:
            attribute = getattr(obj, name)
        except AttributeError:
            continue

        # If the attribute is a function or method, print its name
        if inspect.isfunction(attribute) or inspect.ismethod(attribute):
            print(root + '.' + name)

        # If the attribute is a class, recursively list its functions
        elif inspect.isclass(attribute):
            print(f"Class: {name}")
            list_functions(root + '.' + name, attribute)

# Start the process with the main SDK module
list_functions('langchain', langchain)
