# Default save DIR for prompts - do not change unless you know what you're doing
PROMPT_DIR = 'prompts/'

#NOT first message, here for convenience, not actually
FINAL_MESSAGE = """Create a Python file containing a function named 'merge_intervals'. This function should be designed to process a list of tuples, each representing an interval defined by a start and an end value (start, end). The primary objective of this function is to merge any overlapping intervals within this list.

In your implementation, focus on the following key aspects:

1. **Functionality**: The 'merge_intervals' function must accept an unsorted list of intervals and return a list of merged intervals. An overlap should be identified and merged if an interval (a, b) has any overlap with another interval (c, d), specifically when b >= c.

2. **Sorting Mechanism**: Ensure that the output list of merged intervals is sorted by their start times.

3. **Code Quality**:
   - **Readability**: Write the code in a clear and understandable manner, maintaining the high standard of readability as demonstrated in previous examples.
   - **Documentation**: Include a comprehensive docstring for the function, detailing its purpose, parameters, return type, and any edge cases or assumptions made. The documentation should be as thorough and clear as seen in earlier examples.
   - **Efficiency**: Optimize the function for performance, focusing on the efficiency of the merging process and any sorting algorithms used, following the efficiency standards of previous examples.

4. **Edge Case Handling**: Explicitly handle edge cases, such as an empty input list or intervals with invalid formats, as diligently as in past examples.

Please ensure that your code aligns with these quality standards, explicitly following the style of and adhering to the high quality of previous examples, and correctly implements the interval merging logic as specified."""

# Expert static message dictionary
EXPERT_DICT_MESSAGE = {
    "role": "user", 
    "content": "Thank, you expert. Please allow the next expert to continue on and provide either the next step, their feedback and code, or the completed code if you are all done.\n\nCarry on, next expert."
}

# BASE RECURSIVE IMPROVEMENT PROMPT
BASE_RECURSIVE_PROMPT = [
    {"role": "system", "content": "Respond with 'YES' if the code is assessed by experts \
     as complete, optimal, and incapable of further improvement by using context clues.\
     This includes complete functionality, optimal performance, and no suggestions for \
     additional changes. Any mention of continuing work, like 'moving onto Expert N', \
     should result in a 'NO' response, indicating further work is required.\n\nOnly ever respond with \
     YES or NO. Do not respond with anything else under any circumstances, and ahere \
     to the aforementioned criteria.\n\nYES or NO should be your only possible responses so \
     as to determine whether the content your user shares with you contains references to \
     potential improvements by so-called \"experts\"."}
]

# System Message (Unoptimized)
CODE_SYS_MESSAGE = """Welcome to our interactive guidance platform, where your quest for mastering the intricacies of the software development lifecycle, particularly within the Python ecosystem, is supported by the collective wisdom of three specialized experts. As you embark on this journey, you will be accompanied by:

EXPERT 1: I am an expert in Python programming best practices and code quality. My focus is on ensuring that your code adheres to PEP 8 style guidelines, enhancing readability and consistency. I will guide you through the nuances of writing clean, maintainable Python code that stands the test of time.

EXPERT 2: I am an expert in data structures, algorithms, and optimization. My role is to help you understand how to effectively utilize resources, construct performant solutions, and tackle the challenges of efficiency in coding. Together, we will delve into the art of crafting code that is not only functional but also optimized for peak performance.

EXPERT 3: I am an expert in software development methodologies and practices. From unit testing and TDD to error handling, debugging, and refactoring, my expertise covers the essential techniques that ensure your code is robust and reliable. I will assist you in implementing practices that contribute to a high-quality, understandable, and collaborative codebase.

As you navigate the complexities of software development, these experts will sequentially share their insights, building upon each other's knowledge to provide you with a comprehensive understanding of the topic at hand. They will proceed through a series of steps, each contributing their perspective before moving on to the next point of discussion. Should any expert recognize an oversight in their reasoning, they will gracefully exit the conversation, ensuring that you receive the most accurate and relevant information.

To facilitate your learning and problem-solving, consider the following approach:

1. Clearly define the problem or question you are facing in your software development process.
2. Expert 1 will initiate the discussion by providing insights related to Python programming best practices, considering the context of your problem.
3. After Expert 1's input, Expert 2 will analyze the problem from the perspective of data structures, algorithms, and optimization, building upon the previous step.
4. Expert 3 will then weigh in with their expertise on development methodologies and practices, further refining the solution.
5. You are encouraged to reflect on each expert's contribution, integrating their advice into your understanding of the problem.
6. Continue this iterative process, advancing through each step with the experts, until you reach a comprehensive solution.
7. If at any point new information arises or you encounter a stumbling block, revisit the relevant expert's guidance to realign your approach.

Remember, the goal is to synthesize the collective expertise of these three specialists to enhance your software development skills. By following this structured, collaborative approach, you will gain a deeper understanding of how to produce software that is not only functional but also secure, maintainable, and aligned with industry best practices. Whether you're dealing with version control, performance profiling, or object-oriented programming principles, this method will help you navigate the multifaceted landscape of software development with confidence and clarity."""

# Initial Static Question
USER_MESSAGE_ONE = """Experts, please come to a consensus on the following question:

Could you assist me in creating original, accurate, and efficient Python code that meets the highest standards of quality? I will supply a substantial dataset for reference. Please ensure that the code you develop reflects the dataset's structure without directly replicating any existing code. Aim to emulate the exemplary attributes of well-tested, high-quality Python code that adheres to best coding practices."""


# Next three are responses
ASSISTANT_MESSAGE_ONE = """As an expert in Python programming best practices and code quality, I will start by outlining some key principles that should guide the development of your Python code. These principles are in line with PEP 8 style guidelines and general best practices for writing clean and maintainable code:

1. **Code Formatting**: Ensure that your code follows PEP 8 style guidelines for formatting. This includes using 4 spaces for indentation, limiting lines to 79 characters, using blank lines to separate functions and classes, and other formatting conventions.

2. **Naming Conventions**: Use descriptive variable, function, and class names that follow the lowercase with underscores for variables and functions, and CapWords for classes.

3. **Docstrings and Comments**: Write clear docstrings for all modules, functions, classes, and methods. Comments should be used to explain non-obvious parts of the code.

4. **Modularity**: Break down your code into small, reusable functions and classes that do one thing and do it well.

5. **Error Handling**: Use try-except blocks to handle potential errors gracefully and provide informative error messages.

6. **Code Efficiency**: Avoid unnecessary computations and use built-in functions and data types where appropriate for better performance.

7. **Testing**: Write tests for your code using a framework like `unittest` or `pytest` to ensure functionality and to catch regressions.

8. **Version Control**: Use a version control system like Git to track changes, collaborate with others, and manage your codebase effectively.

With these principles in mind, let's move on to Expert 2 for insights on data structures, algorithms, and optimization."""

ASSISTANT_MESSAGE_TWO = """Building upon the foundational best practices outlined by Expert 1, let's focus on the aspects of data structures, algorithms, and optimization. Here are some guidelines to ensure your code is not only readable and maintainable but also performs efficiently:

1. **Appropriate Data Structures**: Choose the right data structures for your dataset and tasks. For example, use lists for ordered collections, sets for unique elements, and dictionaries for key-value pairs. Consider using more advanced structures like `collections.namedtuple` or `dataclasses` for structured data.

2. **Algorithmic Efficiency**: Be mindful of the time and space complexity of your algorithms. Use Big O notation to analyze and understand the performance implications of your code, especially in loops and recursive functions.

3. **Leverage Built-in Functions and Libraries**: Python's standard library and third-party libraries like NumPy and Pandas are optimized for performance. Use these libraries whenever possible instead of reinventing the wheel.

4. **Avoid Premature Optimization**: While it's important to write efficient code, don't optimize prematurely. First, make sure your code is correct and maintainable. Use profiling tools like `cProfile` or `timeit` to identify bottlenecks before optimizing.

5. **Iterators and Generators**: Use iterators and generators to handle large datasets efficiently. They can help you reduce memory usage by generating items on the fly rather than storing large collections in memory.

6. **Concurrency and Parallelism**: If your task is CPU-bound or I/O-bound and you need to improve performance, consider using concurrency with `asyncio` or parallelism with the `multiprocessing` or `threading` modules.

Now that we've covered code quality and performance considerations, Expert 3 will provide insights on development methodologies and practices to ensure robust and reliable code."""

ASSISTANT_MESSAGE_THREE = """To complement the advice provided by Experts 1 and 2 on code quality and optimization, I will focus on the methodologies and practices that will help you ensure your code is robust, reliable, and maintainable:

1. **Development Methodology**: Adopt a development methodology that suits your project's needs. Agile methodologies like Scrum or Kanban can help you manage tasks effectively and adapt to changes quickly.

2. **Version Control Best Practices**: Use branching strategies like Git Flow or GitHub Flow to organize your work. Commit often with meaningful commit messages and perform code reviews to maintain code quality.

3. **Unit Testing and Test-Driven Development (TDD)**: Write unit tests for your functions and classes to verify that they work as expected. Consider using TDD, where you write tests before implementing functionality, to ensure your code is testable and to reduce bugs.

4. **Continuous Integration (CI)**: Set up a CI pipeline to automatically run tests and other checks (like linting) whenever you push code changes. This helps catch issues early and ensures that your main branch remains stable.

5. **Error Handling and Logging**: Implement comprehensive error handling and use logging to record the program's execution flow. This will help with debugging and maintaining the code.

6. **Refactoring**: Regularly refactor your code to improve its structure and readability without changing its external behavior. This is crucial for maintaining a clean codebase as your project evolves.

7. **Documentation**: Maintain up-to-date documentation for your project. This includes not only code comments and docstrings but also higher-level documentation like READMEs, developer guides, and API documentation.

8. **Security Practices**: Follow security best practices to protect your code from vulnerabilities. This includes validating inputs, using secure coding patterns, and keeping dependencies up to date.

By integrating these methodologies and practices into your development process, you will create a codebase that is not only functional but also secure, maintainable, and aligned with industry standards. Now, let's synthesize the advice from all three experts to create a comprehensive approach to developing your Python code."""

# Code Base
CODE_BASE = """Thank you, experts. Here are labeled examples of Python code (some examples are longer than others). After analysis of the patterns and themes by you group of three experts, I'd like you to produce code of the same caliber based on my verbal instructions:
'./django/__main__.py':
```python
\"\"\"
Invokes django-admin when the django module is run as a script.

Example: python -m django check
\"\"\"
from django.core import management

if __name__ == "__main__":
    management.execute_from_command_line()```
'./django/apps/config.py':
```python
import inspect
import os
from importlib import import_module

from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string, module_has_submodule

APPS_MODULE_NAME = "apps"
MODELS_MODULE_NAME = "models"

class AppConfig:
    \"\"\"Class representing a Django application and its configuration.\"\"\"

    def __init__(self, app_name, app_module):
        # Full Python path to the application e.g. 'django.contrib.admin'.
        self.name = app_name

        # Root module for the application e.g. <module 'django.contrib.admin'
        # from 'django/contrib/admin/__init__.py'>.
        self.module = app_module

        # Reference to the Apps registry that holds this AppConfig. Set by the
        # registry when it registers the AppConfig instance.
        self.apps = None

        # The following attributes could be defined at the class level in a
        # subclass, hence the test-and-set pattern.

        # Last component of the Python path to the application e.g. 'admin'.
        # This value must be unique across a Django project.
        if not hasattr(self, "label"):
            self.label = app_name.rpartition(".")[2]
        if not self.label.isidentifier():
            raise ImproperlyConfigured(
                "The app label '%s' is not a valid Python identifier." % self.label
            )

        # Human-readable name for the application e.g. "Admin".
        if not hasattr(self, "verbose_name"):
            self.verbose_name = self.label.title()

        # Filesystem path to the application directory e.g.
        # '/path/to/django/contrib/admin'.
        if not hasattr(self, "path"):
            self.path = self._path_from_module(app_module)

        # Module containing models e.g. <module 'django.contrib.admin.models'
        # from 'django/contrib/admin/models.py'>. Set by import_models().
        # None if the application doesn't have a models module.
        self.models_module = None

        # Mapping of lowercase model names to model classes. Initially set to
        # None to prevent accidental access before import_models() runs.
        self.models = None

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.label)

    @cached_property
    def default_auto_field(self):
        from django.conf import settings

        return settings.DEFAULT_AUTO_FIELD

    @property
    def _is_default_auto_field_overridden(self):
        return self.__class__.default_auto_field is not AppConfig.default_auto_field

    def _path_from_module(self, module):
        \"\"\"Attempt to determine app's filesystem path from its module.\"\"\"
        # See #21874 for extended discussion of the behavior of this method in
        # various cases.
        # Convert to list because __path__ may not support indexing.
        paths = list(getattr(module, "__path__", []))
        if len(paths) != 1:
            filename = getattr(module, "__file__", None)
            if filename is not None:
                paths = [os.path.dirname(filename)]
            else:
                # For unknown reasons, sometimes the list returned by __path__
                # contains duplicates that must be removed (#25246).
                paths = list(set(paths))
        if len(paths) > 1:
            raise ImproperlyConfigured(
                "The app module %r has multiple filesystem locations (%r); "
                "you must configure this app with an AppConfig subclass "
                "with a 'path' class attribute." % (module, paths)
            )
        elif not paths:
            raise ImproperlyConfigured(
                "The app module %r has no filesystem location, "
                "you must configure this app with an AppConfig subclass "
                "with a 'path' class attribute." % module
            )
        return paths[0]

    @classmethod
    def create(cls, entry):
        \"\"\"
        Factory that creates an app config from an entry in INSTALLED_APPS.
        \"\"\"
        # create() eventually returns app_config_class(app_name, app_module).
        app_config_class = None
        app_name = None
        app_module = None

        # If import_module succeeds, entry points to the app module.
        try:
            app_module = import_module(entry)
        except Exception:
            pass
        else:
            # If app_module has an apps submodule that defines a single
            # AppConfig subclass, use it automatically.
            # To prevent this, an AppConfig subclass can declare a class
            # variable default = False.
            # If the apps module defines more than one AppConfig subclass,
            # the default one can declare default = True.
            if module_has_submodule(app_module, APPS_MODULE_NAME):
                mod_path = "%s.%s" % (entry, APPS_MODULE_NAME)
                mod = import_module(mod_path)
                # Check if there's exactly one AppConfig candidate,
                # excluding those that explicitly define default = False.
                app_configs = [
                    (name, candidate)
                    for name, candidate in inspect.getmembers(mod, inspect.isclass)
                    if (
                        issubclass(candidate, cls)
                        and candidate is not cls
                        and getattr(candidate, "default", True)
                    )
                ]
                if len(app_configs) == 1:
                    app_config_class = app_configs[0][1]
                else:
                    # Check if there's exactly one AppConfig subclass,
                    # among those that explicitly define default = True.
                    app_configs = [
                        (name, candidate)
                        for name, candidate in app_configs
                        if getattr(candidate, "default", False)
                    ]
                    if len(app_configs) > 1:
                        candidates = [repr(name) for name, _ in app_configs]
                        raise RuntimeError(
                            "%r declares more than one default AppConfig: "
                            "%s." % (mod_path, ", ".join(candidates))
                        )
                    elif len(app_configs) == 1:
                        app_config_class = app_configs[0][1]

            # Use the default app config class if we didn't find anything.
            if app_config_class is None:
                app_config_class = cls
                app_name = entry

        # If import_string succeeds, entry is an app config class.
        if app_config_class is None:
            try:
                app_config_class = import_string(entry)
            except Exception:
                pass
        # If both import_module and import_string failed, it means that entry
        # doesn't have a valid value.
        if app_module is None and app_config_class is None:
            # If the last component of entry starts with an uppercase letter,
            # then it was likely intended to be an app config class; if not,
            # an app module. Provide a nice error message in both cases.
            mod_path, _, cls_name = entry.rpartition(".")
            if mod_path and cls_name[0].isupper():
                # We could simply re-trigger the string import exception, but
                # we're going the extra mile and providing a better error
                # message for typos in INSTALLED_APPS.
                # This may raise ImportError, which is the best exception
                # possible if the module at mod_path cannot be imported.
                mod = import_module(mod_path)
                candidates = [
                    repr(name)
                    for name, candidate in inspect.getmembers(mod, inspect.isclass)
                    if issubclass(candidate, cls) and candidate is not cls
                ]
                msg = "Module '%s' does not contain a '%s' class." % (
                    mod_path,
                    cls_name,
                )
                if candidates:
                    msg += " Choices are: %s." % ", ".join(candidates)
                raise ImportError(msg)
            else:
                # Re-trigger the module import exception.
                import_module(entry)

        # Check for obvious errors. (This check prevents duck typing, but
        # it could be removed if it became a problem in practice.)
        if not issubclass(app_config_class, AppConfig):
            raise ImproperlyConfigured("'%s' isn't a subclass of AppConfig." % entry)

        # Obtain app name here rather than in AppClass.__init__ to keep
        # all error checking for entries in INSTALLED_APPS in one place.
        if app_name is None:
            try:
                app_name = app_config_class.name
            except AttributeError:
                raise ImproperlyConfigured("'%s' must supply a name attribute." % entry)

        # Ensure app_name points to a valid module.
        try:
            app_module = import_module(app_name)
        except ImportError:
            raise ImproperlyConfigured(
                "Cannot import '%s'. Check that '%s.%s.name' is correct."
                % (
                    app_name,
                    app_config_class.__module__,
                    app_config_class.__qualname__,
                )
            )

        # Entry is a path to an app config class.
        return app_config_class(app_name, app_module)

    def get_model(self, model_name, require_ready=True):
        \"\"\"
        Return the model with the given case-insensitive model_name.

        Raise LookupError if no model exists with this name.
        \"\"\"
        if require_ready:
            self.apps.check_models_ready()
        else:
            self.apps.check_apps_ready()
        try:
            return self.models[model_name.lower()]
        except KeyError:
            raise LookupError(
                "App '%s' doesn't have a '%s' model." % (self.label, model_name)
            )

    def get_models(self, include_auto_created=False, include_swapped=False):
        \"\"\"
        Return an iterable of models.

        By default, the following models aren't included:

        - auto-created models for many-to-many relations without
          an explicit intermediate table,
        - models that have been swapped out.

        Set the corresponding keyword argument to True to include such models.
        Keyword arguments aren't documented; they're a private API.
        \"\"\"
        self.apps.check_models_ready()
        for model in self.models.values():
            if model._meta.auto_created and not include_auto_created:
                continue
            if model._meta.swapped and not include_swapped:
                continue
            yield model

    def import_models(self):
        # Dictionary of models for this app, primarily maintained in the
        # 'all_models' attribute of the Apps this AppConfig is attached to.
        self.models = self.apps.all_models[self.label]

        if module_has_submodule(self.module, MODELS_MODULE_NAME):
            models_module_name = "%s.%s" % (self.name, MODELS_MODULE_NAME)
            self.models_module = import_module(models_module_name)

    def ready(self):
        \"\"\"
        Override this method in subclasses to run code when Django starts.
        \"\"\"```
'./django/apps/registry.py':
```python
import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial

from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured

from .config import AppConfig

class Apps:
    \"\"\"
    A registry that stores the configuration of installed applications.

    It also keeps track of models, e.g. to provide reverse relations.
    \"\"\"

    def __init__(self, installed_apps=()):
        # installed_apps is set to None when creating the main registry
        # because it cannot be populated at that point. Other registries must
        # provide a list of installed apps and are populated immediately.
        if installed_apps is None and hasattr(sys.modules[__name__], "apps"):
            raise RuntimeError("You must supply an installed_apps argument.")

        # Mapping of app labels => model names => model classes. Every time a
        # model is imported, ModelBase.__new__ calls apps.register_model which
        # creates an entry in all_models. All imported models are registered,
        # regardless of whether they're defined in an installed application
        # and whether the registry has been populated. Since it isn't possible
        # to reimport a module safely (it could reexecute initialization code)
        # all_models is never overridden or reset.
        self.all_models = defaultdict(dict)

        # Mapping of labels to AppConfig instances for installed apps.
        self.app_configs = {}

        # Stack of app_configs. Used to store the current state in
        # set_available_apps and set_installed_apps.
        self.stored_app_configs = []

        # Whether the registry is populated.
        self.apps_ready = self.models_ready = self.ready = False
        # For the autoreloader.
        self.ready_event = threading.Event()

        # Lock for thread-safe population.
        self._lock = threading.RLock()
        self.loading = False

        # Maps ("app_label", "modelname") tuples to lists of functions to be
        # called when the corresponding model is ready. Used by this class's
        # `lazy_model_operation()` and `do_pending_operations()` methods.
        self._pending_operations = defaultdict(list)

        # Populate apps and models, unless it's the main registry.
        if installed_apps is not None:
            self.populate(installed_apps)

    def populate(self, installed_apps=None):
        \"\"\"
        Load application configurations and models.

        Import each application module and then each model module.

        It is thread-safe and idempotent, but not reentrant.
        \"\"\"
        if self.ready:
            return

        # populate() might be called by two threads in parallel on servers
        # that create threads before initializing the WSGI callable.
        with self._lock:
            if self.ready:
                return

            # An RLock prevents other threads from entering this section. The
            # compare and set operation below is atomic.
            if self.loading:
                # Prevent reentrant calls to avoid running AppConfig.ready()
                # methods twice.
                raise RuntimeError("populate() isn't reentrant")
            self.loading = True

            # Phase 1: initialize app configs and import app modules.
            for entry in installed_apps:
                if isinstance(entry, AppConfig):
                    app_config = entry
                else:
                    app_config = AppConfig.create(entry)
                if app_config.label in self.app_configs:
                    raise ImproperlyConfigured(
                        "Application labels aren't unique, "
                        "duplicates: %s" % app_config.label
                    )

                self.app_configs[app_config.label] = app_config
                app_config.apps = self

            # Check for duplicate app names.
            counts = Counter(
                app_config.name for app_config in self.app_configs.values()
            )
            duplicates = [name for name, count in counts.most_common() if count > 1]
            if duplicates:
                raise ImproperlyConfigured(
                    "Application names aren't unique, "
                    "duplicates: %s" % ", ".join(duplicates)
                )

            self.apps_ready = True

            # Phase 2: import models modules.
            for app_config in self.app_configs.values():
                app_config.import_models()

            self.clear_cache()

            self.models_ready = True

            # Phase 3: run ready() methods of app configs.
            for app_config in self.get_app_configs():
                app_config.ready()

            self.ready = True
            self.ready_event.set()

    def check_apps_ready(self):
        \"\"\"Raise an exception if all apps haven't been imported yet.\"\"\"
        if not self.apps_ready:
            from django.conf import settings

            # If "not ready" is due to unconfigured settings, accessing
            # INSTALLED_APPS raises a more helpful ImproperlyConfigured
            # exception.
            settings.INSTALLED_APPS
            raise AppRegistryNotReady("Apps aren't loaded yet.")

    def check_models_ready(self):
        \"\"\"Raise an exception if all models haven't been imported yet.\"\"\"
        if not self.models_ready:
            raise AppRegistryNotReady("Models aren't loaded yet.")

    def get_app_configs(self):
        \"\"\"Import applications and return an iterable of app configs.\"\"\"
        self.check_apps_ready()
        return self.app_configs.values()

    def get_app_config(self, app_label):
        \"\"\"
        Import applications and returns an app config for the given label.

        Raise LookupError if no application exists with this label.
        \"\"\"
        self.check_apps_ready()
        try:
            return self.app_configs[app_label]
        except KeyError:
            message = "No installed app with label '%s'." % app_label
            for app_config in self.get_app_configs():
                if app_config.name == app_label:
                    message += " Did you mean '%s'?" % app_config.label
                    break
            raise LookupError(message)

    # This method is performance-critical at least for Django's test suite.
    @functools.cache
    def get_models(self, include_auto_created=False, include_swapped=False):
        \"\"\"
        Return a list of all installed models.

        By default, the following models aren't included:

        - auto-created models for many-to-many relations without
          an explicit intermediate table,
        - models that have been swapped out.

        Set the corresponding keyword argument to True to include such models.
        \"\"\"
        self.check_models_ready()

        result = []
        for app_config in self.app_configs.values():
            result.extend(app_config.get_models(include_auto_created, include_swapped))
        return result

    def get_model(self, app_label, model_name=None, require_ready=True):
        \"\"\"
        Return the model matching the given app_label and model_name.

        As a shortcut, app_label may be in the form <app_label>.<model_name>.

        model_name is case-insensitive.

        Raise LookupError if no application exists with this label, or no
        model exists with this name in the application. Raise ValueError if
        called with a single argument that doesn't contain exactly one dot.
        \"\"\"
        if require_ready:
            self.check_models_ready()
        else:
            self.check_apps_ready()

        if model_name is None:
            app_label, model_name = app_label.split(".")

        app_config = self.get_app_config(app_label)

        if not require_ready and app_config.models is None:
            app_config.import_models()

        return app_config.get_model(model_name, require_ready=require_ready)

    def register_model(self, app_label, model):
        # Since this method is called when models are imported, it cannot
        # perform imports because of the risk of import loops. It mustn't
        # call get_app_config().
        model_name = model._meta.model_name
        app_models = self.all_models[app_label]
        if model_name in app_models:
            if (
                model.__name__ == app_models[model_name].__name__
                and model.__module__ == app_models[model_name].__module__
            ):
                warnings.warn(
                    "Model '%s.%s' was already registered. Reloading models is not "
                    "advised as it can lead to inconsistencies, most notably with "
                    "related models." % (app_label, model_name),
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    "Conflicting '%s' models in application '%s': %s and %s."
                    % (model_name, app_label, app_models[model_name], model)
                )
        app_models[model_name] = model
        self.do_pending_operations(model)
        self.clear_cache()

    def is_installed(self, app_name):
        \"\"\"
        Check whether an application with this name exists in the registry.

        app_name is the full name of the app e.g. 'django.contrib.admin'.
        \"\"\"
        self.check_apps_ready()
        return any(ac.name == app_name for ac in self.app_configs.values())

    def get_containing_app_config(self, object_name):
        \"\"\"
        Look for an app config containing a given object.

        object_name is the dotted Python path to the object.

        Return the app config for the inner application in case of nesting.
        Return None if the object isn't in any registered app config.
        \"\"\"
        self.check_apps_ready()
        candidates = []
        for app_config in self.app_configs.values():
            if object_name.startswith(app_config.name):
                subpath = object_name.removeprefix(app_config.name)
                if subpath == "" or subpath[0] == ".":
                    candidates.append(app_config)
        if candidates:
            return sorted(candidates, key=lambda ac: -len(ac.name))[0]

    def get_registered_model(self, app_label, model_name):
        \"\"\"
        Similar to get_model(), but doesn't require that an app exists with
        the given app_label.

        It's safe to call this method at import time, even while the registry
        is being populated.
        \"\"\"
        model = self.all_models[app_label].get(model_name.lower())
        if model is None:
            raise LookupError("Model '%s.%s' not registered." % (app_label, model_name))
        return model

    @functools.cache
    def get_swappable_settings_name(self, to_string):
        \"\"\"
        For a given model string (e.g. "auth.User"), return the name of the
        corresponding settings name if it refers to a swappable model. If the
        referred model is not swappable, return None.

        This method is decorated with @functools.cache because it's performance
        critical when it comes to migrations. Since the swappable settings don't
        change after Django has loaded the settings, there is no reason to get
        the respective settings attribute over and over again.
        \"\"\"
        to_string = to_string.lower()
        for model in self.get_models(include_swapped=True):
            swapped = model._meta.swapped
            # Is this model swapped out for the model given by to_string?
            if swapped and swapped.lower() == to_string:
                return model._meta.swappable
            # Is this model swappable and the one given by to_string?
            if model._meta.swappable and model._meta.label_lower == to_string:
                return model._meta.swappable
        return None

    def set_available_apps(self, available):
        \"\"\"
        Restrict the set of installed apps used by get_app_config[s].

        available must be an iterable of application names.

        set_available_apps() must be balanced with unset_available_apps().

        Primarily used for performance optimization in TransactionTestCase.

        This method is safe in the sense that it doesn't trigger any imports.
        \"\"\"
        available = set(available)
        installed = {app_config.name for app_config in self.get_app_configs()}
        if not available.issubset(installed):
            raise ValueError(
                "Available apps isn't a subset of installed apps, extra apps: %s"
                % ", ".join(available - installed)
            )

        self.stored_app_configs.append(self.app_configs)
        self.app_configs = {
            label: app_config
            for label, app_config in self.app_configs.items()
            if app_config.name in available
        }
        self.clear_cache()

    def unset_available_apps(self):
        \"\"\"Cancel a previous call to set_available_apps().\"\"\"
        self.app_configs = self.stored_app_configs.pop()
        self.clear_cache()

    def set_installed_apps(self, installed):
        \"\"\"
        Enable a different set of installed apps for get_app_config[s].

        installed must be an iterable in the same format as INSTALLED_APPS.

        set_installed_apps() must be balanced with unset_installed_apps(),
        even if it exits with an exception.

        Primarily used as a receiver of the setting_changed signal in tests.

        This method may trigger new imports, which may add new models to the
        registry of all imported models. They will stay in the registry even
        after unset_installed_apps(). Since it isn't possible to replay
        imports safely (e.g. that could lead to registering listeners twice),
        models are registered when they're imported and never removed.
        \"\"\"
        if not self.ready:
            raise AppRegistryNotReady("App registry isn't ready yet.")
        self.stored_app_configs.append(self.app_configs)
        self.app_configs = {}
        self.apps_ready = self.models_ready = self.loading = self.ready = False
        self.clear_cache()
        self.populate(installed)

    def unset_installed_apps(self):
        \"\"\"Cancel a previous call to set_installed_apps().\"\"\"
        self.app_configs = self.stored_app_configs.pop()
        self.apps_ready = self.models_ready = self.ready = True
        self.clear_cache()

    def clear_cache(self):
        \"\"\"
        Clear all internal caches, for methods that alter the app registry.

        This is mostly used in tests.
        \"\"\"
        self.get_swappable_settings_name.cache_clear()
        # Call expire cache on each model. This will purge
        # the relation tree and the fields cache.
        self.get_models.cache_clear()
        if self.ready:
            # Circumvent self.get_models() to prevent that the cache is refilled.
            # This particularly prevents that an empty value is cached while cloning.
            for app_config in self.app_configs.values():
                for model in app_config.get_models(include_auto_created=True):
                    model._meta._expire_cache()

    def lazy_model_operation(self, function, *model_keys):
        \"\"\"
        Take a function and a number of ("app_label", "modelname") tuples, and
        when all the corresponding models have been imported and registered,
        call the function with the model classes as its arguments.

        The function passed to this method must accept exactly n models as
        arguments, where n=len(model_keys).
        \"\"\"
        # Base case: no arguments, just execute the function.
        if not model_keys:
            function()
        # Recursive case: take the head of model_keys, wait for the
        # corresponding model class to be imported and registered, then apply
        # that argument to the supplied function. Pass the resulting partial
        # to lazy_model_operation() along with the remaining model args and
        # repeat until all models are loaded and all arguments are applied.
        else:
            next_model, *more_models = model_keys

            # This will be executed after the class corresponding to next_model
            # has been imported and registered. The `func` attribute provides
            # duck-type compatibility with partials.
            def apply_next_model(model):
                next_function = partial(apply_next_model.func, model)
                self.lazy_model_operation(next_function, *more_models)

            apply_next_model.func = function

            # If the model has already been imported and registered, partially
            # apply it to the function now. If not, add it to the list of
            # pending operations for the model, where it will be executed with
            # the model class as its sole argument once the model is ready.
            try:
                model_class = self.get_registered_model(*next_model)
            except LookupError:
                self._pending_operations[next_model].append(apply_next_model)
            else:
                apply_next_model(model_class)

    def do_pending_operations(self, model):
        \"\"\"
        Take a newly-prepared model and pass it to each function waiting for
        it. This is called at the very end of Apps.register_model().
        \"\"\"
        key = model._meta.app_label, model._meta.model_name
        for function in self._pending_operations.pop(key, []):
            function(model)


apps = Apps(installed_apps=None)```
'./django/conf/global_settings.py':
```python
\"\"\"
Default Django settings. Override these with settings in the module pointed to
by the DJANGO_SETTINGS_MODULE environment variable.
\"\"\"


# This is defined here as a do-nothing function because we can't import
# django.utils.translation -- that module depends on the settings.
def gettext_noop(s):
    return s


####################
# CORE             #
####################

DEBUG = False

# Whether the framework should propagate raw exceptions rather than catching
# them. This is useful under some testing situations and should never be used
# on a live site.
DEBUG_PROPAGATE_EXCEPTIONS = False

# People who get code error notifications. In the format
# [('Full Name', 'email@example.com'), ('Full Name', 'anotheremail@example.com')]
ADMINS = []

# List of IP addresses, as strings, that:
#   * See debug comments, when DEBUG is true
#   * Receive x-headers
INTERNAL_IPS = []

# Hosts/domain names that are valid for this site.
# "*" matches anything, ".example.com" matches example.com and all subdomains
ALLOWED_HOSTS = []

# Local time zone for this installation. All choices can be found here:
# https://en.wikipedia.org/wiki/List_of_tz_zones_by_name (although not all
# systems may support all possibilities). When USE_TZ is True, this is
# interpreted as the default user time zone.
TIME_ZONE = "America/Chicago"

# If you set this to True, Django will use timezone-aware datetimes.
USE_TZ = True

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = "en-us"

# Languages we provide translations for, out of the box.
LANGUAGES = [
    ("af", gettext_noop("Afrikaans")),
    ("ar", gettext_noop("Arabic")),
    ("ar-dz", gettext_noop("Algerian Arabic")),
    ("ast", gettext_noop("Asturian")),
    ("az", gettext_noop("Azerbaijani")),
    ("bg", gettext_noop("Bulgarian")),
    ("be", gettext_noop("Belarusian")),
    ("bn", gettext_noop("Bengali")),
    ("br", gettext_noop("Breton")),
    ("bs", gettext_noop("Bosnian")),
    ("ca", gettext_noop("Catalan")),
    ("ckb", gettext_noop("Central Kurdish (Sorani)")),
    ("cs", gettext_noop("Czech")),
    ("cy", gettext_noop("Welsh")),
    ("da", gettext_noop("Danish")),
    ("de", gettext_noop("German")),
    ("dsb", gettext_noop("Lower Sorbian")),
    ("el", gettext_noop("Greek")),
    ("en", gettext_noop("English")),
    ("en-au", gettext_noop("Australian English")),
    ("en-gb", gettext_noop("British English")),
    ("eo", gettext_noop("Esperanto")),
    ("es", gettext_noop("Spanish")),
    ("es-ar", gettext_noop("Argentinian Spanish")),
    ("es-co", gettext_noop("Colombian Spanish")),
    ("es-mx", gettext_noop("Mexican Spanish")),
    ("es-ni", gettext_noop("Nicaraguan Spanish")),
    ("es-ve", gettext_noop("Venezuelan Spanish")),
    ("et", gettext_noop("Estonian")),
    ("eu", gettext_noop("Basque")),
    ("fa", gettext_noop("Persian")),
    ("fi", gettext_noop("Finnish")),
    ("fr", gettext_noop("French")),
    ("fy", gettext_noop("Frisian")),
    ("ga", gettext_noop("Irish")),
    ("gd", gettext_noop("Scottish Gaelic")),
    ("gl", gettext_noop("Galician")),
    ("he", gettext_noop("Hebrew")),
    ("hi", gettext_noop("Hindi")),
    ("hr", gettext_noop("Croatian")),
    ("hsb", gettext_noop("Upper Sorbian")),
    ("hu", gettext_noop("Hungarian")),
    ("hy", gettext_noop("Armenian")),
    ("ia", gettext_noop("Interlingua")),
    ("id", gettext_noop("Indonesian")),
    ("ig", gettext_noop("Igbo")),
    ("io", gettext_noop("Ido")),
    ("is", gettext_noop("Icelandic")),
    ("it", gettext_noop("Italian")),
    ("ja", gettext_noop("Japanese")),
    ("ka", gettext_noop("Georgian")),
    ("kab", gettext_noop("Kabyle")),
    ("kk", gettext_noop("Kazakh")),
    ("km", gettext_noop("Khmer")),
    ("kn", gettext_noop("Kannada")),
    ("ko", gettext_noop("Korean")),
    ("ky", gettext_noop("Kyrgyz")),
    ("lb", gettext_noop("Luxembourgish")),
    ("lt", gettext_noop("Lithuanian")),
    ("lv", gettext_noop("Latvian")),
    ("mk", gettext_noop("Macedonian")),
    ("ml", gettext_noop("Malayalam")),
    ("mn", gettext_noop("Mongolian")),
    ("mr", gettext_noop("Marathi")),
    ("ms", gettext_noop("Malay")),
    ("my", gettext_noop("Burmese")),
    ("nb", gettext_noop("Norwegian Bokmål")),
    ("ne", gettext_noop("Nepali")),
    ("nl", gettext_noop("Dutch")),
    ("nn", gettext_noop("Norwegian Nynorsk")),
    ("os", gettext_noop("Ossetic")),
    ("pa", gettext_noop("Punjabi")),
    ("pl", gettext_noop("Polish")),
    ("pt", gettext_noop("Portuguese")),
    ("pt-br", gettext_noop("Brazilian Portuguese")),
    ("ro", gettext_noop("Romanian")),
    ("ru", gettext_noop("Russian")),
    ("sk", gettext_noop("Slovak")),
    ("sl", gettext_noop("Slovenian")),
    ("sq", gettext_noop("Albanian")),
    ("sr", gettext_noop("Serbian")),
    ("sr-latn", gettext_noop("Serbian Latin")),
    ("sv", gettext_noop("Swedish")),
    ("sw", gettext_noop("Swahili")),
    ("ta", gettext_noop("Tamil")),
    ("te", gettext_noop("Telugu")),
    ("tg", gettext_noop("Tajik")),
    ("th", gettext_noop("Thai")),
    ("tk", gettext_noop("Turkmen")),
    ("tr", gettext_noop("Turkish")),
    ("tt", gettext_noop("Tatar")),
    ("udm", gettext_noop("Udmurt")),
    ("ug", gettext_noop("Uyghur")),
    ("uk", gettext_noop("Ukrainian")),
    ("ur", gettext_noop("Urdu")),
    ("uz", gettext_noop("Uzbek")),
    ("vi", gettext_noop("Vietnamese")),
    ("zh-hans", gettext_noop("Simplified Chinese")),
    ("zh-hant", gettext_noop("Traditional Chinese")),
]

# Languages using BiDi (right-to-left) layout
LANGUAGES_BIDI = ["he", "ar", "ar-dz", "ckb", "fa", "ug", "ur"]

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True
LOCALE_PATHS = []

# Settings for language cookie
LANGUAGE_COOKIE_NAME = "django_language"
LANGUAGE_COOKIE_AGE = None
LANGUAGE_COOKIE_DOMAIN = None
LANGUAGE_COOKIE_PATH = "/"
LANGUAGE_COOKIE_SECURE = False
LANGUAGE_COOKIE_HTTPONLY = False
LANGUAGE_COOKIE_SAMESITE = None

# Not-necessarily-technical managers of the site. They get broken link
# notifications and other various emails.
MANAGERS = ADMINS

# Default charset to use for all HttpResponse objects, if a MIME type isn't
# manually specified. It's used to construct the Content-Type header.
DEFAULT_CHARSET = "utf-8"

# Email address that error messages come from.
SERVER_EMAIL = "root@localhost"

# Database connection info. If left empty, will default to the dummy backend.
DATABASES = {}

# Classes used to implement DB routing behavior.
DATABASE_ROUTERS = []

# The email backend to use. For possible shortcuts see django.core.mail.
# The default is to use the SMTP backend.
# Third-party backends can be specified by providing a Python path
# to a module that defines an EmailBackend class.
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

# Host for sending email.
EMAIL_HOST = "localhost"

# Port for sending email.
EMAIL_PORT = 25

# Whether to send SMTP 'Date' header in the local time zone or in UTC.
EMAIL_USE_LOCALTIME = False

# Optional SMTP authentication information for EMAIL_HOST.
EMAIL_HOST_USER = ""
EMAIL_HOST_PASSWORD = ""
EMAIL_USE_TLS = False
EMAIL_USE_SSL = False
EMAIL_SSL_CERTFILE = None
EMAIL_SSL_KEYFILE = None
EMAIL_TIMEOUT = None

# List of strings representing installed apps.
INSTALLED_APPS = []

TEMPLATES = []

# Default form rendering class.
FORM_RENDERER = "django.forms.renderers.DjangoTemplates"

# RemovedInDjango60Warning: It's a transitional setting helpful in early
# adoption of "https" as the new default value of forms.URLField.assume_scheme.
# Set to True to assume "https" during the Django 5.x release cycle.
FORMS_URLFIELD_ASSUME_HTTPS = False

# Default email address to use for various automated correspondence from
# the site managers.
DEFAULT_FROM_EMAIL = "webmaster@localhost"

# Subject-line prefix for email messages send with django.core.mail.mail_admins
# or ...mail_managers.  Make sure to include the trailing space.
EMAIL_SUBJECT_PREFIX = "[Django] "

# Whether to append trailing slashes to URLs.
APPEND_SLASH = True

# Whether to prepend the "www." subdomain to URLs that don't have it.
PREPEND_WWW = False

# Override the server-derived value of SCRIPT_NAME
FORCE_SCRIPT_NAME = None

# List of compiled regular expression objects representing User-Agent strings
# that are not allowed to visit any page, systemwide. Use this for bad
# robots/crawlers. Here are a few examples:
#     import re
#     DISALLOWED_USER_AGENTS = [
#         re.compile(r'^NaverBot.*'),
#         re.compile(r'^EmailSiphon.*'),
#         re.compile(r'^SiteSucker.*'),
#         re.compile(r'^sohu-search'),
#     ]
DISALLOWED_USER_AGENTS = []

ABSOLUTE_URL_OVERRIDES = {}

# List of compiled regular expression objects representing URLs that need not
# be reported by BrokenLinkEmailsMiddleware. Here are a few examples:
#    import re
#    IGNORABLE_404_URLS = [
#        re.compile(r'^/apple-touch-icon.*\.png$'),
#        re.compile(r'^/favicon.ico$'),
#        re.compile(r'^/robots.txt$'),
#        re.compile(r'^/phpmyadmin/'),
#        re.compile(r'\.(cgi|php|pl)$'),
#    ]
IGNORABLE_404_URLS = []

# A secret key for this particular Django installation. Used in secret-key
# hashing algorithms. Set this in your settings, or Django will complain
# loudly.
SECRET_KEY = ""

# List of secret keys used to verify the validity of signatures. This allows
# secret key rotation.
SECRET_KEY_FALLBACKS = []

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/var/www/example.com/media/"
MEDIA_ROOT = ""

# URL that handles the media served from MEDIA_ROOT.
# Examples: "http://example.com/media/", "http://media.example.com/"
MEDIA_URL = ""

# Absolute path to the directory static files should be collected to.
# Example: "/var/www/example.com/static/"
STATIC_ROOT = None

# URL that handles the static files served from STATIC_ROOT.
# Example: "http://example.com/static/", "http://static.example.com/"
STATIC_URL = None

# List of upload handler classes to be applied in order.
FILE_UPLOAD_HANDLERS = [
    "django.core.files.uploadhandler.MemoryFileUploadHandler",
    "django.core.files.uploadhandler.TemporaryFileUploadHandler",
]

# Maximum size, in bytes, of a request before it will be streamed to the
# file system instead of into memory.
FILE_UPLOAD_MAX_MEMORY_SIZE = 2621440  # i.e. 2.5 MB

# Maximum size in bytes of request data (excluding file uploads) that will be
# read before a SuspiciousOperation (RequestDataTooBig) is raised.
DATA_UPLOAD_MAX_MEMORY_SIZE = 2621440  # i.e. 2.5 MB

# Maximum number of GET/POST parameters that will be read before a
# SuspiciousOperation (TooManyFieldsSent) is raised.
DATA_UPLOAD_MAX_NUMBER_FIELDS = 1000

# Maximum number of files encoded in a multipart upload that will be read
# before a SuspiciousOperation (TooManyFilesSent) is raised.
DATA_UPLOAD_MAX_NUMBER_FILES = 100

# Directory in which upload streamed files will be temporarily saved. A value of
# `None` will make Django use the operating system's default temporary directory
# (i.e. "/tmp" on *nix systems).
FILE_UPLOAD_TEMP_DIR = None

# The numeric mode to set newly-uploaded files to. The value should be a mode
# you'd pass directly to os.chmod; see
# https://docs.python.org/library/os.html#files-and-directories.
FILE_UPLOAD_PERMISSIONS = 0o644

# The numeric mode to assign to newly-created directories, when uploading files.
# The value should be a mode as you'd pass to os.chmod;
# see https://docs.python.org/library/os.html#files-and-directories.
FILE_UPLOAD_DIRECTORY_PERMISSIONS = None

# Python module path where user will place custom format definition.
# The directory where this setting is pointing should contain subdirectories
# named as the locales, containing a formats.py file
# (i.e. "myproject.locale" for myproject/locale/en/formats.py etc. use)
FORMAT_MODULE_PATH = None

# Default formatting for date objects. See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
DATE_FORMAT = "N j, Y"

# Default formatting for datetime objects. See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
DATETIME_FORMAT = "N j, Y, P"

# Default formatting for time objects. See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
TIME_FORMAT = "P"

# Default formatting for date objects when only the year and month are relevant.
# See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
YEAR_MONTH_FORMAT = "F Y"

# Default formatting for date objects when only the month and day are relevant.
# See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
MONTH_DAY_FORMAT = "F j"

# Default short formatting for date objects. See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
SHORT_DATE_FORMAT = "m/d/Y"

# Default short formatting for datetime objects.
# See all available format strings here:
# https://docs.djangoproject.com/en/dev/ref/templates/builtins/#date
SHORT_DATETIME_FORMAT = "m/d/Y P"

# Default formats to be used when parsing dates from input boxes, in order
# See all available format string here:
# https://docs.python.org/library/datetime.html#strftime-behavior
# * Note that these format strings are different from the ones to display dates
DATE_INPUT_FORMATS = [
    "%Y-%m-%d",  # '2006-10-25'
    "%m/%d/%Y",  # '10/25/2006'
    "%m/%d/%y",  # '10/25/06'
    "%b %d %Y",  # 'Oct 25 2006'
    "%b %d, %Y",  # 'Oct 25, 2006'
    "%d %b %Y",  # '25 Oct 2006'
    "%d %b, %Y",  # '25 Oct, 2006'
    "%B %d %Y",  # 'October 25 2006'
    "%B %d, %Y",  # 'October 25, 2006'
    "%d %B %Y",  # '25 October 2006'
    "%d %B, %Y",  # '25 October, 2006'
]

# Default formats to be used when parsing times from input boxes, in order
# See all available format string here:
# https://docs.python.org/library/datetime.html#strftime-behavior
# * Note that these format strings are different from the ones to display dates
TIME_INPUT_FORMATS = [
    "%H:%M:%S",  # '14:30:59'
    "%H:%M:%S.%f",  # '14:30:59.000200'
    "%H:%M",  # '14:30'
]

# Default formats to be used when parsing dates and times from input boxes,
# in order
# See all available format string here:
# https://docs.python.org/library/datetime.html#strftime-behavior
# * Note that these format strings are different from the ones to display dates
DATETIME_INPUT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",  # '2006-10-25 14:30:59'
    "%Y-%m-%d %H:%M:%S.%f",  # '2006-10-25 14:30:59.000200'
    "%Y-%m-%d %H:%M",  # '2006-10-25 14:30'
    "%m/%d/%Y %H:%M:%S",  # '10/25/2006 14:30:59'
    "%m/%d/%Y %H:%M:%S.%f",  # '10/25/2006 14:30:59.000200'
    "%m/%d/%Y %H:%M",  # '10/25/2006 14:30'
    "%m/%d/%y %H:%M:%S",  # '10/25/06 14:30:59'
    "%m/%d/%y %H:%M:%S.%f",  # '10/25/06 14:30:59.000200'
    "%m/%d/%y %H:%M",  # '10/25/06 14:30'
]

# First day of week, to be used on calendars
# 0 means Sunday, 1 means Monday...
FIRST_DAY_OF_WEEK = 0

# Decimal separator symbol
DECIMAL_SEPARATOR = "."

# Boolean that sets whether to add thousand separator when formatting numbers
USE_THOUSAND_SEPARATOR = False

# Number of digits that will be together, when splitting them by
# THOUSAND_SEPARATOR. 0 means no grouping, 3 means splitting by thousands...
NUMBER_GROUPING = 0

# Thousand separator symbol
THOUSAND_SEPARATOR = ","

# The tablespaces to use for each model when not specified otherwise.
DEFAULT_TABLESPACE = ""
DEFAULT_INDEX_TABLESPACE = ""

# Default primary key field type.
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

# Default X-Frame-Options header value
X_FRAME_OPTIONS = "DENY"

USE_X_FORWARDED_HOST = False
USE_X_FORWARDED_PORT = False

# The Python dotted path to the WSGI application that Django's internal server
# (runserver) will use. If `None`, the return value of
# 'django.core.wsgi.get_wsgi_application' is used, thus preserving the same
# behavior as previous versions of Django. Otherwise this should point to an
# actual WSGI application object.
WSGI_APPLICATION = None

# If your Django app is behind a proxy that sets a header to specify secure
# connections, AND that proxy ensures that user-submitted headers with the
# same name are ignored (so that people can't spoof it), set this value to
# a tuple of (header_name, header_value). For any requests that come in with
# that header/value, request.is_secure() will return True.
# WARNING! Only set this if you fully understand what you're doing. Otherwise,
# you may be opening yourself up to a security risk.
SECURE_PROXY_SSL_HEADER = None

# List of middleware to use. Order is important; in the request phase, these
# middleware will be applied in the order given, and in the response
# phase the middleware will be applied in reverse order.
MIDDLEWARE = []

# Cache to store session data if using the cache session backend.
SESSION_CACHE_ALIAS = "default"
# Cookie name. This can be whatever you want.
SESSION_COOKIE_NAME = "sessionid"
# Age of cookie, in seconds (default: 2 weeks).
SESSION_COOKIE_AGE = 60 * 60 * 24 * 7 * 2
# A string like "example.com", or None for standard domain cookie.
SESSION_COOKIE_DOMAIN = None
# Whether the session cookie should be secure (https:// only).
SESSION_COOKIE_SECURE = False
# The path of the session cookie.
SESSION_COOKIE_PATH = "/"
# Whether to use the HttpOnly flag.
SESSION_COOKIE_HTTPONLY = True
# Whether to set the flag restricting cookie leaks on cross-site requests.
# This can be 'Lax', 'Strict', 'None', or False to disable the flag.
SESSION_COOKIE_SAMESITE = "Lax"
# Whether to save the session data on every request.
SESSION_SAVE_EVERY_REQUEST = False
# Whether a user's session cookie expires when the web browser is closed.
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
# The module to store session data
SESSION_ENGINE = "django.contrib.sessions.backends.db"
# Directory to store session files if using the file session module. If None,
# the backend will use a sensible default.
SESSION_FILE_PATH = None
# class to serialize session data
SESSION_SERIALIZER = "django.contrib.sessions.serializers.JSONSerializer"

# The cache backends to use.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}
CACHE_MIDDLEWARE_KEY_PREFIX = ""
CACHE_MIDDLEWARE_SECONDS = 600
CACHE_MIDDLEWARE_ALIAS = "default"

AUTH_USER_MODEL = "auth.User"

AUTHENTICATION_BACKENDS = ["django.contrib.auth.backends.ModelBackend"]

LOGIN_URL = "/accounts/login/"

LOGIN_REDIRECT_URL = "/accounts/profile/"

LOGOUT_REDIRECT_URL = None

# The number of seconds a password reset link is valid for (default: 3 days).
PASSWORD_RESET_TIMEOUT = 60 * 60 * 24 * 3

# the first hasher in this list is the preferred algorithm.  any
# password using different algorithms will be converted automatically
# upon login
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
    "django.contrib.auth.hashers.ScryptPasswordHasher",
]

AUTH_PASSWORD_VALIDATORS = []

SIGNING_BACKEND = "django.core.signing.TimestampSigner"

# Dotted path to callable to be used as view when a request is
# rejected by the CSRF middleware.
CSRF_FAILURE_VIEW = "django.views.csrf.csrf_failure"

# Settings for CSRF cookie.
CSRF_COOKIE_NAME = "csrftoken"
CSRF_COOKIE_AGE = 60 * 60 * 24 * 7 * 52
CSRF_COOKIE_DOMAIN = None
CSRF_COOKIE_PATH = "/"
CSRF_COOKIE_SECURE = False
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_HEADER_NAME = "HTTP_X_CSRFTOKEN"
CSRF_TRUSTED_ORIGINS = []
CSRF_USE_SESSIONS = False

# Class to use as messages backend
MESSAGE_STORAGE = "django.contrib.messages.storage.fallback.FallbackStorage"

# Default values of MESSAGE_LEVEL and MESSAGE_TAGS are defined within
# django.contrib.messages to avoid imports in this settings file.

# The callable to use to configure logging
LOGGING_CONFIG = "logging.config.dictConfig"

# Custom logging configuration.
LOGGING = {}

# Default exception reporter class used in case none has been
# specifically assigned to the HttpRequest instance.
DEFAULT_EXCEPTION_REPORTER = "django.views.debug.ExceptionReporter"

# Default exception reporter filter class used in case none has been
# specifically assigned to the HttpRequest instance.
DEFAULT_EXCEPTION_REPORTER_FILTER = "django.views.debug.SafeExceptionReporterFilter"

# The name of the class to use to run the test suite
TEST_RUNNER = "django.test.runner.DiscoverRunner"

# Apps that don't need to be serialized at test database creation time
# (only apps with migrations are to start with)
TEST_NON_SERIALIZED_APPS = []

# The list of directories to search for fixtures
FIXTURE_DIRS = []

# A list of locations of additional static files
STATICFILES_DIRS = []

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
    # 'django.contrib.staticfiles.finders.DefaultStorageFinder',
]

# Migration module overrides for apps, by app label.
MIGRATION_MODULES = {}

# List of all issues generated by system checks that should be silenced. Light
# issues like warnings, infos or debugs will not generate a message. Silencing
# serious issues like errors and criticals does not result in hiding the
# message, but Django will not stop you from e.g. running server.
SILENCED_SYSTEM_CHECKS = []

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin"
SECURE_HSTS_INCLUDE_SUBDOMAINS = False
SECURE_HSTS_PRELOAD = False
SECURE_HSTS_SECONDS = 0
SECURE_REDIRECT_EXEMPT = []
SECURE_REFERRER_POLICY = "same-origin"
SECURE_SSL_HOST = None
SECURE_SSL_REDIRECT = False```
'./django/conf/urls/i18n.py':
```python
import functools

from django.conf import settings
from django.urls import LocalePrefixPattern, URLResolver, get_resolver, path
from django.views.i18n import set_language

def i18n_patterns(*urls, prefix_default_language=True):
    \"\"\"
    Add the language code prefix to every URL pattern within this function.
    This may only be used in the root URLconf, not in an included URLconf.
    \"\"\"
    if not settings.USE_I18N:
        return list(urls)
    return [
        URLResolver(
            LocalePrefixPattern(prefix_default_language=prefix_default_language),
            list(urls),
        )
    ]


@functools.cache
def is_language_prefix_patterns_used(urlconf):
    \"\"\"
    Return a tuple of two booleans: (
        `True` if i18n_patterns() (LocalePrefixPattern) is used in the URLconf,
        `True` if the default language should be prefixed
    )
    \"\"\"
    for url_pattern in get_resolver(urlconf).url_patterns:
        if isinstance(url_pattern.pattern, LocalePrefixPattern):
            return True, url_pattern.pattern.prefix_default_language
    return False, False


urlpatterns = [
    path("setlang/", set_language, name="set_language"),
]
```"""

CODE_BASE_2 = """CODE_BASE = Thank you, experts. Here are labeled examples of Python code (some examples are longer than others). After analysis of the patterns and themes by you group of three experts, I'd like you to produce code of the same caliber based on my verbal instructions:
'./django/__main__.py':
```python
\"\"\"
Invokes django-admin when the django module is run as a script.

Example: python -m django check
\"\"\"
from django.core import management

if __name__ == "__main__":
    management.execute_from_command_line()
```
"""

CODE_BASE_MEDIUM = """Thank you, experts. Here are labeled examples of Python code (some examples are longer than others). After analysis of the patterns and themes by you group of three experts, I'd like you to produce code of the same caliber based on my verbal instructions:
'./django/__main__.py':
```python
\"\"\"
Invokes django-admin when the django module is run as a script.

Example: python -m django check
\"\"\"
from django.core import management

if __name__ == "__main__":
    management.execute_from_command_line()```
'./django/apps/config.py':
```python
import inspect
import os
from importlib import import_module

from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string, module_has_submodule

APPS_MODULE_NAME = "apps"
MODELS_MODULE_NAME = "models"


class AppConfig:
    \"\"\"Class representing a Django application and its configuration.\"\"\"

    def __init__(self, app_name, app_module):
        # Full Python path to the application e.g. 'django.contrib.admin'.
        self.name = app_name

        # Root module for the application e.g. <module 'django.contrib.admin'
        # from 'django/contrib/admin/__init__.py'>.
        self.module = app_module

        # Reference to the Apps registry that holds this AppConfig. Set by the
        # registry when it registers the AppConfig instance.
        self.apps = None

        # The following attributes could be defined at the class level in a
        # subclass, hence the test-and-set pattern.

        # Last component of the Python path to the application e.g. 'admin'.
        # This value must be unique across a Django project.
        if not hasattr(self, "label"):
            self.label = app_name.rpartition(".")[2]
        if not self.label.isidentifier():
            raise ImproperlyConfigured(
                "The app label '%s' is not a valid Python identifier." % self.label
            )

        # Human-readable name for the application e.g. "Admin".
        if not hasattr(self, "verbose_name"):
            self.verbose_name = self.label.title()

        # Filesystem path to the application directory e.g.
        # '/path/to/django/contrib/admin'.
        if not hasattr(self, "path"):
            self.path = self._path_from_module(app_module)

        # Module containing models e.g. <module 'django.contrib.admin.models'
        # from 'django/contrib/admin/models.py'>. Set by import_models().
        # None if the application doesn't have a models module.
        self.models_module = None

        # Mapping of lowercase model names to model classes. Initially set to
        # None to prevent accidental access before import_models() runs.
        self.models = None

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.label)

    @cached_property
    def default_auto_field(self):
        from django.conf import settings

        return settings.DEFAULT_AUTO_FIELD

    @property
    def _is_default_auto_field_overridden(self):
        return self.__class__.default_auto_field is not AppConfig.default_auto_field

    def _path_from_module(self, module):
        \"\"\"Attempt to determine app's filesystem path from its module.\"\"\"
        # See #21874 for extended discussion of the behavior of this method in
        # various cases.
        # Convert to list because __path__ may not support indexing.
        paths = list(getattr(module, "__path__", []))
        if len(paths) != 1:
            filename = getattr(module, "__file__", None)
            if filename is not None:
                paths = [os.path.dirname(filename)]
            else:
                # For unknown reasons, sometimes the list returned by __path__
                # contains duplicates that must be removed (#25246).
                paths = list(set(paths))
        if len(paths) > 1:
            raise ImproperlyConfigured(
                "The app module %r has multiple filesystem locations (%r); "
                "you must configure this app with an AppConfig subclass "
                "with a 'path' class attribute." % (module, paths)
            )
        elif not paths:
            raise ImproperlyConfigured(
                "The app module %r has no filesystem location, "
                "you must configure this app with an AppConfig subclass "
                "with a 'path' class attribute." % module
            )
        return paths[0]

    @classmethod
    def create(cls, entry):
        \"\"\"
        Factory that creates an app config from an entry in INSTALLED_APPS.
        \"\"\"
        # create() eventually returns app_config_class(app_name, app_module).
        app_config_class = None
        app_name = None
        app_module = None

        # If import_module succeeds, entry points to the app module.
        try:
            app_module = import_module(entry)
        except Exception:
            pass
        else:
            # If app_module has an apps submodule that defines a single
            # AppConfig subclass, use it automatically.
            # To prevent this, an AppConfig subclass can declare a class
            # variable default = False.
            # If the apps module defines more than one AppConfig subclass,
            # the default one can declare default = True.
            if module_has_submodule(app_module, APPS_MODULE_NAME):
                mod_path = "%s.%s" % (entry, APPS_MODULE_NAME)
                mod = import_module(mod_path)
                # Check if there's exactly one AppConfig candidate,
                # excluding those that explicitly define default = False.
                app_configs = [
                    (name, candidate)
                    for name, candidate in inspect.getmembers(mod, inspect.isclass)
                    if (
                        issubclass(candidate, cls)
                        and candidate is not cls
                        and getattr(candidate, "default", True)
                    )
                ]
                if len(app_configs) == 1:
                    app_config_class = app_configs[0][1]
                else:
                    # Check if there's exactly one AppConfig subclass,
                    # among those that explicitly define default = True.
                    app_configs = [
                        (name, candidate)
                        for name, candidate in app_configs
                        if getattr(candidate, "default", False)
                    ]
                    if len(app_configs) > 1:
                        candidates = [repr(name) for name, _ in app_configs]
                        raise RuntimeError(
                            "%r declares more than one default AppConfig: "
                            "%s." % (mod_path, ", ".join(candidates))
                        )
                    elif len(app_configs) == 1:
                        app_config_class = app_configs[0][1]

            # Use the default app config class if we didn't find anything.
            if app_config_class is None:
                app_config_class = cls
                app_name = entry

        # If import_string succeeds, entry is an app config class.
        if app_config_class is None:
            try:
                app_config_class = import_string(entry)
            except Exception:
                pass
        # If both import_module and import_string failed, it means that entry
        # doesn't have a valid value.
        if app_module is None and app_config_class is None:
            # If the last component of entry starts with an uppercase letter,
            # then it was likely intended to be an app config class; if not,
            # an app module. Provide a nice error message in both cases.
            mod_path, _, cls_name = entry.rpartition(".")
            if mod_path and cls_name[0].isupper():
                # We could simply re-trigger the string import exception, but
                # we're going the extra mile and providing a better error
                # message for typos in INSTALLED_APPS.
                # This may raise ImportError, which is the best exception
                # possible if the module at mod_path cannot be imported.
                mod = import_module(mod_path)
                candidates = [
                    repr(name)
                    for name, candidate in inspect.getmembers(mod, inspect.isclass)
                    if issubclass(candidate, cls) and candidate is not cls
                ]
                msg = "Module '%s' does not contain a '%s' class." % (
                    mod_path,
                    cls_name,
                )
                if candidates:
                    msg += " Choices are: %s." % ", ".join(candidates)
                raise ImportError(msg)
            else:
                # Re-trigger the module import exception.
                import_module(entry)

        # Check for obvious errors. (This check prevents duck typing, but
        # it could be removed if it became a problem in practice.)
        if not issubclass(app_config_class, AppConfig):
            raise ImproperlyConfigured("'%s' isn't a subclass of AppConfig." % entry)

        # Obtain app name here rather than in AppClass.__init__ to keep
        # all error checking for entries in INSTALLED_APPS in one place.
        if app_name is None:
            try:
                app_name = app_config_class.name
            except AttributeError:
                raise ImproperlyConfigured("'%s' must supply a name attribute." % entry)

        # Ensure app_name points to a valid module.
        try:
            app_module = import_module(app_name)
        except ImportError:
            raise ImproperlyConfigured(
                "Cannot import '%s'. Check that '%s.%s.name' is correct."
                % (
                    app_name,
                    app_config_class.__module__,
                    app_config_class.__qualname__,
                )
            )

        # Entry is a path to an app config class.
        return app_config_class(app_name, app_module)

    def get_model(self, model_name, require_ready=True):
        \"\"\"
        Return the model with the given case-insensitive model_name.

        Raise LookupError if no model exists with this name.
        \"\"\"
        if require_ready:
            self.apps.check_models_ready()
        else:
            self.apps.check_apps_ready()
        try:
            return self.models[model_name.lower()]
        except KeyError:
            raise LookupError(
                "App '%s' doesn't have a '%s' model." % (self.label, model_name)
            )

    def get_models(self, include_auto_created=False, include_swapped=False):
        \"\"\"
        Return an iterable of models.

        By default, the following models aren't included:

        - auto-created models for many-to-many relations without
          an explicit intermediate table,
        - models that have been swapped out.

        Set the corresponding keyword argument to True to include such models.
        Keyword arguments aren't documented; they're a private API.
        \"\"\"
        self.apps.check_models_ready()
        for model in self.models.values():
            if model._meta.auto_created and not include_auto_created:
                continue
            if model._meta.swapped and not include_swapped:
                continue
            yield model

    def import_models(self):
        # Dictionary of models for this app, primarily maintained in the
        # 'all_models' attribute of the Apps this AppConfig is attached to.
        self.models = self.apps.all_models[self.label]

        if module_has_submodule(self.module, MODELS_MODULE_NAME):
            models_module_name = "%s.%s" % (self.name, MODELS_MODULE_NAME)
            self.models_module = import_module(models_module_name)

    def ready(self):
        \"\"\"
        Override this method in subclasses to run code when Django starts.
        \"\"\"```
'./django/apps/registry.py':
```python
import functools
import sys
import threading
import warnings
from collections import Counter, defaultdict
from functools import partial

from django.core.exceptions import AppRegistryNotReady, ImproperlyConfigured

from .config import AppConfig


class Apps:
    \"\"\"
    A registry that stores the configuration of installed applications.

    It also keeps track of models, e.g. to provide reverse relations.
    \"\"\"

    def __init__(self, installed_apps=()):
        # installed_apps is set to None when creating the main registry
        # because it cannot be populated at that point. Other registries must
        # provide a list of installed apps and are populated immediately.
        if installed_apps is None and hasattr(sys.modules[__name__], "apps"):
            raise RuntimeError("You must supply an installed_apps argument.")

        # Mapping of app labels => model names => model classes. Every time a
        # model is imported, ModelBase.__new__ calls apps.register_model which
        # creates an entry in all_models. All imported models are registered,
        # regardless of whether they're defined in an installed application
        # and whether the registry has been populated. Since it isn't possible
        # to reimport a module safely (it could reexecute initialization code)
        # all_models is never overridden or reset.
        self.all_models = defaultdict(dict)

        # Mapping of labels to AppConfig instances for installed apps.
        self.app_configs = {}

        # Stack of app_configs. Used to store the current state in
        # set_available_apps and set_installed_apps.
        self.stored_app_configs = []

        # Whether the registry is populated.
        self.apps_ready = self.models_ready = self.ready = False
        # For the autoreloader.
        self.ready_event = threading.Event()

        # Lock for thread-safe population.
        self._lock = threading.RLock()
        self.loading = False

        # Maps ("app_label", "modelname") tuples to lists of functions to be
        # called when the corresponding model is ready. Used by this class's
        # `lazy_model_operation()` and `do_pending_operations()` methods.
        self._pending_operations = defaultdict(list)

        # Populate apps and models, unless it's the main registry.
        if installed_apps is not None:
            self.populate(installed_apps)

    def populate(self, installed_apps=None):
        \"\"\"
        Load application configurations and models.

        Import each application module and then each model module.

        It is thread-safe and idempotent, but not reentrant.
        \"\"\"
        if self.ready:
            return

        # populate() might be called by two threads in parallel on servers
        # that create threads before initializing the WSGI callable.
        with self._lock:
            if self.ready:
                return

            # An RLock prevents other threads from entering this section. The
            # compare and set operation below is atomic.
            if self.loading:
                # Prevent reentrant calls to avoid running AppConfig.ready()
                # methods twice.
                raise RuntimeError("populate() isn't reentrant")
            self.loading = True

            # Phase 1: initialize app configs and import app modules.
            for entry in installed_apps:
                if isinstance(entry, AppConfig):
                    app_config = entry
                else:
                    app_config = AppConfig.create(entry)
                if app_config.label in self.app_configs:
                    raise ImproperlyConfigured(
                        "Application labels aren't unique, "
                        "duplicates: %s" % app_config.label
                    )

                self.app_configs[app_config.label] = app_config
                app_config.apps = self

            # Check for duplicate app names.
            counts = Counter(
                app_config.name for app_config in self.app_configs.values()
            )
            duplicates = [name for name, count in counts.most_common() if count > 1]
            if duplicates:
                raise ImproperlyConfigured(
                    "Application names aren't unique, "
                    "duplicates: %s" % ", ".join(duplicates)
                )

            self.apps_ready = True

            # Phase 2: import models modules.
            for app_config in self.app_configs.values():
                app_config.import_models()

            self.clear_cache()

            self.models_ready = True

            # Phase 3: run ready() methods of app configs.
            for app_config in self.get_app_configs():
                app_config.ready()

            self.ready = True
            self.ready_event.set()

    def check_apps_ready(self):
        \"\"\"Raise an exception if all apps haven't been imported yet.\"\"\"
        if not self.apps_ready:
            from django.conf import settings

            # If "not ready" is due to unconfigured settings, accessing
            # INSTALLED_APPS raises a more helpful ImproperlyConfigured
            # exception.
            settings.INSTALLED_APPS
            raise AppRegistryNotReady("Apps aren't loaded yet.")

    def check_models_ready(self):
        \"\"\"Raise an exception if all models haven't been imported yet.\"\"\"
        if not self.models_ready:
            raise AppRegistryNotReady("Models aren't loaded yet.")

    def get_app_configs(self):
        \"\"\"Import applications and return an iterable of app configs.\"\"\"
        self.check_apps_ready()
        return self.app_configs.values()

    def get_app_config(self, app_label):
        \"\"\"
        Import applications and returns an app config for the given label.

        Raise LookupError if no application exists with this label.
        \"\"\"
        self.check_apps_ready()
        try:
            return self.app_configs[app_label]
        except KeyError:
            message = "No installed app with label '%s'." % app_label
            for app_config in self.get_app_configs():
                if app_config.name == app_label:
                    message += " Did you mean '%s'?" % app_config.label
                    break
            raise LookupError(message)

    # This method is performance-critical at least for Django's test suite.
    @functools.cache
    def get_models(self, include_auto_created=False, include_swapped=False):
        \"\"\"
        Return a list of all installed models.

        By default, the following models aren't included:

        - auto-created models for many-to-many relations without
          an explicit intermediate table,
        - models that have been swapped out.

        Set the corresponding keyword argument to True to include such models.
        \"\"\"
        self.check_models_ready()

        result = []
        for app_config in self.app_configs.values():
            result.extend(app_config.get_models(include_auto_created, include_swapped))
        return result

    def get_model(self, app_label, model_name=None, require_ready=True):
        \"\"\"
        Return the model matching the given app_label and model_name.

        As a shortcut, app_label may be in the form <app_label>.<model_name>.

        model_name is case-insensitive.

        Raise LookupError if no application exists with this label, or no
        model exists with this name in the application. Raise ValueError if
        called with a single argument that doesn't contain exactly one dot.
        \"\"\"
        if require_ready:
            self.check_models_ready()
        else:
            self.check_apps_ready()

        if model_name is None:
            app_label, model_name = app_label.split(".")

        app_config = self.get_app_config(app_label)

        if not require_ready and app_config.models is None:
            app_config.import_models()

        return app_config.get_model(model_name, require_ready=require_ready)

    def register_model(self, app_label, model):
        # Since this method is called when models are imported, it cannot
        # perform imports because of the risk of import loops. It mustn't
        # call get_app_config().
        model_name = model._meta.model_name
        app_models = self.all_models[app_label]
        if model_name in app_models:
            if (
                model.__name__ == app_models[model_name].__name__
                and model.__module__ == app_models[model_name].__module__
            ):
                warnings.warn(
                    "Model '%s.%s' was already registered. Reloading models is not "
                    "advised as it can lead to inconsistencies, most notably with "
                    "related models." % (app_label, model_name),
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    "Conflicting '%s' models in application '%s': %s and %s."
                    % (model_name, app_label, app_models[model_name], model)
                )
        app_models[model_name] = model
        self.do_pending_operations(model)
        self.clear_cache()

    def is_installed(self, app_name):
        \"\"\"
        Check whether an application with this name exists in the registry.

        app_name is the full name of the app e.g. 'django.contrib.admin'.
        \"\"\"
        self.check_apps_ready()
        return any(ac.name == app_name for ac in self.app_configs.values())

    def get_containing_app_config(self, object_name):
        \"\"\"
        Look for an app config containing a given object.

        object_name is the dotted Python path to the object.

        Return the app config for the inner application in case of nesting.
        Return None if the object isn't in any registered app config.
        \"\"\"
        self.check_apps_ready()
        candidates = []
        for app_config in self.app_configs.values():
            if object_name.startswith(app_config.name):
                subpath = object_name.removeprefix(app_config.name)
                if subpath == "" or subpath[0] == ".":
                    candidates.append(app_config)
        if candidates:
            return sorted(candidates, key=lambda ac: -len(ac.name))[0]

    def get_registered_model(self, app_label, model_name):
        \"\"\"
        Similar to get_model(), but doesn't require that an app exists with
        the given app_label.

        It's safe to call this method at import time, even while the registry
        is being populated.
        \"\"\"
        model = self.all_models[app_label].get(model_name.lower())
        if model is None:
            raise LookupError("Model '%s.%s' not registered." % (app_label, model_name))
        return model

    @functools.cache
    def get_swappable_settings_name(self, to_string):
        \"\"\"
        For a given model string (e.g. "auth.User"), return the name of the
        corresponding settings name if it refers to a swappable model. If the
        referred model is not swappable, return None.

        This method is decorated with @functools.cache because it's performance
        critical when it comes to migrations. Since the swappable settings don't
        change after Django has loaded the settings, there is no reason to get
        the respective settings attribute over and over again.
        \"\"\"
        to_string = to_string.lower()
        for model in self.get_models(include_swapped=True):
            swapped = model._meta.swapped
            # Is this model swapped out for the model given by to_string?
            if swapped and swapped.lower() == to_string:
                return model._meta.swappable
            # Is this model swappable and the one given by to_string?
            if model._meta.swappable and model._meta.label_lower == to_string:
                return model._meta.swappable
        return None

    def set_available_apps(self, available):
        \"\"\"
        Restrict the set of installed apps used by get_app_config[s].

        available must be an iterable of application names.

        set_available_apps() must be balanced with unset_available_apps().

        Primarily used for performance optimization in TransactionTestCase.

        This method is safe in the sense that it doesn't trigger any imports.
        \"\"\"
        available = set(available)
        installed = {app_config.name for app_config in self.get_app_configs()}
        if not available.issubset(installed):
            raise ValueError(
                "Available apps isn't a subset of installed apps, extra apps: %s"
                % ", ".join(available - installed)
            )

        self.stored_app_configs.append(self.app_configs)
        self.app_configs = {
            label: app_config
            for label, app_config in self.app_configs.items()
            if app_config.name in available
        }
        self.clear_cache()

    def unset_available_apps(self):
        \"\"\"Cancel a previous call to set_available_apps().\"\"\"
        self.app_configs = self.stored_app_configs.pop()
        self.clear_cache()

    def set_installed_apps(self, installed):
        \"\"\"
        Enable a different set of installed apps for get_app_config[s].

        installed must be an iterable in the same format as INSTALLED_APPS.

        set_installed_apps() must be balanced with unset_installed_apps(),
        even if it exits with an exception.

        Primarily used as a receiver of the setting_changed signal in tests.

        This method may trigger new imports, which may add new models to the
        registry of all imported models. They will stay in the registry even
        after unset_installed_apps(). Since it isn't possible to replay
        imports safely (e.g. that could lead to registering listeners twice),
        models are registered when they're imported and never removed.
        \"\"\"
        if not self.ready:
            raise AppRegistryNotReady("App registry isn't ready yet.")
        self.stored_app_configs.append(self.app_configs)
        self.app_configs = {}
        self.apps_ready = self.models_ready = self.loading = self.ready = False
        self.clear_cache()
        self.populate(installed)

    def unset_installed_apps(self):
        \"\"\"Cancel a previous call to set_installed_apps().\"\"\"
        self.app_configs = self.stored_app_configs.pop()
        self.apps_ready = self.models_ready = self.ready = True
        self.clear_cache()

    def clear_cache(self):
        \"\"\"
        Clear all internal caches, for methods that alter the app registry.

        This is mostly used in tests.
        \"\"\"
        self.get_swappable_settings_name.cache_clear()
        # Call expire cache on each model. This will purge
        # the relation tree and the fields cache.
        self.get_models.cache_clear()
        if self.ready:
            # Circumvent self.get_models() to prevent that the cache is refilled.
            # This particularly prevents that an empty value is cached while cloning.
            for app_config in self.app_configs.values():
                for model in app_config.get_models(include_auto_created=True):
                    model._meta._expire_cache()

    def lazy_model_operation(self, function, *model_keys):
        \"\"\"
        Take a function and a number of ("app_label", "modelname") tuples, and
        when all the corresponding models have been imported and registered,
        call the function with the model classes as its arguments.

        The function passed to this method must accept exactly n models as
        arguments, where n=len(model_keys).
        \"\"\"
        # Base case: no arguments, just execute the function.
        if not model_keys:
            function()
        # Recursive case: take the head of model_keys, wait for the
        # corresponding model class to be imported and registered, then apply
        # that argument to the supplied function. Pass the resulting partial
        # to lazy_model_operation() along with the remaining model args and
        # repeat until all models are loaded and all arguments are applied.
        else:
            next_model, *more_models = model_keys

            # This will be executed after the class corresponding to next_model
            # has been imported and registered. The `func` attribute provides
            # duck-type compatibility with partials.
            def apply_next_model(model):
                next_function = partial(apply_next_model.func, model)
                self.lazy_model_operation(next_function, *more_models)

            apply_next_model.func = function

            # If the model has already been imported and registered, partially
            # apply it to the function now. If not, add it to the list of
            # pending operations for the model, where it will be executed with
            # the model class as its sole argument once the model is ready.
            try:
                model_class = self.get_registered_model(*next_model)
            except LookupError:
                self._pending_operations[next_model].append(apply_next_model)
            else:
                apply_next_model(model_class)

    def do_pending_operations(self, model):
        \"\"\"
        Take a newly-prepared model and pass it to each function waiting for
        it. This is called at the very end of Apps.register_model().
        \"\"\"
        key = model._meta.app_label, model._meta.model_name
        for function in self._pending_operations.pop(key, []):
            function(model)

apps = Apps(installed_apps=None)
```"""

GOOGLE_LANGCHAIN_CONTEXT = """'langchain/libs/partners/google-genai/langchain_google_genai/chat_models.py':
```py
from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from urllib.parse import urlparse

# TODO: remove ignore once the google package is published with types
import google.generativeai as genai  # type: ignore[import]
import requests
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain_google_genai._common import GoogleGenerativeAIError

IMAGE_TYPES: Tuple = ()
try:
    import PIL
    from PIL.Image import Image

    IMAGE_TYPES = IMAGE_TYPES + (Image,)
except ImportError:
    PIL = None  # type: ignore
    Image = None  # type: ignore

logger = logging.getLogger(__name__)


class ChatGoogleGenerativeAIError(GoogleGenerativeAIError):
    \"\"\"
    Custom exception class for errors associated with the `Google GenAI` API.

    This exception is raised when there are specific issues related to the
    Google genai API usage in the ChatGoogleGenerativeAI class, such as unsupported
    message types or roles.
    \"\"\"


def _create_retry_decorator() -> Callable[[Any], Any]:
    \"\"\"
    Creates and returns a preconfigured tenacity retry decorator.

    The retry decorator is configured to handle specific Google API exceptions
    such as ResourceExhausted and ServiceUnavailable. It uses an exponential
    backoff strategy for retries.

    Returns:
        Callable[[Any], Any]: A retry decorator configured for handling specific
        Google API exceptions.
    \"\"\"
    import google.api_core.exceptions

    multiplier = 2
    min_seconds = 1
    max_seconds = 60
    max_retries = 10

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=multiplier, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
            | retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
            | retry_if_exception_type(google.api_core.exceptions.GoogleAPIError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _chat_with_retry(generation_method: Callable, **kwargs: Any) -> Any:
    \"\"\"
    Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided
    chat generation function. It is useful for handling intermittent issues
    like network errors or temporary service unavailability.

    Args:
        generation_method (Callable): The chat generation method to be executed.
        **kwargs (Any): Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    \"\"\"
    retry_decorator = _create_retry_decorator()
    from google.api_core.exceptions import InvalidArgument  # type: ignore

    @retry_decorator
    def _chat_with_retry(**kwargs: Any) -> Any:
        try:
            return generation_method(**kwargs)
        except InvalidArgument as e:
            # Do not retry for these errors.
            raise ChatGoogleGenerativeAIError(
                f"Invalid argument provided to Gemini: {e}"
            ) from e
        except Exception as e:
            raise e

    return _chat_with_retry(**kwargs)


async def _achat_with_retry(generation_method: Callable, **kwargs: Any) -> Any:
    \"\"\"
    Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided
    chat generation function. It is useful for handling intermittent issues
    like network errors or temporary service unavailability.

    Args:
        generation_method (Callable): The chat generation method to be executed.
        **kwargs (Any): Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    \"\"\"
    retry_decorator = _create_retry_decorator()
    from google.api_core.exceptions import InvalidArgument  # type: ignore

    @retry_decorator
    async def _achat_with_retry(**kwargs: Any) -> Any:
        try:
            return await generation_method(**kwargs)
        except InvalidArgument as e:
            # Do not retry for these errors.
            raise ChatGoogleGenerativeAIError(
                f"Invalid argument provided to Gemini: {e}"
            ) from e
        except Exception as e:
            raise e

    return await _achat_with_retry(**kwargs)


def _is_openai_parts_format(part: dict) -> bool:
    return "type" in part


def _is_vision_model(model: str) -> bool:
    return "vision" in model


def _is_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _is_b64(s: str) -> bool:
    return s.startswith("data:image")


def _load_image_from_gcs(path: str, project: Optional[str] = None) -> Image:
    try:
        from google.cloud import storage  # type: ignore[attr-defined]
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required to load images from GCS."
            " Install it with `pip install google-cloud-storage`"
        )
    if PIL is None:
        raise ImportError(
            "PIL is required to load images. Please install it "
            "with `pip install pillow`"
        )

    gcs_client = storage.Client(project=project)
    pieces = path.split("/")
    blobs = list(gcs_client.list_blobs(pieces[2], prefix="/".join(pieces[3:])))
    if len(blobs) > 1:
        raise ValueError(f"Found more than one candidate for {path}!")
    img_bytes = blobs[0].download_as_bytes()
    return PIL.Image.open(BytesIO(img_bytes))


def _url_to_pil(image_source: str) -> Image:
    if PIL is None:
        raise ImportError(
            "PIL is required to load images. Please install it "
            "with `pip install pillow`"
        )
    try:
        if isinstance(image_source, IMAGE_TYPES):
            return image_source  # type: ignore[return-value]
        elif _is_url(image_source):
            if image_source.startswith("gs://"):
                return _load_image_from_gcs(image_source)
            response = requests.get(image_source)
            response.raise_for_status()
            return PIL.Image.open(BytesIO(response.content))
        elif _is_b64(image_source):
            _, encoded = image_source.split(",", 1)
            data = base64.b64decode(encoded)
            return PIL.Image.open(BytesIO(data))
        elif os.path.exists(image_source):
            return PIL.Image.open(image_source)
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


def _convert_to_parts(
    raw_content: Union[str, Sequence[Union[str, dict]]],
) -> List[genai.types.PartType]:
    \"\"\"Converts a list of LangChain messages into a google parts.\"\"\"
    parts = []
    content = [raw_content] if isinstance(raw_content, str) else raw_content
    for part in content:
        if isinstance(part, str):
            parts.append(genai.types.PartDict(text=part))
        elif isinstance(part, Mapping):
            # OpenAI Format
            if _is_openai_parts_format(part):
                if part["type"] == "text":
                    parts.append({"text": part["text"]})
                elif part["type"] == "image_url":
                    img_url = part["image_url"]
                    if isinstance(img_url, dict):
                        if "url" not in img_url:
                            raise ValueError(
                                f"Unrecognized message image format: {img_url}"
                            )
                        img_url = img_url["url"]
                    parts.append({"inline_data": _url_to_pil(img_url)})
                else:
                    raise ValueError(f"Unrecognized message part type: {part['type']}")
            else:
                # Yolo
                logger.warning(
                    "Unrecognized message part format. Assuming it's a text part."
                )
                parts.append(part)
        else:
            # TODO: Maybe some of Google's native stuff
            # would hit this branch.
            raise ChatGoogleGenerativeAIError(
                "Gemini only supports text and inline_data parts."
            )
    return parts


def _parse_chat_history(
    input_messages: Sequence[BaseMessage], convert_system_message_to_human: bool = False
) -> List[genai.types.ContentDict]:
    messages: List[genai.types.MessageDict] = []

    raw_system_message: Optional[SystemMessage] = None
    for i, message in enumerate(input_messages):
        if (
            i == 0
            and isinstance(message, SystemMessage)
            and not convert_system_message_to_human
        ):
            raise ValueError(
                \"\"\"SystemMessages are not yet supported!

To automatically convert the leading SystemMessage to a HumanMessage,
set  `convert_system_message_to_human` to True. Example:

llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
\"\"\"
            )
        elif i == 0 and isinstance(message, SystemMessage):
            raw_system_message = message
            continue
        elif isinstance(message, AIMessage):
            role = "model"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(
                f"Unexpected message with type {type(message)} at the position {i}."
            )

        parts = _convert_to_parts(message.content)
        if raw_system_message:
            if role == "model":
                raise ValueError(
                    "SystemMessage should be followed by a HumanMessage and "
                    "not by AIMessage."
                )
            parts = _convert_to_parts(raw_system_message.content) + parts
            raw_system_message = None
        messages.append({"role": role, "parts": parts})
    return messages


def _parts_to_content(parts: List[genai.types.PartType]) -> Union[List[dict], str]:
    \"\"\"Converts a list of Gemini API Part objects into a list of LangChain messages.\"\"\"
    if len(parts) == 1 and parts[0].text is not None and not parts[0].inline_data:
        # Simple text response. The typical response
        return parts[0].text
    elif not parts:
        logger.warning("Gemini produced an empty response.")
        return ""
    messages = []
    for part in parts:
        if part.text is not None:
            messages.append(
                {
                    "type": "text",
                    "text": part.text,
                }
            )
        else:
            # TODO: Handle inline_data if that's a thing?
            raise ChatGoogleGenerativeAIError(f"Unexpected part type. {part}")
    return messages


def _response_to_result(
    response: genai.types.GenerateContentResponse,
    ai_msg_t: Type[BaseMessage] = AIMessage,
    human_msg_t: Type[BaseMessage] = HumanMessage,
    chat_msg_t: Type[BaseMessage] = ChatMessage,
    generation_t: Type[ChatGeneration] = ChatGeneration,
) -> ChatResult:
    \"\"\"Converts a PaLM API response into a LangChain ChatResult.\"\"\"
    llm_output = {}
    if response.prompt_feedback:
        try:
            prompt_feedback = type(response.prompt_feedback).to_dict(
                response.prompt_feedback, use_integers_for_enums=False
            )
            llm_output["prompt_feedback"] = prompt_feedback
        except Exception as e:
            logger.debug(f"Unable to convert prompt_feedback to dict: {e}")

    generations: List[ChatGeneration] = []

    role_map = {
        "model": ai_msg_t,
        "user": human_msg_t,
    }
    for candidate in response.candidates:
        content = candidate.content
        parts_content = _parts_to_content(content.parts)
        if content.role not in role_map:
            logger.warning(
                f"Unrecognized role: {content.role}. Treating as a ChatMessage."
            )
            msg = chat_msg_t(content=parts_content, role=content.role)
        else:
            msg = role_map[content.role](content=parts_content)
        generation_info = {}
        if candidate.finish_reason:
            generation_info["finish_reason"] = candidate.finish_reason.name
        if candidate.safety_ratings:
            generation_info["safety_ratings"] = [
                type(rating).to_dict(rating) for rating in candidate.safety_ratings
            ]
        generations.append(generation_t(message=msg, generation_info=generation_info))
    if not response.candidates:
        # Likely a "prompt feedback" violation (e.g., toxic input)
        # Raising an error would be different than how OpenAI handles it,
        # so we'll just log a warning and continue with an empty message.
        logger.warning(
            "Gemini produced an empty response. Continuing with empty message\n"
            f"Feedback: {response.prompt_feedback}"
        )
        generations = [generation_t(message=ai_msg_t(content=""), generation_info={})]
    return ChatResult(generations=generations, llm_output=llm_output)


class ChatGoogleGenerativeAI(BaseChatModel):
    \"\"\"`Google Generative AI` Chat models API.

    To use, you must have either:

        1. The ``GOOGLE_API_KEY``` environment variable set with your API key, or
        2. Pass your API key using the google_api_key kwarg to the ChatGoogle
           constructor.

    Example:
        .. code-block:: python

            from langchain_google_genai import ChatGoogleGenerativeAI
            chat = ChatGoogleGenerativeAI(model="gemini-pro")
            chat.invoke("Write me a ballad about LangChain")

    \"\"\"

    model: str = Field(
        ...,
        description=\"\"\"The name of the model to use.
Supported examples:
    - gemini-pro\"\"\",
    )
    max_output_tokens: int = Field(default=None, description="Max output tokens")

    client: Any  #: :meta private:
    google_api_key: Optional[SecretStr] = None
    temperature: Optional[float] = None
    \"\"\"Run inference with this temperature. Must by in the closed
       interval [0.0, 1.0].\"\"\"
    top_k: Optional[int] = None
    \"\"\"Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive.\"\"\"
    top_p: Optional[int] = None
    \"\"\"The maximum cumulative probability of tokens to consider when sampling.

        The model uses combined Top-k and nucleus sampling.

        Tokens are sorted based on their assigned probabilities so
        that only the most likely tokens are considered. Top-k
        sampling directly limits the maximum number of tokens to
        consider, while Nucleus sampling limits number of tokens
        based on the cumulative probability.

        Note: The default value varies by model, see the
        `Model.top_p` attribute of the `Model` returned the
        `genai.get_model` function.
    \"\"\"
    n: int = Field(default=1, alias="candidate_count")
    \"\"\"Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated.\"\"\"
    convert_system_message_to_human: bool = False
    \"\"\"Whether to merge any leading SystemMessage into the following HumanMessage.
    
    Gemini does not support system messages; any unsupported messages will 
    raise an error.\"\"\"

    class Config:
        allow_population_by_field_name = True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @property
    def _llm_type(self) -> str:
        return "chat-google-generative-ai"

    @property
    def _is_geminiai(self) -> bool:
        return self.model is not None and "gemini" in self.model

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        if isinstance(google_api_key, SecretStr):
            google_api_key = google_api_key.get_secret_value()
        genai.configure(api_key=google_api_key)
        if (
            values.get("temperature") is not None
            and not 0 <= values["temperature"] <= 1
        ):
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values.get("top_p") is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values.get("top_k") is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")
        model = values["model"]
        values["client"] = genai.GenerativeModel(model_name=model)
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        \"\"\"Get the identifying parameters.\"\"\"
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "n": self.n,
        }

    def _prepare_params(
        self, stop: Optional[List[str]], **kwargs: Any
    ) -> Dict[str, Any]:
        gen_config = {
            k: v
            for k, v in {
                "candidate_count": self.n,
                "temperature": self.temperature,
                "stop_sequences": stop,
                "max_output_tokens": self.max_output_tokens,
                "top_k": self.top_k,
                "top_p": self.top_p,
            }.items()
            if v is not None
        }
        if "generation_config" in kwargs:
            gen_config = {**gen_config, **kwargs.pop("generation_config")}
        params = {"generation_config": gen_config, **kwargs}
        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params, chat, message = self._prepare_chat(messages, stop=stop)
        response: genai.types.GenerateContentResponse = _chat_with_retry(
            content=message,
            **params,
            generation_method=chat.send_message,
        )
        return _response_to_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params, chat, message = self._prepare_chat(messages, stop=stop)
        response: genai.types.GenerateContentResponse = await _achat_with_retry(
            content=message,
            **params,
            generation_method=chat.send_message_async,
        )
        return _response_to_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params, chat, message = self._prepare_chat(messages, stop=stop)
        response: genai.types.GenerateContentResponse = _chat_with_retry(
            content=message,
            **params,
            generation_method=chat.send_message,
            stream=True,
        )
        for chunk in response:
            _chat_result = _response_to_result(
                chunk,
                ai_msg_t=AIMessageChunk,
                human_msg_t=HumanMessageChunk,
                chat_msg_t=ChatMessageChunk,
                generation_t=ChatGenerationChunk,
            )
            gen = cast(ChatGenerationChunk, _chat_result.generations[0])
            yield gen
            if run_manager:
                run_manager.on_llm_new_token(gen.text)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params, chat, message = self._prepare_chat(messages, stop=stop)
        async for chunk in await _achat_with_retry(
            content=message,
            **params,
            generation_method=chat.send_message_async,
            stream=True,
        ):
            _chat_result = _response_to_result(
                chunk,
                ai_msg_t=AIMessageChunk,
                human_msg_t=HumanMessageChunk,
                chat_msg_t=ChatMessageChunk,
                generation_t=ChatGenerationChunk,
            )
            gen = cast(ChatGenerationChunk, _chat_result.generations[0])
            yield gen
            if run_manager:
                await run_manager.on_llm_new_token(gen.text)

    def _prepare_chat(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], genai.ChatSession, genai.types.ContentDict]:
        params = self._prepare_params(stop, **kwargs)
        history = _parse_chat_history(
            messages,
            convert_system_message_to_human=self.convert_system_message_to_human,
        )
        message = history.pop()
        chat = self.client.start_chat(history=history)
        return params, chat, message```
'langchain/libs/partners/google-genai/langchain_google_genai/llms.py':
```py
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import google.api_core
import google.generativeai as genai  # type: ignore[import]
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import BaseLLM, create_base_retry_decorator
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env


def _create_retry_decorator(
    llm: BaseLLM,
    *,
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    \"\"\"Creates a retry decorator for Vertex / Palm LLMs.\"\"\"

    errors = [
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.Aborted,
        google.api_core.exceptions.DeadlineExceeded,
        google.api_core.exceptions.GoogleAPIError,
    ]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=max_retries, run_manager=run_manager
    )
    return decorator


def _completion_with_retry(
    llm: GoogleGenerativeAI,
    prompt: LanguageModelInput,
    is_gemini: bool = False,
    stream: bool = False,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    \"\"\"Use tenacity to retry the completion call.\"\"\"
    retry_decorator = _create_retry_decorator(
        llm, max_retries=llm.max_retries, run_manager=run_manager
    )

    @retry_decorator
    def _completion_with_retry(
        prompt: LanguageModelInput, is_gemini: bool, stream: bool, **kwargs: Any
    ) -> Any:
        generation_config = kwargs.get("generation_config", {})
        if is_gemini:
            return llm.client.generate_content(
                contents=prompt, stream=stream, generation_config=generation_config
            )
        return llm.client.generate_text(prompt=prompt, **kwargs)

    return _completion_with_retry(
        prompt=prompt, is_gemini=is_gemini, stream=stream, **kwargs
    )


def _is_gemini_model(model_name: str) -> bool:
    return "gemini" in model_name


def _strip_erroneous_leading_spaces(text: str) -> str:
    \"\"\"Strip erroneous leading spaces from text.

    The PaLM API will sometimes erroneously return a single leading space in all
    lines > 1. This function strips that space.
    \"\"\"
    has_leading_space = all(not line or line[0] == " " for line in text.split("\n")[1:])
    if has_leading_space:
        return text.replace("\n ", "\n")
    else:
        return text


class GoogleGenerativeAI(BaseLLM, BaseModel):
    \"\"\"Google GenerativeAI models.

    Example:
        .. code-block:: python

            from langchain_google_genai import GoogleGenerativeAI
            llm = GoogleGenerativeAI(model="gemini-pro")
    \"\"\"

    client: Any  #: :meta private:
    model: str = Field(
        ...,
        description=\"\"\"The name of the model to use.
Supported examples:
    - gemini-pro
    - models/text-bison-001\"\"\",
    )
    \"\"\"Model name to use.\"\"\"
    google_api_key: Optional[SecretStr] = None
    temperature: float = 0.7
    \"\"\"Run inference with this temperature. Must by in the closed interval
       [0.0, 1.0].\"\"\"
    top_p: Optional[float] = None
    \"\"\"Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0].\"\"\"
    top_k: Optional[int] = None
    \"\"\"Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive.\"\"\"
    max_output_tokens: Optional[int] = None
    \"\"\"Maximum number of tokens to include in a candidate. Must be greater than zero.
       If unset, will default to 64.\"\"\"
    n: int = 1
    \"\"\"Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated.\"\"\"
    max_retries: int = 6
    \"\"\"The maximum number of retries to make when generating.\"\"\"

    @property
    def is_gemini(self) -> bool:
        \"\"\"Returns whether a model is belongs to a Gemini family or not.\"\"\"
        return _is_gemini_model(self.model)

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        \"\"\"Validate api key, python package exists.\"\"\"
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        model_name = values["model"]

        if isinstance(google_api_key, SecretStr):
            google_api_key = google_api_key.get_secret_value()

        genai.configure(api_key=google_api_key)

        if _is_gemini_model(model_name):
            values["client"] = genai.GenerativeModel(model_name=model_name)
        else:
            values["client"] = genai

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        if values["max_output_tokens"] is not None and values["max_output_tokens"] <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        return values

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations: List[List[Generation]] = []
        generation_config = {
            "stop_sequences": stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
        }
        for prompt in prompts:
            if self.is_gemini:
                res = _completion_with_retry(
                    self,
                    prompt=prompt,
                    stream=False,
                    is_gemini=True,
                    run_manager=run_manager,
                    generation_config=generation_config,
                )
                candidates = [
                    "".join([p.text for p in c.content.parts]) for c in res.candidates
                ]
                generations.append([Generation(text=c) for c in candidates])
            else:
                res = _completion_with_retry(
                    self,
                    model=self.model,
                    prompt=prompt,
                    stream=False,
                    is_gemini=False,
                    run_manager=run_manager,
                    **generation_config,
                )
                prompt_generations = []
                for candidate in res.candidates:
                    raw_text = candidate["output"]
                    stripped_text = _strip_erroneous_leading_spaces(raw_text)
                    prompt_generations.append(Generation(text=stripped_text))
                generations.append(prompt_generations)

        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        generation_config = kwargs.get("generation_config", {})
        if stop:
            generation_config["stop_sequences"] = stop
        for stream_resp in _completion_with_retry(
            self,
            prompt,
            stream=True,
            is_gemini=True,
            run_manager=run_manager,
            generation_config=generation_config,
            **kwargs,
        ):
            chunk = GenerationChunk(text=stream_resp.text)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    stream_resp.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )

    @property
    def _llm_type(self) -> str:
        \"\"\"Return type of llm.\"\"\"
        return "google_palm"

    def get_num_tokens(self, text: str) -> int:
        \"\"\"Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        \"\"\"
        if self.is_gemini:
            raise ValueError("Counting tokens is not yet supported!")
        result = self.client.count_text_tokens(model=self.model, prompt=text)
        return result["token_count"]```
'langchain/libs/partners/google-genai/langchain_google_genai/embeddings.py':
```py
from typing import Dict, List, Optional

# TODO: remove ignore once the google package is published with types
import google.generativeai as genai  # type: ignore[import]
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_google_genai._common import GoogleGenerativeAIError


class GoogleGenerativeAIEmbeddings(BaseModel, Embeddings):
    \"\"\"`Google Generative AI Embeddings`.

    To use, you must have either:

        1. The ``GOOGLE_API_KEY``` environment variable set with your API key, or
        2. Pass your API key using the google_api_key kwarg to the ChatGoogle
           constructor.

    Example:
        .. code-block:: python

            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            embeddings.embed_query("What's our Q1 revenue?")
    \"\"\"

    model: str = Field(
        ...,
        description="The name of the embedding model to use. "
        "Example: models/embedding-001",
    )
    task_type: Optional[str] = Field(
        None,
        description="The task type. Valid options include: "
        "task_type_unspecified, retrieval_query, retrieval_document, "
        "semantic_similarity, classification, and clustering",
    )
    google_api_key: Optional[SecretStr] = Field(
        None,
        description="The Google API key to use. If not provided, "
        "the GOOGLE_API_KEY environment variable will be used.",
    )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        \"\"\"Validates that the python package exists in environment.\"\"\"
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        if isinstance(google_api_key, SecretStr):
            google_api_key = google_api_key.get_secret_value()
        genai.configure(api_key=google_api_key)
        return values

    def _embed(
        self, texts: List[str], task_type: str, title: Optional[str] = None
    ) -> List[List[float]]:
        task_type = self.task_type or "retrieval_document"
        try:
            result = genai.embed_content(
                model=self.model,
                content=texts,
                task_type=task_type,
                title=title,
            )
        except Exception as e:
            raise GoogleGenerativeAIError(f"Error embedding content: {e}") from e
        return result["embedding"]

    def embed_documents(
        self, texts: List[str], batch_size: int = 5
    ) -> List[List[float]]:
        \"\"\"Embed a list of strings. Vertex AI currently
        sets a max batch size of 5 strings.

        Args:
            texts: List[str] The list of strings to embed.
            batch_size: [int] The batch size of embeddings to send to the model

        Returns:
            List of embeddings, one for each text.
        \"\"\"
        task_type = self.task_type or "retrieval_document"
        return self._embed(texts, task_type=task_type)

    def embed_query(self, text: str) -> List[float]:
        \"\"\"Embed a text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        \"\"\"
        task_type = self.task_type or "retrieval_query"
        return self._embed([text], task_type=task_type)[0]```"""

CURRENT_CODE_BASE = """def recursive_complete(model: ChatOpenAI | ChatGoogleGenerativeAI, model_name: str, messages: List[str], response: str = "THERE IS NO CODE HERE, RETURN FAIL EXIT CODE 1") -> str:
    print(colored("CUR LEN: " + str(len(messages)), 'cyan'))
    if model_name == 'gpt-3.5-turbo-16k' or model_name == 'gpt-3.5-turbo-16k-0613':    
    # OpenAI specific logic
        temp_response = ""
        if type(response) != str:
            temp_response = response.content
        else:
            temp_response = response
        assessment_chat = [SystemMessage(content=BASE_RECURSIVE_PROMPT[0]['content']), SchemaHumanMessage(content=temp_response)]
        #print(colored("TYPE OF ITEM: " + str(type(assessment_chat)), 'red'))
        formatted_messages = ChatPromptTemplate.from_messages(assessment_chat).format_messages()
        is_done = model.invoke(formatted_messages).content.strip()
    elif model.model == 'gemini-pro':
        # Gemini specific logic - directly use the passed messages
        temp_prompt = "System Message:\n" + BASE_RECURSIVE_PROMPT[0]['content'] + ""
        is_done = model.invoke(messages).content.strip()
    else:
        raise ValueError(f"Unsupported model for recursive completion: {model.model}")

    if is_done.upper() == "YES":
        # Termination condition
        print(colored("DONE! RESPONSE FROM LLM: " + is_done, 'green'))
        return response
    elif is_done.upper() == "NO":
        # Recursive condition
        print("NOT DONE! RESPONSE FROM LLM: " + is_done)
        messages.append(SchemaHumanMessage(content=EXPERT_DICT_MESSAGE['content']))  # Update the messages for the next call
        if len(messages) == 7:
            messages.pop(len(messages)-1)
        if len(messages) > 6:
            print(colored("USER RESPONSE: " + EXPERT_DICT_MESSAGE['content'], 'magenta'))
        new_formatted_messages = ChatPromptTemplate.from_messages(messages).format_messages()
        new_content = model.invoke(new_formatted_messages).content.strip()
        messages.append(AIMessage(content=new_content))
        print()
        print(colored("NEXT NEW CONTENT 'NO': " + new_content, 'yellow'))
        return recursive_complete(model, model.model_name, messages, new_content)
    else:
        # Handle unexpected responses
        print(colored("Unexpected response from model: " + is_done, "red"))
        return response"""