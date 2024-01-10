#!/usr/bin/env python3

print(";; ...doing...",__name__)

# Version Space Candidate Elimination inside of MeTTa
# This implementation focuses on bringing this machine learning algorithm into the MeTTa relational programming environment.
# Douglas R. Miles 2023

# Standard Library Imports
import atexit, io, inspect, json, os, re, subprocess, sys, traceback
from collections import Counter
from glob import glob
from time import monotonic_ns, time

# Third-Party Imports
from pyswip import (Atom as PySwipAtom, Term, call, Functor, PL_discard_foreign_frame, PL_new_term_ref, PL_open_foreign_frame,
                    registerForeign, PL_PRUNED, PL_retry, PL_FA_NONDETERMINISTIC, PL_foreign_control, PL_foreign_context, PL_FIRST_CALL, PL_REDO, Variable, Prolog as PySwip)
from pyswip.easy import newModule, Query

import hyperonpy as hp
from hyperon.atoms import *
from hyperon.base import AbstractSpace, SpaceRef, GroundingSpace, interpret
from hyperon.base import *
from hyperon.ext import register_atoms, register_tokens
from hyperon.runner import MeTTa

# Readline Imports (Platform Specific)
try: import readline
except ImportError: import pyreadline3 as readline

# Global Variables
VSPACE_VERBOSE = os.environ.get("VSPACE_VERBOSE")
# 0 = for scripts/demos
# 1 = developer
# 2 = debugger
verbose = 1
if VSPACE_VERBOSE is not None:
 try: verbose = int(VSPACE_VERBOSE) # Convert it to an integer
 except ValueError: ""

# Error Handling for Janus
try: from janus import *
except Exception as e:
 if verbose>0: print(f"; Error: {e}")

# Error Handling for OpenAI
try:
 import openai
 try: openai.api_key = os.environ["OPENAI_API_KEY"]
 except KeyError: ""
except Exception as e:
 if verbose>0: print(f"; Error: {e}")



histfile = os.path.join(os.path.expanduser("~"), ".metta_history")
is_init = True
oper_dict = {}
janus_dict = {}
syms_dict = {}

def parent_space():
    return the_python_runner.parent.space()

def  child_space():
    return the_python_runner.space()

def   self_space():
    return the_new_runner_space
space_refs = {
    #'&vspace': lambda: the_verspace,
    '&gptspace': lambda: the_gptspace,
    #'&flybase': lambda: the_flybase,
    '&parent': lambda: parent_space(),
    '&child': lambda: child_space(),
    '&self': self_space}


try:
    readline.set_history_length(300)
    readline.read_history_file(histfile)
    readline.set_history_length(300)
    h_len = readline.get_current_history_length()
except FileNotFoundError:
    open(histfile, 'wb').close()
    h_len = 0

def add_to_history_if_unique(item, position_from_last=1):
    for i in range(1, readline.get_current_history_length() + 1):
        if readline.get_history_item(i) == item: return
    insert_to_history(item, position_from_last)

def insert_to_history(item, position_from_last=5):
    hist = [readline.get_history_item(i) for i in range(1, readline.get_current_history_length() + 1)]
    # Remove the item from the list if it already exists before the insertion position
    insertion_position = max(0, len(hist) - position_from_last)
    if item in hist[:insertion_position]:
        hist.remove(item)
    # Insert the item at the desired position
    hist.insert(insertion_position, item)
    # Clear and repopulate the history
    readline.clear_history()
    for h in hist:
        readline.add_history(h)

def readline_add_history(t):
    readline.add_history(t)


insert_to_history('!(get-by-key &my-dict "A")')
insert_to_history("@metta !")
insert_to_history("!(mine-overlaps)")
insert_to_history("!(try-overlaps)")
insert_to_history("!(load-flybase-full)")
insert_to_history("!(load-flybase-tiny)")
#insert_to_history("!(load-vspace)")
insert_to_history("!(learn-vspace)")
insert_to_history('!(match &parent (gene_map_table $Dmel $abo $G  $C $D $E)  (gene_map_table $Dmel $abo $G  $C $D $E) )')
insert_to_history('!(match &parent (gene_map_table $Dmel $abo (GeneValueNode "FBgn0000018") $C $D $E) (gene_map_table $Dmel $abo (GeneValueNode "FBgn0000018") $C $D $E))')
insert_to_history('!(add-atom &parent (gene_map_table (ConceptNode "Dmel") (ConceptNode "abo") (GeneValueNode "FBgn0000018") (ConceptNode "2-44") (ConceptNode "32C1-32C1") (StringValue "2L:10973443..10975293(-1)")))')
insert_to_history("!(match &parent $ $)")
insert_to_history("!(match &flybase $ $)")
insert_to_history('!(match &flybase (gene_map_table $Dmel $abo $G  $C $D $E)  (gene_map_table $Dmel $abo $G  $C $D $E) )')
insert_to_history('!(match &flybase (gene_map_table $Dmel $abo (GeneValueNode "FBgn0000018") $C $D $E) (gene_map_table $Dmel $abo (GeneValueNode "FBgn0000018") $C $D $E))')
insert_to_history('!(add-atom &flybase (gene_map_table (ConceptNode "Dmel") (ConceptNode "abo") (GeneValueNode "FBgn0000018") (ConceptNode "2-44") (ConceptNode "32C1-32C1") (StringValue "2L:10973443..10975293(-1)")))')
insert_to_history('!(match &flybase (gene_map_table $Dmel $abo FBgn0000018 $C $D $E) (gene_map_table $Dmel $abo FBgn0000018 $C $D $E))')
insert_to_history('!(test_custom_v_space)', position_from_last=1)

OPTIONS = ['apple', 'banana', 'cherry', 'date', 'elderberry']

# The completer function
def completer(text, state):
    options = [i for i in OPTIONS if i.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return None

# Register the completer function
readline.set_completer(completer)

# Use the tab key for completion
readline.parse_and_bind('tab: complete')

def save(prev_h_len, histfile):
    new_h_len = readline.get_current_history_length()
    readline.set_history_length(400)
    readline.append_history_file(new_h_len - prev_h_len, histfile)
atexit.register(save, h_len, histfile)

def export_to_metta(func):
    setattr(func, 'metta', True)
    if verbose>3: print_cmt(f"{func}={getattr(func, 'export_to_metta', False)}")
    return func

def export_flags(**kwargs):
    def decorator(func):
        if verbose > 1: print(f";   export_flags({repr(func)})", end=" ")
        for n in kwargs:
            setattr(func, n, kwargs[n])
        if verbose > 1:
            for n in kwargs:
                print(f"{repr(n)}={repr(getattr(func, n, None))}", end=" ")
            print()
        return func
    return decorator

def get_call_parts(func):
    sig = inspect.signature(func)
    params = sig.parameters
    # Constructing full parameter strings
    param_parts = []
    params_call_parts = []
    var_args = var_kwargs = None

    for param_name, param in params.items():
        part = param_name
        call_part = param_name
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            var_args = f'*{param_name}'
            part =  var_args
            call_part = var_args
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            var_kwargs = f'**{param_name}'
            part =  var_kwargs
            call_part = var_kwargs
        elif param.default != inspect.Parameter.empty and isinstance(param.default, (int, str)):
            part += f'={repr(param.default)}'
        param_parts.append(part)
        params_call_parts.append(call_part)

    return param_parts, params_call_parts


def add_janus_methods(module, dict = janus_dict):
    for name, func in inspect.getmembers(module):
        if inspect.isfunction(func):
            if getattr(func, "Janus", False) or getattr(func, "MeTTa", False):
                use_name = getattr(func, 'name', name)
                non_underscore_attrs = {attr: getattr(func, attr) for attr in dir(func) if not attr.startswith('_')}
                if len(non_underscore_attrs)==0: continue
                param_parts, params_call_parts = get_call_parts(func)
                add_to_janus(use_name, param_parts, params_call_parts, func,non_underscore_attrs, janus_dict)

@export_flags(MeTTa=True)
def add_python_module(module, dict=oper_dict):
    for name, func in inspect.getmembers(module):
        if inspect.isfunction(func):
           add_python_function(name, func, dict)

def add_python_function(name, func, dict):
    use_name = getattr(func, 'name', name)
    non_underscore_attrs = {attr: getattr(func, attr) for attr in dir(func) if not attr.startswith('_')}
    if len(non_underscore_attrs)==0: return False
    param_parts, params_call_parts = get_call_parts(func)

    added = False

    if getattr(func, "Janus", False):
        added = add_to_janus(use_name, param_parts, params_call_parts,
            func, non_underscore_attrs, janus_dict) or added

    if getattr(func, 'MeTTa', False):
        added = add_to_metta(use_name, param_parts, params_call_parts,
            getattr(func, 'op', "OperationAtom"), getattr(func, 'unwrap', False),
            func, non_underscore_attrs, dict) or added

    if not added and verbose > 3:
        print_cmt(f"unused({name}({param_parts})) attributes: {non_underscore_attrs}")

    return added


@export_flags(MeTTa=True)
def add_to_metta(name, param_parts, params_call_parts, op_kind, unwrap, func, non_underscore_attrs, dict=oper_dict):
    hyphens, underscores = name.replace('_', '-'), name.replace('-', '_')

    # Construct the param_str from param_parts
    metta_params_str = ' '.join(param_parts)

    s = f"!({hyphens})" if metta_params_str == "" else f"!({hyphens} {metta_params_str})"
    add_to_history_if_unique(s)

    if hyphens in dict:
        return True

    # Construct the param_str from param_parts
    param_str = ', '.join(param_parts)
    # Using params_call_parts in the function call inside lambda
    params_for_call = ', '.join(params_call_parts)

    # Constructing the source code to execute
    src = f'op = {op_kind}("{hyphens}", lambda {param_str}: [{underscores}({params_for_call})], unwrap={unwrap})'
    local_vars = {}

    if verbose > 1:
        print_cmt(f"{src} # {non_underscore_attrs}"[5:])

    try:
        exec(src, globals(), local_vars)
        dict[hyphens] = local_vars['op']
        dict[underscores] = local_vars['op']
        return True
    except SyntaxError as e:
        print_cmt(f"Syntax error in executing: {src}")
        print_cmt(f"Error details: {e}")
        return False


def add_to_janus(name, param_parts, params_call_parts, func, non_underscore_attrs, dict = janus_dict):

    if getattr(func, 'CallsVSpace', False): return False
    #if not getattr(func, "Janus", False): return False

    suggestedName = getattr(func, 'name', name)

    if suggestedName is not None:
        use_name = suggestedName
    else: use_name = name

    for key, item in dict.items():
        if key==use_name:
            return True

    suggestedFlags = getattr(func, 'flags', None)
    if suggestedFlags is None:
        suggestedFlags = 0

    suggestedArity = getattr(func, 'arity', None)
    if suggestedArity is None:
        num_args = len(param_parts)
        suggestedArity = num_args
        func.arity = suggestedArity

    #if verbose > 1:
    print_cmt(f"registerForeign({use_name}, arity = {param_parts}/{suggestedArity}, flags = {suggestedFlags} ) {non_underscore_attrs}")

    #if not getattr(func, "Janus", False): return False
    dict[use_name]=func
    registerForeign(func, arity = suggestedArity, flags = suggestedFlags )
    return True




#############################################################################################################################
# @export_flags(MeTTa=True)
# def add_python_module(module, dict = oper_dict):
#
#     for name, func in inspect.getmembers(module):
#         if inspect.isfunction(func):
#             if getattr(func, 'MeTTa', False):
#                 suggestedName = getattr(func, 'name', None)
#                 if suggestedName is not None:
#                     use_name = suggestedName
#                 else: use_name = name
#                 suggestedArity = getattr(func, 'arity', None)
#                 sig = inspect.signature(func)
#                 params = sig.parameters
#
#                 num_args = len([p for p in params.values() if p.default == p.empty and p.kind == p.POSITIONAL_OR_KEYWORD])
#                 # Check for varargs
#                 has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params.values())
#                 if suggestedArity is None: suggestedArity = num_args
#
#                 keyword_arg_names = []
#                 # Iterate through parameters
#                 for name, param in params.items():
#                     # Check if the parameter is either VAR_KEYWORD or has a default value
#                     if param.kind == inspect.Parameter.VAR_KEYWORD or param.default != inspect.Parameter.empty:
#                         keyword_arg_names.append(name)
#
#                 if is_varargs==True or has_varargs:
#                     suggestedArity = -1
#                 add_to_metta(use_name, suggestedArity,
#                    getattr(func, 'op', "OperationAtom"),
#                    getattr(func, 'unwrap', False), func, dict)
#
# @export_flags(MeTTa=True)
# def add_to_metta(name, length, op_kind, unwrap, funct, dict = oper_dict):
#     hyphens, underscores = name.replace('_', '-'), name.replace('-', '_')
#     mettavars = ' '.join(f"${chr(97 + i)}" for i in range(length)).strip()
#     pyvars = ', '.join(chr(97 + i) for i in range(length)).strip()
#
#     if length == -1: #varargs
#         pyvars = "*args"
#         mettavars = "..."
#
#     s = f"!({hyphens})" if mettavars == "" else f"!({hyphens} {mettavars})"
#     add_to_history_if_unique(s); #print(s)
#     if hyphens not in dict:
#         src, local_vars = f'op = {op_kind}( "{hyphens}", lambda {pyvars}: [{underscores}({pyvars})], unwrap={unwrap})', {}
#         if verbose>1: print_cmt(f"add_to_metta={src}")
#         if verbose>8: print_cmt(f"funct={dir(funct)}")
#         exec(src, globals(), local_vars)
#         dict[hyphens] = local_vars['op']
#         dict[underscores] = local_vars['op']
#
#############################################################################################################################


# Function to find subclasses of clazz in a module
def find_subclasses_of(module, clazz):
    subclasses = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, clazz) and obj is not clazz:
            subclasses[name]=obj
    return subclasses.items()

    for name, claz in find_subclasses_of(module,AbstractSpace):
        print(f"found class {claz} with name {name}")
        # inspect the constructor and syntesize a function that will create an object


@export_flags(MeTTa=True)
def add_to_swip(name, dict = oper_dict):
    hyphens, underscores = name.replace('_', '-'), name.replace('-', '_')
    add_to_history_if_unique(f"!({hyphens})")
    if hyphens not in dict:
        src, local_vars = f'op = lambda : [swip_exec("{underscores}")]', {}
        exec(src, {}, local_vars)
        if verbose>1: print_cmt(f"swip: {hyphens}")
        dict[hyphens] = OperationAtom(hyphens, local_vars['op'], unwrap=False)

def addSpaceName(name, space):
    global syms_dict, space_refs
    prev = getSpaceByName(name)
    name = str(name)
    if not name.startswith("&"):
        name = "&" + name
    syms_dict[name] = lambda _: G(asSpaceRef(space))
    if prev is None:
        space_refs[name] = lambda : space

def getSpaceByName(name):
    global space_refs
    if name is ValueAtom:
        name = name.get_value()
    if name is GroundingSpace:
        return name
    name = str(name)
    if not name.startswith("&"):
        name = "&" + name
    found = space_refs.get(name, None)
    if found is None: return None
    return found()

def getNameBySpace(target_space):
    if target_space is None:
        return None
    global space_refs, syms_dict
    # Search in space_refs
    for name, space_func in space_refs.items():
        S = space_func()
        if S is target_space:
            return name
        if S:
            if id(S) == id(target_space):
                return name
    # Search in syms_dict
    for name, space_func in syms_dict.items():
        GR = space_func(None)
        if GR:
            if id(GR) == id(target_space):
                return name
            if id(GR.get_object()) == id(target_space):
                return name
    return None

vspace_ordinal = 0

# Mainly a sanity loading test class
class MettaLearner:
    ""


class Circles:
    def __init__(self, initial_data=None):
        self.data = {}
        if initial_data:
            for key, value in initial_data.items():
                self.__setitem__(key, value)

    def _get_key(self, key):
        try:
            hash_key = hash(key)
            return ('hash', hash_key)
        except TypeError:
            id_key = id(key)
            return ('id', id_key)

    def __getitem__(self, key):
        key_type, key_value = self._get_key(key)
        return self.data[(key_type, key_value)][1]

    def __setitem__(self, key, value):
        key_type, key_value = self._get_key(key)
        self.data[(key_type, key_value)] = (key, value)

    def __delitem__(self, key):
        key_type, key_value = self._get_key(key)
        del self.data[(key_type, key_value)]

    def __contains__(self, key):
        key_type, key_value = self._get_key(key)
        return (key_type, key_value) in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for key_tuple in self.data.keys():
            yield key_tuple

    def original_keys(self):
        for key, _ in self.data.values():
            yield key

    def get(self, key, default=None):
        key_type, key_value = self._get_key(key)
        if (key_type, key_value) in self.data:
            return self.data[(key_type, key_value)][1]
        else:
            return default

    def items(self):
        return [(key, value) for key, value in self.data.values()]

    def keys(self):
        return [key for key, _ in self.data.values()]

    def values(self):
        return [value for _, value in self.data.values()]

    def clear(self):
        self.data.clear()

    def pop(self, key, default=None):
        key_type, key_value = self._get_key(key)
        return self.data.pop((key_type, key_value), (None, default))[1]

    def popitem(self):
        _, (key, value) = self.data.popitem()
        return (key, value)

    def setdefault(self, key, default=None):
        key_type, key_value = self._get_key(key)
        return self.data.setdefault((key_type, key_value), (key, default))[1]

    def update(self, other):
        for key, value in other.items():
            self.__setitem__(key, value)


# subclass to later capture any utility we can add to 'subst'
def asSpaceRef(obj):
    if isinstance(obj, (VSpaceRef, SpaceRef)):
        return obj
    return VSpaceRef(obj)

class VSpaceRef(SpaceRef):

    """
    A reference to a Space, which may be accessed directly, wrapped in a grounded atom,
    or passed to a MeTTa interpreter.
    """

    def __init__(self, space_obj):
        """
        Initialize a new SpaceRef based on the given space object, either a CSpace
        or a custom Python object.
        """
        super().__init__(space_obj)
        self.py_space_obj = space_obj
        #if type(space_obj) is hp.CSpace:
        #    self.cspace = space_obj
        #else:
        #    self.cspace = hp.space_new_custom(space_obj)

    def is_VSpace(self):
        return isinstance(self.py_space_obj,VSpace)

    def get_atoms(self):
        """
        Returns a list of all Atoms in the Space, or None if that is impossible
        """
        if self.is_VSpace():
            return self.py_space_obj.get_atoms()

        res = hp.space_list(self.cspace)
        if res == None:
            return None
        result = []
        for r in res:
            result.append(Atom._from_catom(r))
        return result


    def __del__(self):
        """Free the underlying CSpace object """
        return
        if self.is_VSpace(): self.py_space_obj.__del__()
        else: hp.space_free(self.cspace)

    def __eq__(self, other):
        """Compare two SpaceRef objects for equality, based on their underlying spaces."""
        if not isinstance(other,SpaceRef): return False
        if self.is_VSpace(): return get_payload(self) is other.get_payload(self)
        else: return hp.space_eq(self.cspace, other.cspace)


    @staticmethod
    def _from_cspace(cspace):
        """
        Create a new SpaceRef based on the given CSpace object.
        """
        return asSpaceRef(cspace)

    def copy(self):
        """
        Returns a new copy of the SpaceRef, referencing the same underlying Space.
        """
        return self

    def add_atom(self, atom):
        """
        Add an Atom to the Space.
        """
        if self.is_VSpace():
            return self.py_space_obj.add(atom)

        hp.space_add(self.cspace, atom.catom)

    def remove_atom(self, atom):
        """
        Delete the specified Atom from the Space.
        """
        if self.is_VSpace():
            return self.py_space_obj.remove(atom)

        return hp.space_remove(self.cspace, atom.catom)

    def replace_atom(self, atom, replacement):
        """
        Replaces the specified Atom, if it exists in the Space, with the supplied replacement.
        """
        if self.is_VSpace():
            return self.py_space_obj.replace(atom, replacement)

        return hp.space_replace(self.cspace, atom.catom, replacement.catom)

    def atom_count(self):
        """
        Returns the number of Atoms in the Space, or -1 if it cannot be readily computed.
        """

        if self.is_VSpace():
            return self.py_space_obj.atom_count()

        return hp.space_atom_count(self.cspace)


    def get_payload(self):
        """
        Returns the Space object referenced by the SpaceRef, or None if the object does not have a
        direct Python interface.
        """
        if self.is_VSpace():
            return self.py_space_obj;

        return hp.space_get_payload(self.cspace)

    def query(self, pattern):
        """
        Performs the specified query on the Space, and returns the result as a BindingsSet.
        """
        if self.is_VSpace():
            return self.py_space_obj.query(pattern);

        result = hp.space_query(self.cspace, pattern.catom)
        return BindingsSet(result)

    def subst(self, pattern, templ):
        """
        Performs a substitution within the Space
        """

        if self.is_VSpace():
            return self.py_space_obj.subst(pattern, templ);

        cspace = super().cspace
        return [Atom._from_catom(catom) for catom in
                hp.space_subst(cspace, pattern.catom,
                                         templ.catom)]


def foreign_framed(func):
    def wrapper(*args, **kwargs):
        swipl_fid = PL_open_foreign_frame()
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            if verbose>0: print_cmt(f"Error: {e}")
            if verbose>0: traceback.print_exc()
        finally:
            PL_discard_foreign_frame(swipl_fid)
        return result
    return wrapper


@export_flags(MeTTa=True)
class VSpace(AbstractSpace):

    def from_space(self, cspace):
        self.gspace = GroundingSpaceRef(cspace)

    def __init__(self, space_name=None, unwrap=False):
        super().__init__()
        #addSpaceName(ispace_name,self)
        if space_name is None:
            global vspace_ordinal
            ispace_name = f"&vspace_{vspace_ordinal}"
            vspace_ordinal=vspace_ordinal+1
            space_name = ispace_name
        self.sp_name = PySwipAtom(space_name)
        swip.assertz(f"was_asserted_space('{space_name}')")
        #swip.assertz(f"was_space_type('{space_name}',asserted_space)")
        self.sp_module = newModule("user")
        self.unwrap = unwrap
        addSpaceName(space_name,self)

    def __del__(self):
        return
        pass

    def swip_space_name(self):
        return swipRef(self.sp_name)
        #return self.sp_name

    @foreign_framed
    def query(self, query_atom):
        new_bindings_set = BindingsSet.empty()
        #swipl_load = PL_new_term_ref()
        metta_vars = [atom for atom in query_atom.iterate() if atom.get_type() == AtomKind.VARIABLE]
        metaVarNames = [str(atom) for atom in metta_vars]
        circles = Circles()
        swivars = [m2s(circles,item,1) for item in metta_vars]
        varsList = Variable()
        varsList.unify(swivars)
        varNames = Variable()
        varNames.unify(metaVarNames)
        swip_obj = m2s(circles,query_atom)
        if verbose>1: print_cmt(f"circles={circles}")
        #if verbose>1: print_cmt(f"metta_vars={metta_vars}, swivars={swivars}")
        q = PySwipQ(Functor('metta_iter_bind',4)
          (self.swip_space_name(), swip_obj, varsList, varNames), module=self.sp_module)

        while q.nextSolution():
            swivars = varsList.value
            bindings = Bindings()
            vn = 0
            for mv in metta_vars:
                 svar = swivars[vn]
                 sval = svar
                 if verbose>1: pt(f"svar({vn})=",svar, " ")
                 if isinstance(svar, Variable):
                     sval = sval.value
                 else: sval = svar
                 if verbose>1: pt(f"sval({vn})=",sval, " ")
                 mval = s2m(circles,sval)
                 if verbose>1: pt(f"mval({vn})=",mval, " ")
                 bindings.add_var_binding(mv, mval)
                 vn = vn + 1

            new_bindings_set.push(bindings)
        q.closeQuery()
        return new_bindings_set

    def _call(self, functor_name, *args):
        q = PySwipQ(Functor(functor_name, len(args) + 1)(self.swip_space_name(), *args), module=self.sp_module)
        try: return q.nextSolution()
        except Exception as e:
            if verbose>0: print_cmt(f"Error: {e}")
            if verbose>0: traceback.print_exc()
        finally: q.closeQuery()

    @foreign_framed
    def add(self, atom):
        circles = Circles()
        return self._call("add-atom", m2s(circles,atom))

    @foreign_framed
    def add_atom(self, atom):
        circles = Circles()
        return self._call("add-atom", m2s(circles,atom))

    @foreign_framed
    def remove_atom(self, atom):
        circles = Circles()
        return self._call("remove-atom", m2s(circles,atom))

    @foreign_framed
    def remove(self, atom):
        circles = Circles()
        return self._call("remove-atom", m2s(circles,atom))

    @foreign_framed
    def replace(self, from_atom, to_atom):
        circles = Circles()
        return self._call("replace-atom", m2s(circles,from_atom), m2s(circles,to_atom))

    @foreign_framed
    def subst(self, pattern, templ):
        """
        Performs a substitution within the Space
        """
        circles = Circles()
        return self._call("subst_pattern_template", m2s(circles,pattern), m2s(circles,templ))

    @foreign_framed
    def atom_count(self):
        result = list(swip.query(f"'atom-count'('{self.sp_name}',AtomCount)"))
        if verbose>1: print_cmt(result)
        if result is None: return 0
        if len(result)==0: return 0
        CB = result[0]
        if CB is None: return 0
        C = CB['AtomCount']
        if not isinstance(C,int):
            C = C.value
        return C

    @foreign_framed
    def get_atoms(self):
        circles = Circles()
        result = list(swip.query(f"'get-atoms'('{self.sp_name}',AtomsList)"))
        if result is None: return []
        if len(result)==0: return []
        CB = result[0]
        if CB is None: return []
        C = CB['AtomsList']
        if verbose>1: print_cmt(f"get_atoms={type(C)}")
        R = s2m(circles,C)
        return R

    def atoms_iter(self):

        swipl_fid = PL_open_foreign_frame()
        Atoms = Variable("Iter")
        q = PySwipQ(Functor("atoms_iter", 2)(self.swip_space_name(), Atoms), module=self.sp_module)

        def closeff():
            nonlocal swipl_fid
            ff = swipl_fid
            swipl_fid = None
            if ff is not None:
                PL_discard_foreign_frame(ff)


        class LazyIter:

            circles = Circles()

            def __init__(self, q, v):
                self.q, self.v = q, v

            def __iter__(self):
                return self

            def __next__(self):
                if self.q.nextSolution():
                    return s2m(circles,self.v.value.value)
                closeff()
                raise StopIteration

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.q.closeQuery()
                closeff()

        return LazyIter(q, Atoms)

    def copy(self):
        return self

class VSpaceCallRust(VSpace):
    def __init__(self, space_name=None, unwrap=False):
        super().__init__()

    #def eval_in_rust_mettaf():



class PySwipQ(Query):

    def __init__(self, *terms, **kwargs):
        if verbose > 1:
            for obj in terms:
                println(obj)
        super().__init__(*terms, **kwargs)

    def nextSolution(self):
        return Query.nextSolution()
        #return PL_next_solution(Query.qid)
    #nextSolution = staticmethod(nextSolution)

    def cutQuery(self):
        Query.cutQuery()
        #PL_cut_query(Query.qid)
    #cutQuery = staticmethod(cutQuery)

    def closeQuery(self):
        Query.closeQuery()
    #    if Query.qid is not None:
    #        PL_close_query(Query.qid)
    #        Query.qid = None
    #closeQuery = staticmethod(closeQuery)

access_error = True

@export_flags(MeTTa=True)
class FederatedSpace(VSpace):

    def __init__(self, space_name, unwrap=False):
        super().__init__(space_name, unwrap)

    def _checked_impl(self, method_name, *args):
        if access_error:
            raise Exception(f"Error in FederatedSpace.{method_name}: Implementation for {method_name}({', '.join(map(str, args))}) is not complete.")
        return super()

    def query(self, query_atom):
        return self._checked_impl("query", query_atom).query(query_atom)

    def add(self, atom):
        return self._checked_impl("add", atom).add(atom)

    def remove(self, atom):
        return self._checked_impl("remove", atom).remove(atom)

    def replace(self, from_atom, to_atom):
        return self._checked_impl("replace", from_atom, to_atom).replace(from_atom, to_atom)

    def atom_count(self):
        return self._checked_impl("atom_count").atom_count()

    def atoms_iter(self):
        return self._checked_impl("atoms_iter").atoms_iter()

    def copy(self):
        return self


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

@export_flags(MeTTa=True)
def test_custom_v_space():
    #test_custom_space(lambda: (lambda vs: vs.incrHome() and vs)(VSpace()))
    test_custom_v_space1()
    test_custom_v_space2()

@export_flags(MeTTa=True)
def test_custom_v_space1():
    test_custom_space(lambda: VSpace())

@export_flags(MeTTa=True)
def test_custom_v_space2():
    test_custom_space(lambda: the_nb_space)

    #test_custom_space(lambda: the_new_runner_space)

def test_custom_space(LambdaSpaceFn):

    def passTest(msg):
        print(f"Pass Test:({msg})")

    def failTest(msg):
        print(f"raise AssertionError({msg})")
        #raise AssertionError(msg)

    def self_assertEqualNoOrder(list1, list2, msg=None):
        """
        Asserts that two lists are equal, regardless of their order.
        """
        def py_sorted(n):

            class MyIterable:
                def __init__(self, data):
                    self.data = data
                    self.index = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    if self.index < len(self.data):
                        result = self.data[self.index]
                        self.index += 1
                        return result
                    raise StopIteration

            try:
                if isinstance(n, ExpressionAtom):
                    return py_sorted(n.get_children())
                return sorted(n)
            except TypeError:
                def custom_sort(item):
                    try:
                        if isinstance(item, (int, float)):
                            return (0, item)
                        elif isinstance(item, ExpressionAtom):
                            return py_sorted(item.get_children())
                        else:
                            return (1, str(item))
                    except TypeError:
                        return (1, str(item))

            try: return sorted(n, key=custom_sort)
            except TypeError: n # return sorted(MyIterable(n), key=custom_sort)


        if py_sorted(list1) != py_sorted(list2):
            failTest(msg or f"Lists differ: {list1} != {list2}")
        else: passTest(msg or f" {list1} == {list2} ")

    def self_assertTrue(expr, msg=None):
        """
        Asserts that an expression is true.
        """
        if not expr:
            failTest(msg or f"Expression is not true: {expr}")
        else: passTest(msg or f"Expression is true: {expr}")

    def self_assertFalse(expr, msg=None):
        """
        Asserts that an expression is false.
        """
        if expr:
            failTest(msg or f"Expression is not false: {expr}")
        else: passTest(msg or f"Expression is false: {expr}")

    def self_assertEqual(val1, val2, msg=None):
        """
        Asserts that two values are equal.
        """
        if val1 != val2:
            failTest(msg or f"Values differ: {val1} != {val2}")
        else: passTest(msg or f"Values same: {val1} == {val2}")


    print(f"test_custom_space--------------------------------------------:({LambdaSpaceFn})------------------------------------------")



    test_space = LambdaSpaceFn()
    test_space.test_attrib = "Test Space Payload Attrib"

    kb = asSpaceRef(test_space)


    kb.add_atom(S("a"))
    kb.add_atom(S("b"))
    #kb.add_atom(E(S("a"),S("b")))

    self_assertEqual(kb.atom_count(), 2)
    self_assertEqual(kb.get_payload().test_attrib, "Test Space Payload Attrib")
    self_assertEqualNoOrder(kb.get_atoms(), [S("a"), S("b")])

    kb = asSpaceRef(LambdaSpaceFn())
    kb.add_atom(S("a"))
    kb.add_atom(S("b"))
    kb.add_atom(S("c"))

    self_assertTrue(kb.remove_atom(S("b")),"remove_atom on a present atom should return true")
    self_assertFalse(kb.remove_atom(S("bogus")),"remove_atom on a missing atom should return false")
    self_assertEqualNoOrder(kb.get_atoms(), [S("a"), S("c")])

    kb = asSpaceRef(LambdaSpaceFn())
    kb.add_atom(S("a"))
    kb.add_atom(S("b"))
    kb.add_atom(S("c"))

    self_assertTrue(kb.replace_atom(S("b"), S("d")))
    self_assertEqualNoOrder(kb.get_atoms(), [S("a"), S("d"), S("c")])

    kb = asSpaceRef(LambdaSpaceFn())
    kb.add_atom(E(S("A"), S("B")))
    kb.add_atom(E(S("C"), S("D")))
    # Checking that multiple matches can be returned
    kb.add_atom(E(S("A"), S("E")))

    result = kb.query(E(S("A"), V("XX")))
    self_assertEqualNoOrder(result, [{"XX": S("B")}, {"XX": S("E")}])

    m = MeTTa()

    # Make a little space and add it to the MeTTa interpreter's space
    little_space = asSpaceRef(LambdaSpaceFn())
    little_space.add_atom(E(S("A"), S("B")))
    space_atom = G(little_space)
    m.space().add_atom(E(S("little-space"), space_atom))

    # Make sure we can get the little space back, and then query it
    kb_result = m.space().query(E(S("little-space"), V("s")))
    result_atom = kb_result[0].get("s")
    self_assertEqual(result_atom, space_atom)

    result = result_atom.get_object().query(E(S("A"), V("v")))
    self_assertEqualNoOrder(result, [{"v": S("B")}])

    # Add the MeTTa space to the little space for some space recursion
    if verbose>1: print_cmt("mspace")
    mspace = m.space()
    gmspace = G(mspace)
    A = E(S("big-space"), gmspace)
    if verbose>1: print_cmt("little_space.add_atom")
    little_space.add_atom(A)
    if verbose>1: print_cmt("Next Space")
    nested = asSpaceRef(LambdaSpaceFn())
    nested.add_atom(E(S("A"), S("B")))
    space_atom = G(nested)

    runner = MeTTa()
    runner.space().add_atom(space_atom)
    runner.tokenizer().register_token("nested", lambda token: space_atom)

    result = runner.run("!(match nested (A $x1) $x1)")
    self_assertEqual([[S("B")]], result)
    print(f"test_custom_space--------------------------------------------:({LambdaSpaceFn})------------------------------------------")


@export_flags(MeTTa=False)
def s2m(circles,swip_obj, depth=0):
    r = s2m1(circles,swip_obj, depth)
    if verbose<=1: return r
    for i in range(depth+1):
        print("   ",end='')
    print_cmt(f"r({type(r)})={str(r)}/{repr(r)}")
    return r

def s2m1(circles,swip_obj, depth=0):

    if verbose>1:
        for i in range(depth):
            print("   ",end='')
        print_cmt(f's2m({len(circles)},{type(swip_obj)}): {str(swip_obj)}/{repr(swip_obj)}')

    # Already converted
    if isinstance(swip_obj, (VariableAtom, GroundedAtom, Atom, ExpressionAtom)):
        return swip_obj

    if isinstance(swip_obj, str):
        return S(swip_obj)

    assert isinstance(circles, Circles), f"circles must be an instance of the Circles class not {type(circles)}"

    # Handle numbers and convert them to ValueAtom objects in MeTTa
    if isinstance(swip_obj, (int, float)):
        return ValueAtom(swip_obj)

    #oid = id(swip_obj)

    for n in circles.original_keys():
        v = circles[n]
        if v is swip_obj:
            return n

    var = circles.get(swip_obj, None)
    if var is not None:
        return var


    if isinstance(swip_obj, PySwipAtom):
        return S(str(swip_obj))

    if isinstance(swip_obj, Variable):
        sval = swip_obj.get_value()
        if isinstance(sval, Variable):
           sval = sval.get_value()
        if isinstance(sval, Variable):
           n = swip_obj.chars
           mname = sv2mv(n) if n else "$Var"
           mV = V(mname)
           circles[mname] = swip_obj
           circles[id(mV)] = swip_obj
           circles[swip_obj] = mV
        return s2m(circles,sval)

    if isinstance(swip_obj, Functor):
        # Convert the functor to an expression in MeTTa
        if isinstance(swip_obj.name, PySwipAtom):
            sfn = swip_obj.name.value
        else: sfn = swip_obj.name
        if sfn=="[|]": sfn = "::"
        fn = S(sfn)
        argz = [s2m(circles,arg) for arg in swip_obj.args]
        return E(fn, *argz)

    # Handle PySwip lists
    #if isinstance(swip_obj, list):



    mva = [s2m(circles,item) for item in swip_obj]
    try:
        return E(*mva)
    except TypeError:
        return ExpressionAtom(mva)


    raise ValueError(f"Unknown PySwip object type: {type(swip_obj)} {swip_obj}")

mylist_expr = E()
def sv2mv(s):
    return s.replace("_", "$", 1) if s.startswith("_") else "$" + s


@export_flags(MeTTa=False)
def m2s(circles,metta_obj, depth=0):
    r = m2s1(circles,metta_obj, depth)
    if depth==0:
        v = swipRef(r)
    else:
        v = r
    if verbose<=1: return v
    for i in range(depth+1):
        print("   ",end='')

    print(f"r({type(r)})={r}")
    return v

def swipAtom(m):
    a = PySwipAtom(str(m))
    return a

def swipRef(a):
    if isinstance(a, (Term)):
       return a
    v = Variable()
    v.unify(a)
    return v

def unwrap_pyobjs(metta_obj):
    if isinstance(metta_obj, ExpressionAtom):
       return metta_obj
    elif isinstance(metta_obj, GroundedAtom):
       metta_obj = metta_obj.get_object()
    if isinstance(metta_obj, ValueObject):
       metta_obj = metta_obj.value
    return metta_obj

def m2s1(circles, metta_obj, depth=0, preferStringToAtom = None, preferListToCompound = False):

    var = circles.get(metta_obj, None)
    if var is not None:
        return var

    metta_obj = unwrap_pyobjs(metta_obj)

    var = circles.get(metta_obj, None)
    if var is not None:
        return var

    if verbose>1:
        for i in range(depth):
            print("   ",end='')
        print(f'm2s({len(circles)},{type(metta_obj)}): {metta_obj}')

    if isinstance(metta_obj, (Variable, PySwipAtom, Functor, Term)):
        return metta_obj

    if isinstance(metta_obj, str):
        return metta_obj

    if isinstance(metta_obj, bool):
        if metta_obj is True:
            return swipAtom("True")
        else:
            return swipAtom("False")

    elif isinstance(metta_obj, (int, float)):
        return metta_obj

    elif isinstance(metta_obj, OperationObject):
        return m2s1(circles, metta_obj.id, depth+1)

    elif isinstance(metta_obj, SymbolAtom):
        if preferStringToAtom is None:
            preferStringToAtom = (depth>0)

        name = metta_obj.get_name();
        #if preferStringToAtom: return name
        return swipAtom(name)

    sV = None

    if isinstance(metta_obj, VariableAtom):
        oid = mv2svn(metta_obj)
        var = circles.get("$" + oid, None)
        # We are in a circluar reference?
        if var is not None:
            #print(f"{oid}={len(circles)}={type(circles)}={type(metta_obj)}")
            return var

        sV = Variable(name = oid)
        circles["$" + oid] = sV
        circles[metta_obj] = sV
        circles[sV] = metta_obj
        return sV

    oid = id(metta_obj)

    preferListToCompound = True
    if isinstance(metta_obj, SpaceRef):
        return swipAtom(getNameBySpace(metta_obj))
        #L = E(S("SpaceRef"),S(getNameBySpace(metta_obj)))
        #L = list_to_termv(L.get_children())
        #L = list_to_termv(circles,metta_obj.get_atoms(),depth+1)
    elif isinstance(metta_obj, list):
        L = list_to_termv(circles,metta_obj,depth+1)
    elif isinstance(metta_obj, ExpressionAtom):
        L = list_to_termv(circles,metta_obj.get_children(),depth+1)
    elif isinstance(metta_obj, tuple):
        L = list_to_termv(circles,tuple_to_list(metta_obj),depth+1)
    else:
        raise ValueError(f"Unknown MeTTa object type_1: {metta_obj} {type(metta_obj)} {dir(metta_obj)}")

    if depth==0:
        sV = Variable()
        sV.unify(L)
        circles[oid] = sV
        circles[sV] = metta_obj
        circles[metta_obj] = sV
        return sV

    circles[L] = metta_obj
    circles[metta_obj] = L
    return L

def tuple_to_list(t):
    return list(map(tuple_to_list, t)) if isinstance(t, (tuple, list)) else t

# Example usage:
#nested_tuple = (1, 2, (3, 4, (5, 6)), 7)
#converted_list = tuple_to_list(nested_tuple)
#print(converted_list)  # Output will be [1, 2, [3, 4, [5, 6]], 7]

def mv2svn(metta_obj):
    named = metta_obj.get_name().replace('$','_')
    if len(named)==0: return "_0"
    s=named[0]
    if(s == '_' or (s.isalpha() and  s.isupper())):
        return named
    else:
        return "_" + named



def m2s3(circles, metta_obj, depth, preferStringToAtom, preferListToCompound):
    for name, value in circles:
        if  name is metta_obj:
            return value

    if isinstance(metta_obj, SpaceRef):
        return swiplist_to_swip(circles,metta_obj.get_atoms(),depth+1)

    if isinstance(metta_obj, list):
        return swiplist_to_swip(circles,metta_obj)

    if isinstance(metta_obj, ExpressionAtom):
        ch = metta_obj.get_children()
        length = len(ch)
        retargs = []
        if (length==0):
            return swiplist_to_swip(circles,retargs)


    # for testing
    if preferListToCompound:
        for i in range(0,length):
            retargs.append(m2s(circles,ch[i],depth + 1))
        return swiplist_to_swip(circles,retargs)


    f = m2s1(circles,ch[0], depth+1, preferStringToAtom = True)

    for i in range(1,length):
        retargs.append(m2s(circles,ch[i],depth + 1))

    # Convert MeTTa list to PySwip list
    if ch[0].get_name() == "::":
        return swiplist_to_swip(circles,retargs)

    # Converting to functor... Maybe a list later on
    return Functor(f, len(retargs), list_to_termv(circles,retargs))

    if verbose>0: print_cmt(f"Unknown MeTTa object type: {type(metta_obj)}={metta_obj}")

    raise ValueError(f"Unknown MeTTa object type_3: {type(metta_obj)}")

def swiplist_to_swip(circles,retargs, depth=0):
    sv = [m2s1(circles,item,depth) for item in retargs]
    v = Variable()
    v.unify(sv)
    return v

def list_to_termv(circles,retargs, depth=0):
    sv = [m2s1(circles,item,depth) for item in retargs]
    return sv


import numpy as np
class VSNumpyValue(MatchableObject):

    def __eq__(self, metta_obj):
        return isinstance(metta_obj, VSNumpyValue) and\
               (self.content.shape == metta_obj.content.shape) and\
               (self.content == metta_obj.content).all()

    def match_(self, metta_obj):
        sh = self.content.shape
        bindings = {}
        if isinstance(metta_obj, GroundedAtom):
            metta_obj = metta_obj.get_object()
        # Match by equality with another VSNumpyValue
        if isinstance(metta_obj, VSNumpyValue):
            return [{}] if metta_obj == self else []
        # if isinstance(metta_obj, VSPatternValue):
        #     metta_obj = metta_obj.to_expr()
        if isinstance(metta_obj, ExpressionAtom):
            ch = metta_obj.get_children()
            # TODO: constructors and operations
            if len(ch) != sh[0]:
                return []
            for i in range(len(ch)):
                res = self.content[i]
                typ = _np_atom_type(res)
                res = VSNumpyValue(res)
                if isinstance(ch[i], VariableAtom):
                    bindings[ch[i].get_name()] = G(res, typ)
                elif isinstance(ch[i], ExpressionAtom):
                    bind_add = res.match_(ch[i])
                    if bind_add == []:
                        return []
                    bindings.update(bind_add[0])
        return [] if len(bindings) == 0 else [bindings]


class VSPatternValue(MatchableObject):

    def match_(self, orig_metta_obj):
        metta_obj = orig_metta_obj
        if isinstance(metta_obj, GroundedAtom):
            metta_obj = metta_obj.get_object().content
        #if not isinstance(metta_obj, VSPatternValue):
        #    return metta_obj.match_(self)
        # TODO: match to patterns
        #return []
        metta_obj = orig_metta_obj
        bindings = {}
        if isinstance(metta_obj, GroundedAtom):
            metta_obj = metta_obj.get_object()
        # Match by equality with another VSNumpyValue
        if isinstance(metta_obj, VSNumpyValue):
            return [{}] if metta_obj == self else []
        # if isinstance(metta_obj, VSPatternValue):
        #     metta_obj = metta_obj.to_expr()
        if isinstance(metta_obj, ExpressionAtom):
            ch = metta_obj.get_children()
            # TODO: constructors and operations
            if len(ch) != sh[0]:
                return []
            for i in range(len(ch)):
                res = self.content[i]
                typ = _np_atom_type(res)
                res = VSNumpyValue(res)
                if isinstance(ch[i], VariableAtom):
                    bindings[ch[i].get_name()] = G(res, typ)
                elif isinstance(ch[i], ExpressionAtom):
                    bind_add = res.match_(ch[i])
                    if bind_add == []:
                        return []
                    bindings.update(bind_add[0])
        return [] if len(bindings) == 0 else [bindings]

class VSPatternOperation(OperationObject):

    def __init__(self, name, op, unwrap=False, rec=False):
        super().__init__(name, op, unwrap)
        self.rec = rec

    def execute(self, *args, res_typ=AtomType.UNDEFINED):
        print(f"args={args}")
        if self.rec and isinstance(args[0], ExpressionAtom):
            args = args[0].get_children()
            args = [self.execute(arg)[0]\
                if isinstance(arg, ExpressionAtom) else arg for arg in args]
        # If there is a variable or VSPatternValue in arguments, create VSPatternValue
        # instead of executing the operation
        for arg in args:
            if isinstance(arg, GroundedAtom) and\
               isinstance(arg.get_object(), VSPatternValue) or\
               isinstance(arg, VariableAtom):
                return [G(VSPatternValue([self, args]))]
        return super().execute(*args, res_typ=res_typ)

class VSpacePatternOperation(OperationObject):

    def __init__(self, name, op, unwrap=False, rec=False):
        super().__init__(name, op, unwrap)
        self.rec = rec
        self._catom = None

    @property
    def catom(self):
        return self.get_catom()

    @catom.setter
    def catom(self, value):
        self.set_catom(value)

    def get_catom(self):
        # Your getter logic here
        return self._catom

    def set_catom(self, value):
        # Your setter logic here
        self._catom = value

    def execute(self, *args, res_typ=AtomType.UNDEFINED):
        print(f"args={args}")
        if self.rec and isinstance(args[0], ExpressionAtom):
            args = args[0].get_children()
            args = [self.execute(arg)[0] if isinstance(arg, ExpressionAtom) else arg for arg in args]
        # If there is a variable or VSPatternValue in arguments, create VSPatternValue
        # instead of executing the operation
        for arg in args:
            if isinstance(arg, GroundedAtom) and isinstance(arg.get_object(), VSPatternValue):
                return [G(VSPatternValue([self, args]))]
            if isinstance(arg, VariableAtom):
                return [G(VSPatternValue([self, args]))]
        # from super()
        # type-check?
        if False and self.unwrap:
            for arg in args:
                if not isinstance(arg, GroundedAtom):
                    # REM:
                    # Currently, applying grounded operations to pure atoms is not reduced.
                    # If we want, we can raise an exception, or to form a error expression instead,
                    # so a MeTTa program can catch and analyze it.
                    # raise RuntimeError("Grounded operation " + self.name + " with unwrap=True expects only grounded arguments")
                    raise NoReduceError()
            args = [arg.get_object().content for arg in args]
            return [G(ValueObject(self.op(*args)), res_typ)]
        else:
            result = self.op(*args)
            if not isinstance(result, list):
                raise RuntimeError("Grounded operation `" + self.name + "` should return list")
            return result


def _np_atom_type(npobj):
    pt("npobj=",npobj)
    return E(S('NPArray'), E(*[ValueAtom(s, 'Number') for s in npobj.shape]))

def dewrap(arg):
    r = unwrap_pyobjs(arg)
    print(f"dw({type(arg)})={type(r)}")
    return r

def wrapnpop(func):
    def wrapper(*args):
        a = [dewrap(arg) for arg in args]
        res = func(*a)
        typ = _np_atom_type(res)
        return [G(VSNumpyValue(res), typ)]
    return wrapper


def color(t, c):
    cmap = [90, 91, 31, 93, 92, 32, 36, 96, 94, 34, 35, 95, 38]
    return f"\033[{cmap[c % len(cmap)]}m{t}\033[0m"


def oblique(t):
    return f"\033[3m{t}\033[0m"


def underline(t):
    return f"\033[4m{t}\033[0m"


def expr_vars(expr):
    if isinstance(expr, SymbolAtom):
        return []
    elif isinstance(expr, VariableAtom):
        return [str(expr)]
    elif isinstance(expr, ExpressionAtom):
        return [e for c in expr.get_children() for e in expr_vars(c)]
    elif isinstance(expr, GroundedAtom):
        return []
    else:
        raise Exception("Unexpected sexpr type: " + str(type(expr)))


def color_expr(expr, level=0, unif_vars=None):
    name = str(expr)
    if level == 0:
        unif_vars = frozenset(e for e, c in Counter(expr_vars(expr)).items() if c > 1) \
            if unif_vars is None else frozenset()
    if isinstance(expr, SymbolAtom):
        return name
    elif isinstance(expr, VariableAtom):
        return oblique(name) if name in unif_vars else name
    elif isinstance(expr, ExpressionAtom):
        return (color("(", level) +
                " ".join(color_expr(c, level + 1, unif_vars) for c in expr.get_children()) +
                color(")", level))
    elif isinstance(expr, GroundedAtom):
        return underline(name)
    else:
        raise Exception("Unexpected sexpr type: " + str(type(expr)))



@export_flags(MeTTa=True)
def print_l_e(obj):
    if obj is None:
        print("None!")
        return obj

    if isinstance(obj, str):
        print(obj)
        return obj

    try:
        # Attempt to iterate over the object
        for item in obj:
            try:
                color_expr(item)
            except Exception:
                print(item)
    except TypeError:
        # If a TypeError is raised, the object is not iterable
        # if verbose>0: print_cmt(type(obj))
        print(obj)
    return obj

@export_flags(MeTTa=True)
def print_cmt(*args, prefix=";; "):
   for arg in args:
       println(arg, prefix=prefix)
       flush_console()

@export_flags(MeTTa=True, name="print", unwrap=True)
def println(orig, prefix=""):
    """
    Prints the given object and returns it.

    Args:
        orig: The object to be printed.

    Returns:
        The same object that was passed in.
    """
    try:
      prefix_print(prefix, orig)
    except Exception as e:
      if verbose>0: print_cmt(f"println-Error: {e}")
    flush_console()
    return orig

def prefix_print(prefix, orig):

    obj = unwrap_pyobjs(orig)

    if isinstance(obj, str):
        objlns = obj.splitlines()
        for r in objlns:
            print(prefix, r)
        return

    if isinstance(obj, (AbstractSpace, GroundingSpaceRef, SpaceRef)):
        s = obj
        f = getattr(s,"atom_count", None)
        if f is not None: prefix_print(prefix+" atom-count:", f())
        f = getattr(s,"get_atoms", None)
        if f is not None:
            prefix_print(prefix+" ", f())
            return
        f = getattr(s,"atoms_iter", None)
        if f is not None:
            prefix_print(prefix+" ", f())
            return
        f = getattr(s,"query", None)
        if f is not None:
            prefix_print(prefix+" ", f(V("_")))
            return

    if isinstance(obj, (int, float)):
        prefix_print(prefix+" ",repr(obj))
        return

    try:
        if hasattr(obj, '__next__'):  # Check if obj is an iterator
            while True:
                try:
                    prefix_print(prefix+" ", next(obj))
                except StopIteration:
                    break
        else:
            for r in obj:             # Check if obj is an iteratable
                prefix_print(prefix+" ", r)

        return

    except TypeError: ""


    if isinstance(obj, (Term, Variable)):
        fn = Functor("writeln")
        print(prefix, end=' tv:')
        call(fn(obj))
        return

    if hasattr(obj, '__str__'):
        prefix_print(prefix+" s:",str(obj))
        return

    if isinstance(obj, (Functor, PySwipAtom)):
        fn = Functor("writeq")
        print(prefix, end=' q')
        call(fn(obj))
        return

    prefix_print(prefix+" ",repr(obj))





@export_flags(MeTTa=True)
def pt1(obj):
    if isinstance(obj, str):
        print(f"{repr(obj)}", end= " ")
    elif not isinstance(obj, (Term, Variable)):
        print(f" pt: {type(obj)}={str(obj)}={repr(obj)}", end= " ")
        if isinstance(obj, list):
            obj = obj[0]
            print(f" pt(0): {type(obj)}={str(obj)}={repr(obj)}", end= " ")
    else:
        fn = Functor("pp")
        call(fn(obj))
    return obj

@export_flags(MeTTa=True)
def pt(*objs):
    r = objs
    for o in objs:
        if isinstance(o, str):
            print(o, end="")
        else: r= pt1(o)
    print()
    return r


@export_flags(MeTTa=True, op="VSpacePatternOperation")
def test_s(metta_obj):
    circles = Circles()
    pt(metta_obj)
    swip_obj = m2s(circles,metta_obj)
    pt(swip_obj)
    new_mo = s2m(circles,swip_obj)
    pt(new_mo)
    return new_mo


def get_sexpr_input(prmpt):
    expr, inside_quotes, prev_char = "", False, None

    while True:
        line = input(prmpt)
        flush_console()
        for char in line:
            if char == '"' and prev_char != '\\':
                inside_quotes = not inside_quotes
            expr += char
            prev_char = char

        if not inside_quotes and expr.count("(") == expr.count(")"):
            break
        prmpt = "continue...>>> "
        expr += " "

    return expr

@export_flags(MeTTa=True)
def sync_space(named):
    ""

def the_running_metta_space():
    global the_new_runner_space
    global the_python_runner
    #if the_python_runner.parent!=the_python_runner:
    #    return the_python_runner.parent.space()
    if the_new_runner_space is not None: return the_new_runner_space
    return the_python_runner.parent.space()

# Borrowed impl from Adam Vandervorst
import os
from importlib import import_module
import hyperonpy as hp
from hyperon.atoms import Atom, AtomType, OperationAtom
from hyperon.base import GroundingSpaceRef, Tokenizer, SExprParser

class ExtendedMeTTa(MeTTa):

    def __init__(self, cmetta = None, space = None, env_builder = None):
        self.pymods = {}

        if cmetta is not None:
            self.cmetta = cmetta
        else:
            if space is None:
                space = GroundingSpaceRef()
            if env_builder is None:
                env_builder = hp.env_builder_use_default()
            self.cmetta = hp.metta_new(space.cspace, env_builder)

    #def __init__(self, space = None, cwd = ".", cmetta = None):
    #    if cmetta is not None:
    #        self.cmetta = cmetta
    #    else:
    #        #self.cmetta = None
    #       if space is None:
    #           space = GroundingSpaceRef()
    #       tokenizer = Tokenizer()
    #       self.py_space = space
    #       self.py_tokenizer = tokenizer
    #       self.cmetta = hp.metta_new(self.py_space.cspace, self.py_tokenizer.ctokenizer, cwd)

    def set_cmetta(self, metta):
        if isinstance(metta,MeTTa):
            metta = metta.cmetta
        self.cmetta = metta
        self.load_py_module("hyperon.stdlib")
        hp.metta_load_module(self.cmetta, "stdlib")
        self.register_atom('extend-py!',
            OperationAtom('extend-py!',
                          lambda name: self.load_py_module(name) or [],
                              [AtomType.UNDEFINED, AtomType.ATOM], unwrap=False))
        self.register_atom("transform", OperationAtom("transform", lambda pattern, template: the_running_metta_space().subst(pattern, template),
                                                      type_names=[AtomType.ATOM, AtomType.ATOM, AtomType.UNDEFINED], unwrap=False))
        self.register_atom("join", OperationAtom("join", lambda a, b: interpret(the_running_metta_space(), a) + interpret(the_running_metta_space(), b),
                                                 type_names=[AtomType.ATOM, AtomType.ATOM, AtomType.ATOM], unwrap=False))



    #def __del__(self): hp.metta_free(self.cmetta)

    def space(self):
        return GroundingSpaceRef._from_cspace(hp.metta_space(self.cmetta))

    def tokenizer(self):
        return Tokenizer._from_ctokenizer(hp.metta_tokenizer(self.cmetta))

    #def register_token(self, regexp, constr):
    #    self.tokenizer().register_token(regexp, constr)

    #def register_atom(self, name, symbol):
    #    self.register_token(name, lambda _: symbol)

    def _parse_all(self, program):
        parser = SExprParser(program)
        while True:
            atom = parser.parse(self.tokenizer())
            if atom is None:
                break
            yield atom

    def parse_all(self, program):
        return list(self._parse_all(program))

    def parse_single(self, program):
        return next(self._parse_all(program))


    def load_py_module(self, name):
        """Loads the given python module"""
        if not isinstance(name, str):
            name = repr(name)
        try:
            mod = import_module(name)
            self.pymods[name] = mod
            for n in dir(mod):
                obj = getattr(mod, n)
                if '__name__' in dir(obj) and obj.__name__ == 'metta_register':
                    obj(self)
            return mod
        except:
            return None

    def import_file(self, fname):
        """Loads the program file and runs it"""
        path = fname.split(os.sep)
        if len(path) == 1:
            path = ['.'] + path
        f = open(os.sep.join(path), "r")
        program = f.read()
        f.close()
        # changing cwd
        # TODO: Changing the working dir will not be necessary when the stdlib ops can access the correct runner context.  See https://github.com/trueagi-io/hyperon-experimental/issues/410
        prev_cwd = os.getcwd()
        os.chdir(os.sep.join(path[:-1]))
        result = self.run(program)
        # restoring cwd
        os.chdir(prev_cwd)
        return result


    def run(self, program, flat=False):
        """Runs the program"""
        parser = SExprParser(program)
        results = hp.metta_run(self.cmetta, parser.cparser)
        err_str = hp.metta_err_str(self.cmetta)
        if (err_str is not None):
            raise RuntimeError(err_str)
        if flat:
            return [Atom._from_catom(catom) for result in results for catom in result]
        else:
            return [[Atom._from_catom(catom) for catom in result] for result in results]


# Borrowed impl from Adam Vandervorst
class LazyMeTTa(ExtendedMeTTa):

    #def __init__(self, space = None, cwd = ".", cmetta = None):
    #    super.__init__(space, cwd, cmetta)

    def lazy_import_file(self, fname):
        path = fname.split(os.sep)
        with open(os.sep.join(self.cwd + path), "r") as f:
            program = f.read()
        self.lazy_run(self._parse_all(program))

    def lazy_run(self, stream):
        for i, (expr, result_set) in enumerate(self.lazy_run_loop(stream)):
            if result_set:
                print(f"> {color_expr(expr)}")
                for result in result_set:
                    print(color_expr(result))
            else:
                print(f"> {color_expr(expr)} /")

    def lazy_run_loop(self, stream):
        interpreting = False
        commented = False
        for expr in stream:
            if expr == S('!') and not commented:
                interpreting = True
            elif expr == S('/*'):
                commented = True
            elif expr == S('*/'):
                commented = False
            elif interpreting and not commented:
                yield expr, interpret(the_running_metta_space(), expr)
                interpreting = False
            elif not commented:
                the_running_metta_space().add_atom(expr)


def split_or_none(s, delimiter):
    parts = s.split(delimiter, 1)  # split only at the first occurrence
    return parts[0], (parts[1] if len(parts) > 1 else None)

# from awakening health
def _response2bindings(txt):
        res = re.findall(r'\{.*?\}', txt)
        new_bindings_set = BindingsSet.empty()
        if res == []:
            return new_bindings_set
        res = res[0][1:-1]
        _var, val = res.split(':')
        var = re.findall(r'\".*?\"', _var)
        var = var[0][1:-1] if len(var) > 0 else _var.replace(" ", "")
        if var[0] == '$':
            var = var[1:]
        var = V(var)
        try:
            val = ValueAtom(int(val))
            bindings = Bindings()
            bindings.add_var_binding(var, val)
            new_bindings_set.push(bindings)
        except ValueError:
            ss = re.findall(r'\".*?\"', val)
            if ss == []:
                ss = ['"' + val + '"']
            for s in ss:
                val = S(s[1:-1])
                bindings = Bindings()
                bindings.add_var_binding(var, val)
                new_bindings_set.push(bindings)
        return new_bindings_set

# from awakening health
class GptSpace(GroundingSpace):
    def query(self, query_atom):
        tot_str = "Answer the question taking into account the following information (each fact is in brackets):\n"
        for atom in self.atoms_iter():
            tot_str += str(atom) + "\n"
        tot_str += "If the question contains letters in brackets with $ sign, for example ($x), provide the answer in the json format in curly brackets, that is { $x: your answer }.\n"
        # tot_str += "If information is not provided, return the entry to be queried in JSON {unknown value: UNKNOWN}."
        tot_str += "The question is: " + str(query_atom)[1:-1] + "?"
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{'role': 'system', 'content': 'Reason carefully about user request'},
                    {'role': "user", "content": tot_str}],
                temperature=0)
        txt = response['choices'][0]['message']['content']
        return _response2bindings(txt)

    def copy(self):
        return self

# from awakening health
class GptIntentSpace(GroundingSpace):
    def query(self, query_atom):
        tot_str = "Analyze the topic of the utterance: " + str(query_atom)[1:-1] + "\n"
        tot_str += "Try to pick the most relevant topic from the following list (each topic in brackets):"
        for atom in self.atoms_iter():
            tot_str += str(atom) + "\n"
        tot_str += "If neither of the listed topics seems relevant, answer (chit-chat)."
        tot_str += "Provide the answer in the json format in curly brackets in the form { topic: your answer }.\n"
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{'role': 'system', 'content': 'Reason carefully about user request'},
                    {'role': "user", "content": tot_str}],
                temperature=0)
        txt = response['choices'][0]['message']['content']
        return _response2bindings(txt)

    def copy(self):
        return self


def self_space_info():
    return ""

def wrapsqlop(func):
    def wrapper(*args):
        if len(args) > 0 and isinstance(args[0], GroundedAtom) and isinstance(args[0].get_object(), SpaceRef):
            space = args[0].get_object()
            args = args[1:]
            a = [repr(arg) if isinstance(arg, SymbolAtom) else arg.get_object().value for arg in args]
            res = func(space, *a)
            return [ValueAtom(val) for val in res]
        return []

    return wrapper


from hyperon.atoms import *
from hyperon.ext import register_atoms

import psycopg2
from hyperon import *
from hyperon.ext import register_atoms
import re


def results2bindings(vars, values):
    new_bindings_set = BindingsSet.empty()
    if len(values) == 0 or len(vars) != len(values[0]):
        return new_bindings_set

    for value in values:
        bindings = Bindings()
        for i in range(len(vars)):
            bindings.add_var_binding(vars[i], ValueAtom(str(value[i])))
        new_bindings_set.push(bindings)

    return new_bindings_set


class SqlHelper:
    colums_word = "ColumnNames"
    insert_command_sql = "INSERT INTO"

    @staticmethod
    def get_query_atoms(query_atom):
        children = query_atom.get_children()
        new_query_atoms = []
        for ch in children:
            if 'limit' not in repr(ch).lower():
                new_query_atoms.append(ch)
        return new_query_atoms

    @staticmethod
    def get_fields_and_conditions(query_atom):
        ''' parse sql query and get columns to select and conditions for filtering '''
        atoms = query_atom.get_children()
        fields = {}
        conditions = {}
        limit = ""
        vars_map = {}
        for atom in atoms:
            if isinstance(atom, ExpressionAtom):
                items = atom.get_children()
                if len(items) == 3:
                    id_fields = items[1].get_children()
                    current_field_info = items[2].get_children()
                    if len(id_fields) != 2 or len(current_field_info) != 2:
                        raise SyntaxError("Incorrect number of arguments")
                    # (musicbrainz.artist (id $id) (name $name))
                    # identification field
                    id_name = repr(id_fields[0])
                    vars_map[id_name] = repr(id_fields[1])
                    # field to select
                    field_name = repr(current_field_info[0])
                    vars_map[field_name] = repr(current_field_info[1])
                    # table
                    table = repr(items[0])
                    if table not in fields:
                        fields[table] = set()
                    if table not in conditions:
                        conditions[table] = set()
                    # add id field to corresponding category (filed/condition)
                    if isinstance(id_fields[1], VariableAtom):
                        fields[table].add(id_name)
                    else:
                        conditions[table].add(id_name)
                    # add selected field to corresponding category (filed/condition)
                    if isinstance(current_field_info[1], VariableAtom):
                        fields[table].add(field_name)
                    else:
                        conditions[table].add(field_name)

                if len(items) == 2 and ("limit" in repr(items[0]).lower()):
                    limit = repr(items[1])
        return fields, conditions, limit, vars_map

    @staticmethod
    def get_fields_and_values(query_atom):
        ''' parse sql query and get columns to select and conditions for filtering '''
        atoms = query_atom.get_children()
        fields_map = {}
        for atom in atoms:
            if isinstance(atom, ExpressionAtom):
                items = atom.get_children()
                if len(items) != 2:
                    raise SyntaxError("Incorrect number of arguments")
                # (musicbrainz.artist (id $id) (name $name)
                # field to select
                field_name = repr(items[0])
                fields_map[field_name] = repr(items[1])
        return fields_map

    def save_query_result(self, sql_space, space, query_atom):
        # if no fields provided get them from information_schema.columns
        res = sql_space.query(query_atom)
        variables = []
        for val in res:
            temp_dict = {}
            for k, v in val.items():
                temp_dict['$' + str(k)] = str(v)
            variables.append(temp_dict)
        atoms = self.get_query_atoms(query_atom)
        new_atoms = []
        for var in variables:
            for atom in atoms:
                if isinstance(atom, ExpressionAtom):
                    temp = repr(atom)
                    for k, v in var.items():
                        temp = temp.replace(k, v)
                    new_atoms.append(temp)
        for atom in new_atoms:
            space.add_atom(E(S(atom)))
        return res

    def insert(self, space, query_atom):
        fields_map = SqlHelper.get_fields_and_values(query_atom)
        res = []
        table = fields_map.pop("table")
        values = []
        for field_name, field_value in fields_map.items():
            values.append(field_value.replace('"', "") if "(" in field_value and field_value[-2] == ')'
                          else field_value.replace('"', "'"))
        fields_str = ", ".join(list(fields_map.keys()))
        values_str = ", ".join(list(values))
        query = f'''{self.insert_command_sql} {table} ({fields_str}) VALUES ({values_str}) RETURNING 0;'''
        res.extend(space.query(E(S(query))))
        return res

#@ MeTTa
class SqlSpace(GroundingSpace):
    def __init__(self, database, host, user, password, port):
        super().__init__()
        self.conn = psycopg2.connect(database=database,
                                     host=host,
                                     user=user,
                                     password=password,
                                     port=port)
        self.cursor = self.conn.cursor()

    def from_space(self, cspace):
        self.gspace = GroundingSpaceRef(cspace)

    def construct_query(self, query_atom):
        fields, conditions, limit, vars_map = SqlHelper.get_fields_and_conditions(query_atom)
        sql_query = "SELECT"

        vars_names = []
        for k, values in fields.items():
            for val in values:
                sql_query = sql_query + f" {k}.{val},"
                vars_names.append(vars_map[val])
        sql_query = sql_query[:-1] + " FROM "
        for k in fields.keys():
            sql_query = sql_query + f"{k},"

        sql_condition = " WHERE"
        for k, values in conditions.items():
            for val in values:
                if val in vars_map:
                    sql_condition = sql_condition + f" {k}.{val} = {vars_map[val]} AND"
        if len(sql_condition) > 6:
            sql_query = sql_query[:-1] + sql_condition[:-4]
        else:
            sql_query = sql_query[:-1]
        if len(limit) > 0:
            sql_query = sql_query + f" LIMIT {limit}"
        return sql_query, vars_names

    def insert(self, sql_query):
        try:
            if len(sql_query) > 6:
                self.cursor.execute(sql_query)
                self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            bindings_set = BindingsSet.empty()
            bindings = Bindings()
            bindings.add_var_binding("error on insert: ", ValueAtom(error))
            bindings_set.push(bindings)
            return bindings_set
        return BindingsSet.empty()

    def query(self, query_atom):
        try:
            atoms = query_atom.get_children()
            if len(atoms) > 0 and SqlHelper.insert_command_sql in repr(atoms[0]):
                return self.insert(repr(atoms[0]))
            else:
                new_bindings_set = BindingsSet.empty()
                sql_query, vars_names = self.construct_query(query_atom)
                if len(sql_query) > 6:
                    self.cursor.execute(sql_query)
                    values = self.cursor.fetchall()
                    if len(vars_names) == 0 and len(values) > 0:
                        vars = [f"var{i + 1}" for i in range(len(values[0]))]
                    else:
                        vars = [v[1:] for v in vars_names]
                    if len(vars) > 0 and len(values) > 0:
                        return results2bindings(vars, values)
                return new_bindings_set
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)


def wrapsqlop(func):
    def wrapper(*args):
        if len(args) > 1:
            if isinstance(args[0], GroundedAtom):
                space1 = args[0].get_object()
                if isinstance(space1, SpaceRef):
                    if isinstance(args[1], GroundedAtom):
                        space2 = args[1].get_object()
                        if isinstance(space2, SpaceRef):
                            args = args[2:]
                            res = func(space1, space2, *args)
                            return [ValueAtom(val) for val in res]
                    else:
                        args = args[1:]
                        res = func(space1, *args)
                        return [ValueAtom(val) for val in res]
        return []

    return wrapper


@register_atoms
def sql_space_atoms():
    helper = SqlHelper()
    newSQLSpaceAtom = OperationAtom('new-sql-space', lambda database, host, user, password, port: [
        G(SpaceRef(SqlSpace(database, host, user, password, port)))], unwrap=False)
    saveQueryResult = G(OperationObject('sql.save-query-result', wrapsqlop(helper.save_query_result), unwrap=False))
    sqlInsert = G(OperationObject('sql.insert', wrapsqlop(helper.insert), unwrap=False))
    return {
        r"new-sql-space": newSQLSpaceAtom,
        r"sql.save-query-result": saveQueryResult,
        r"sql.insert": sqlInsert
    }


def pl_select(*args):
    print_cmt("pl_select: ",args)
    flush_console()

def pl_insert(*args):
    print_cmt("pl_insert: ",args)
    flush_console()

def np_array(args):
    print_cmt("np_array=",args)
    return np.array(args)

def np_vector(*args):
    print_cmt("np_vector=",args)
    return np.array(args)

realMetta = None

def metta_register(metta):
    print(f";; metta_register={the_python_runner}/{metta}")
    global realMetta
    try:
        if not metta is None:
            if not isinstance(metta,ExtendedMeTTa):
                realMetta = metta
            register_vspace_atoms_pm(metta)
    #the_python_runner.set_cmetta(metta)
    #print(";;", metta.pymods)
        print(";;", the_python_runner.pymods)

    except Exception as e:
        if verbose>0: print(f"; Error: {e}")


@register_atoms
def register_vspace_atoms():
    register_vspace_atoms_pm(None)

@export_flags(MeTTa=True)
@register_atoms(pass_metta=True)
def register_vspace_atoms_pm(mettaIn):

    global realMetta

    if mettaIn is None:
         mettaIn = realMetta

    metta = mettaIn

    global oper_dict
    if verbose>1: print_cmt(f"register_vspace_atoms metta={metta} the_python_runner = {the_python_runner} {self_space_info()}")

    counter = 0
    #if not metta is None: the_python_runner.set_cmetta(metta)

    if not isinstance(metta, VSpace):
        if not metta is None:
        	the_python_runner.parent = metta

    def new_value_atom_func():
        nonlocal counter
        counter += 1
        return [ValueAtom({'A': counter, 6: 'B'})]

    # We don't add types for operations, because numpy operations types are too loose
    nmVectorAtom = G(VSPatternOperation('np.array', wrapnpop(lambda *args: np_array(args)), unwrap=False))
    nmArrayAtom = G(VSPatternOperation('np.vector', wrapnpop(lambda *args: np_vector(*args)), unwrap=False, rec=True))
    nmUVectorAtom = G(VSPatternOperation('np.uarray', wrapnpop(lambda *args: np_array(args)), unwrap=False))
    nmUArrayAtom = G(VSPatternOperation('np.uvector', wrapnpop(lambda *args: np_vector(*args)), unwrap=False, rec=True))
    nmAddAtom = G(VSPatternOperation('np.add', wrapnpop(np.add), unwrap=False))
    nmSubAtom = G(VSPatternOperation('np.sub', wrapnpop(np.subtract), unwrap=False))
    nmMulAtom = G(VSPatternOperation('np.mul', wrapnpop(np.multiply), unwrap=False))
    nmDivAtom = G(VSPatternOperation('np.div', wrapnpop(np.divide), unwrap=False))
    nmMMulAtom = G(VSPatternOperation('np.matmul', wrapnpop(np.matmul), unwrap=False))

    testS = G(VSpacePatternOperation('test-s', wrapnpop(test_s), rec = True, unwrap=False))

    # DMILES:  I actujally like the behaviour below.
    newValueAtom = OperationAtom('new-value-atom', new_value_atom_func, unwrap=False)
    # (new-value-atom)
    # this was stored in the space..
    # !(match &self $ $)
    # and each time the space is matched the counter will be incremented
    # !(match &self $ $)
    # !(match &self $ $)
    # (new-value-atom)
    # they share a counter
    # !(match &self $ $)

    runnerAtom = G(the_python_runner, AtomType.ATOM)
    add_python_module(sys.modules[__name__], dict = oper_dict)
    oper_dict.update({
        r"TupleCount": OperationAtom( 'TupleCount', lambda atom: [ValueAtom(len(atom.get_children()), 'Number')], [AtomType.ATOM, "Number"], unwrap=False),
        r"np\.vector": nmVectorAtom,
        r"np\.array": nmArrayAtom,
        r"np\.uvector": nmUVectorAtom,
        r"np\.uarray": nmUArrayAtom,
        r"np\.add": nmAddAtom,
        r"np\.sub": nmSubAtom,
        r"np\.mul": nmMulAtom,
        r"np\.matmul": nmMMulAtom,
        r"test_s": testS,

        r"np\.div": nmDivAtom,

        r"new-gpt-space": OperationAtom('new-gpt-space', lambda: [G(SpaceRef(GptSpace()))], unwrap=False),
        r"new-gpt-intent-space": OperationAtom('new-gpt-intent-space', lambda: [G(SpaceRef(GptIntentSpace()))], unwrap=False),

        r"new-v-space": OperationAtom('new-v-space', lambda: [G(SpaceRef(VSpace()))], unwrap=False),
        r"the-v-space": OperationAtom('the-v-space', lambda: [G(SpaceRef(the_verspace))], unwrap=False),


        r"new-value-atom": newValueAtom,
        #'&self': runnerAtom,
        #'&swip': ValueAtom(swip),



       'pl_select': G(OperationObject('pl_select', wrapsqlop(pl_select), unwrap=False)),
       'pl_insert': G(OperationObject('pl_insert', wrapsqlop(pl_insert), unwrap=False)),


        '&my-dict': ValueAtom({'A': 5, 6: 'B'}),
        'get-by-key': OperationAtom('get-by-key', lambda d, k: d[k]),

        # Our FFI to PySWIP
        #'load-vspace': OperationAtom('load-vspace', lambda: [load_vspace()]),
        'mine-overlaps': OperationAtom('mine-overlaps', lambda: [mine_overlaps()]),
        'try-overlaps': OperationAtom('try-overlaps', lambda: [try_overlaps()]),
        'load-flybase-full': OperationAtom('load-flybase-full', lambda: [load_flybase("inf")]),
        'load-flybase-tiny': OperationAtom('load-flybase-tiny', lambda: [load_flybase(20000)]),

        r"fb.test-nondeterministic-foreign": OperationAtom('test-nondeterministic-foreign', lambda: test_nondeterministic_foreign, unwrap=False),

        'vspace-main': OperationAtom('vspace-main', vspace_main),
        'metta_learner::vspace-main': OperationAtom('vspace-main', lambda *args: [vspace_main(*args)]),
        'swip-exec': OperationAtom('swip-exec', lambda s: [swip_exec(s)]),
        'py-eval': OperationAtom('py-eval', lambda s: [eval(s)]) })

    add_python_module(sys.modules[__name__], dict = oper_dict)
    return oper_dict


# For now lets test with only  Atoms
@register_tokens(pass_metta=True)
def register_vspace_tokens(metta):

    if verbose>1: print_cmt(f"register_vspace_tokens metta={metta} {self_space_info()}")

    if hasattr(the_python_runner,"set_cmetta"):
        the_python_runner.set_cmetta(metta.cmetta)

    if hasattr(the_python_runner,"cmetta"):
        the_python_runner.cmetta = metta.cmetta


    if not isinstance(metta, VSpace):
        the_python_runner.parent = metta

    def run_resolved_symbol_op(the_python_runner, atom, *args):
        expr = E(atom, *args)
        if verbose>1: print_cmt(f"run_resolved_symbol_op: atom={atom}, args={args}, expr={expr} metta={metta} {self_space_info()}")
        result1 = hp.metta_evaluate_atom(the_python_runner.cmetta, expr.catom)
        result = [Atom._from_catom(catom) for catom in result1]
        if verbose>1: print_cmt(f"run_resolved_symbol_op: result1={result1}, result={result}")
        return result

    def resolve_atom(metta, token):
        return _resolve_atom(metta, token, verbose)

    def _resolve_atom(metta, token, verbose):
        # TODO: nested modules...
        verbose = verbose +1

        if token is None: return token

        if verbose>1: print_cmt(f"resolve_atom: token={token}/{type(token)} metta={metta}")

        if "::" in token:
            runner_name, atom_name = token.split('::')
        else:
            atom_name = token
            runner_name = ""

        if atom_name in oper_dict:
            if verbose>1: print_cmt(f"resolve_atom: token={token} metta={metta}")
            return oper_dict[atom_name]

        atom_name2 = atom_name.replace('_', '-')

        if atom_name2 in oper_dict:
            if verbose>0: print_cmt(f"resolve_atom: token={token} metta={metta}")
            return oper_dict[atom_name2]

        if atom_name=="vspace-main":
            vspace_main()
            return
        # FIXME: using `run` for this is an overkill
        ran = metta.run('! ' + runner_name)[0][0];
        if verbose>1: print_cmt(f"resolve_atom: token={token} ran={type(ran)} metta={metta} {self_space_info()}")
        try:
            this_runner = ran.get_object()
        except Exception as e:
            this_runner = the_python_runner
            # If there's an error, print it
            #print(f"Error ran.get_object: {e}")

        #if !isinstance(this_runner, MeTTa):

        #if !isinstance(this_runner, MeTTa): this_runner = metta


        atom = this_runner.run('! ' + atom_name)[0][0]
        # A hack to make the_python_runner::&self work
        # TODO? the problem is that we need to return an operation to make this
        # work in parent expressions, thus, it is unclear how to return pure
        # symbols
        if atom.get_type() == hp.AtomKind.GROUNDED:
            return atom

        # TODO: borrow atom type to op
        return OperationAtom( token, lambda *args: run_resolved_symbol_op(the_python_runner, atom, *args), unwrap=False)

    def resolve_underscores(metta, token):
        atom_name = token.replace('_', '-')
        if atom_name in oper_dict:
            if verbose>1: print_cmt(f"resolve_atom: token={token} metta={metta}")
            return oper_dict[atom_name]

    syms_dict.update({
        '&gptspace': lambda _: G(asSpaceRef(getSpaceByName('&gptspace'))),
        '&flybase': lambda _: G(asSpaceRef(getSpaceByName('&flybase'))),
        #'&vspace': lambda _: G(getSpaceByName('&vspace')),
        #'&vbase_class': lambda _: G((the_verspace)),
        '&parent_ref': lambda _: G(asSpaceRef(getSpaceByName("&parent"))),
        '&parent': lambda _: G((getSpaceByName("&parent"))),
        '&child': lambda _: G((getSpaceByName("&child"))),
        '&the_runner': lambda _: ValueAtom(the_python_runner),
        '&the_metta': lambda _: ValueAtom(the_python_runner.parent),
        r"[^\s]+::[^\s]+": lambda token: resolve_atom(metta, token)
        #r"[^\s][^\s]+[^!]": lambda token: resolve_atom(metta, token)
        #r"[^\s]+_[^\s]+": lambda token: resolve_underscores(metta, token)
    })
    for key in syms_dict:
        if key.startswith("&"):
            add_to_history_if_unique(f"!{key}")
    return syms_dict

def res_unify(s,v):
  if isinstance(v, str):
      if isinstance(s, str):
          return s == v
      return s.unify(swipAtom(v))
  return s.unify(v)

# Define the foreign functions
@export_flags(Janus=True)
def query_from_space(space_name, query_atom, result):
    space = getSpaceByName(space_name)
    if space:
        atoms = space.query(query_atom)
        return res_unify(result,atoms)
    return False

@export_flags(Janus=True)
def add_to_space(space_name, atom):
    space = getSpaceByName(space_name)
    if space:
        circles = Circles()
        atom = s2m(circles,atom)
        if isinstance(space, SpaceRef):
            return space.add_atom(atom)
        return space.add(atom)
    return False

@export_flags(Janus=True)
def remove_from_space(space_name, atom):
    space = getSpaceByName(space_name)
    if space:
        circles = Circles()
        atom = s2m(circles,atom)
        return space.remove(atom)
    return False

@export_flags(Janus=True)
def replace_in_space(space_name, from_atom, to_atom):
    space = getSpaceByName(space_name)
    if space:
        circles = Circles()
        to_atom = s2m(circles,to_atom)
        from_atom = s2m(circles,from_atom)
        return space.replace(from_atom, to_atom)
    return False

@export_flags(Janus=True)
def atom_count_from_space(space_name, result):
    space = getSpaceByName(space_name)
    if space:
        return res_unify(result,space.atom_count())
    return False

@export_flags(Janus=True)
def get_atoms_from_space(space_name, result):
    space = getSpaceByName(space_name)
    if space:
        circles = Circles()
        atoms = list(space.get_atoms())
        satoms = [m2s(circles, atom) for atom in atoms]
        return res_unify(result,satoms)
    return False

context_atom_iters = {}

def MkExpr(py_list):
    list = py_list
    return E(*list)

class IteratorAndConversionDict:
    def __init__(self, iterator=None, conversion_dict=None):
        self.iterator = iterator
        self.conversion_dict = conversion_dict if conversion_dict is not None else {}

    def set_iterator(self, iterator):
        self.iterator = iterator

    def set_conversion_dict(self, conversion_dict):
        self.conversion_dict = conversion_dict

    def get_iterator(self):
        return self.iterator

    def get_conversion_dict(self):
        return self.conversion_dict

@export_flags(Janus=True)
def get_atoms_iter_from_space(space_name):
    space = getSpaceByName(space_name)
    if space:
        get_iterator =  getattr(space,"atoms_iter", None) # Create a new iterator
        if get_iterator is not None:
            return get_iterator()
        else:
            get_iterator =  getattr(space,"get_atoms", None) # Create a new iterator
            if get_iterator is not None:
                return iter(get_iterator)
            else:
                V = V("X")
                iterator =  space.query(V) # Create a new iterator
                return iterator


@export_flags(Janus=True, arity=2, flags=PL_FA_NONDETERMINISTIC)
def atoms_iter_from_space(space_name, result, context):
    global idKey, context_atom_iters
    control = PL_foreign_control(context)
    context = PL_foreign_context(context)
    id = context

    if control == PL_FIRST_CALL:
        id = idKey
        idKey= idKey+1
        iterator = get_atoms_iter_from_space(space_name)
        if iterator is not None:
            try:
                circles = Circles()
                while True:
                    value = next(iterator)
                    if res_unify(result,m2s(circles,value)):
                        context_atom_iters[id] = IteratorAndConversionDict(iterator,circles)  # Store it in the dictionary
                        return PL_retry(context)
                    return PL_retry(context)
            except StopIteration:
                del context_atom_iters[id]  # Clean up
        return False

    elif control == PL_REDO:
        iteratorAndCircs = context_atom_iters.get(id)
        if iteratorAndCircs is not None:
            try:
                iterator = iteratorAndCircs.get_iterator()
                circles = iteratorAndCircs.get_conversion_dict()
                while True:
                    value = next(iterator)
                    if res_unify(result,m2s(circles,value)):
                        return PL_retry(context)
                del context_atom_iters[id]  # Clean up
                return False
            except StopIteration:
                del context_atom_iters[id]  # Clean up
                return False
        pass

    elif control == PL_PRUNED:
        # Clean up the iterator when we're done
        if id in context_atom_iters:
            del context_atom_iters[id]
        pass

@export_flags(Janus=True)
def add_to_history_if_unique_pl(item, position_from_last=1):
    for i in range(1, readline.get_current_history_length() + 1):
        if readline.get_history_item(i) == item: return
    insert_to_history(item, position_from_last)


def reg_pyswip_foreign():

    add_janus_methods(sys.modules[__name__], dict = oper_dict)
    test_nondeterministic_foreign()


    def py_eval(e, result):
        return res_unify(result,eval(str(e)))
    py_eval.arity = 2
    registerForeign(py_eval)

    # Register the foreign functions in PySwip
    #registerForeign(new_rust_space, arity=1)
    #registerForeign(query_from_space, arity=3)
    #registerForeign(add_to_space, arity=2)
    #registerForeign(remove_from_space, arity=2)
    #registerForeign(replace_in_space, arity=3)
    #registerForeign(atom_count_from_space, arity=2)
    #registerForeign(atoms_iter_from_space, arity=2)
    #registerForeign(get_atoms_from_space, arity=2)
    add_to_history_if_unique.arity=1
    registerForeign(add_to_history_if_unique)



    #?- query_from_space('example', 'my_atom', Result).
    #?- add_to_space('example', 'new_atom').
    #?- remove_from_space('example', 'some_atom').
    #?- replace_in_space('example', 'old_atom', 'new_atom').
    #?- atom_count_from_space('example', Count).
    #?- atoms_iter_from_space('example', Atoms).


@export_flags(Janus=True)
def find_rust_space(space_name, result):
    space = getSpaceByName(space_name)
    named = getNameBySpace(space)
    if space:
        return res_unify(result,named)
    return False

rustspace_ordinal = 0
@export_flags(Janus=True)
def new_rust_space(result):
    rustspace_ordinal=rustspace_ordinal+1
    name = f"&vspace_{rustspace_ordinal}"
    space = GroundingSpace()
    addSpaceName(name,space)
    return res_unify(result,swipAtom(name))


@export_flags(MeTTa=True, Janus=True)
def test_nondeterministic_foreign1():

    def nondet(a, context):
        control = PL_foreign_control(context)
        context = PL_foreign_context(context)
        if control == PL_FIRST_CALL:
            context = 0
            a.unify(int(context))
            context += 1
            return PL_retry(context)
        elif control == PL_REDO:
            a.unify(int(context))
            if context == 10:
                return False
            context += 1
            return PL_retry(context)
        elif control == PL_PRUNED:
            pass


    nondet.arity = 1
    registerForeign(nondet, flags=PL_FA_NONDETERMINISTIC)
    result = list(swip.query("nondet(X)"))

    print_cmt(result)

    if len(result) != 10:
        print_cmt('Query should return 10 results')

    for i in range(10):
        if {'X': i} not in result:
            print_cmt('Expected result X:{} not present'.format(i))


@export_flags(MeTTa=True, Janus=True)
def test_nondeterministic_foreign2():

    def number_generator():
        for i in range(10):
            yield i

    iterator = number_generator()

    def nondet2(a, context):
        control = PL_foreign_control(context)
        context = PL_foreign_context(context)
        #global iterator  # Use the global iterator object

        if control == PL_FIRST_CALL:
            try:
                value = next(iterator)  # Start the iterator
                a.unify(int(value) + 1)  # Add 1 to yield numbers from 1 to 10
                return PL_retry(context)
            except StopIteration:
                return False
        elif control == PL_REDO:
            try:
                value = next(iterator)
                a.unify(int(value) + 1)  # Add 1 to yield numbers from 1 to 10
                return PL_retry(context)
            except StopIteration:
                return False
        elif control == PL_PRUNED:
            pass

    nondet2.arity = 1
    registerForeign(nondet2, flags=PL_FA_NONDETERMINISTIC)
    result = list(swip.query("nondet2(X)"))

    print_cmt(result)

    if len(result) != 10:
        print_cmt('Query should return 10 results')

idKey = 1
@export_flags(MeTTa=True, Janus=True)
def test_nondeterministic_foreign3():

    def number_generator(size):
        for i in range(size):
            yield i

    context_iterators = {}  # Dictionary to store iterators by context

    def nondet3(sz, a, context):
        global idKey
        control = PL_foreign_control(context)
        context = PL_foreign_context(context)
        id = context

        if control == PL_FIRST_CALL:
            id = idKey
            idKey= idKey+1
            iterator = number_generator(sz)  # Create a new iterator
            context_iterators[id] = iterator  # Store it in the dictionary
            try:
                value = next(iterator)
                a.unify(int(value) + 1)
                context = id
                return PL_retry(context)
            except StopIteration:
                del context_iterators[id]  # Clean up
                return False

        elif control == PL_REDO:
            iterator = context_iterators.get(id)
            if iterator is not None:
                try:
                    value = next(iterator)
                    a.unify(int(value) + 1)
                    return PL_retry(context)
                except StopIteration:
                    del context_iterators[id]  # Clean up
                    return False
            pass

        elif control == PL_PRUNED:
            # Clean up the iterator when we're done
            if id in context_iterators:
                del context_iterators[id]
            pass

    nondet3.arity = 2
    registerForeign(nondet3, arity=2, flags=PL_FA_NONDETERMINISTIC)
    result = list(swip.query("nondet3(4,X)"))

    print_cmt(result)

    if len(result) != 4:
        print_cmt('nondet3 should return 4 results')


@export_flags(MeTTa=True, Janus=True)
def test_nondeterministic_foreign():

    test_nondeterministic_foreign1()
    test_nondeterministic_foreign2()
    test_nondeterministic_foreign3()


    def hello(t):
        print_cmt("Hello,", t)

    hello.arity = 1

    registerForeign(hello, arity=1)


    def hello1(t):
        readline.replace_history_item(0, t)
        print_cmt("Hello1,", t)


    hello1.arity = 1

    registerForeign(hello1, arity=1)




    swip.assertz("father(michael,john)")
    swip.assertz("father(michael,gina)")

    result = list(swip.query("father(michael,X), hello(X)"))

    print_cmt(result)

    if len(result) != 2:
        print_cmt('Query should return two results')
    for name in ('john', 'gina'):
        if {'X': name} not in result:
            print_cmt('Expected result X:{} not present'.format(name))


    #def test_atoms_and_strings_distinction(self):
    test_string = "string"

    def get_str(string):
        string.value = test_string

    def test_for_string(string, test_result):
        test_result.value = (test_string == string.decode('utf-8'))

    get_str.arity = 1
    test_for_string.arity = 2

    registerForeign(get_str)
    registerForeign(test_for_string)

    result = list(swip.query("get_str(String), test_for_string(String, Result)"))

    print_cmt(result)

    if result[0]['Result'] != 'true':
          print_cmt('A string return value should not be converted to an atom.')

    print_cmt()
    print_cmt()
    print_cmt()
    flush_console()



@export_flags(Janus=True)
def swip_to_metta_wrapper(swip_obj, metta_obj):
    circles = Circles()
    result1 = m2s(circles,s2m(circles,swip_obj))
    result2 = m2s(circles,metta_obj)
    #metta_obj.unify(m2s(circles,result))
    return result2.unify(result1)
    #return True

@export_flags(Janus=True)
def metta_to_swip_wrapper(metta_obj, swip_obj):
    circles = Circles()
    result1 = m2s(circles,metta_obj)
    result2 = m2s(circles,swip_obj)
    #swip_obj.unify(result)
    return result2.unify(result1)
    #return True

@export_flags(MeTTa=True)
def metta_to_swip_tests1():
    # Register the methods as foreign predicates
    registerForeign(swip_to_metta_wrapper, arity=2)
    registerForeign(metta_to_swip_wrapper, arity=2)
    circles = Circles()
    # Usage:
    swip_functor = Functor(PySwipAtom("example"), 2, [PySwipAtom("sub1"), 3.14])
    print_cmt(f"swip_functor={swip_functor}"),
    metta_expr = s2m(circles,swip_functor)
    print_cmt(f"metta_expr={metta_expr}"),
    converted_back_to_swip = m2s(circles,metta_expr)
    print_cmt(f"converted_back_to_swip={converted_back_to_swip}"),


    # Now you can use the methods in PySwip queries
    print_cmt(list(swip.query("swip_to_metta_wrapper('example', X).")))
    print_cmt(list(swip.query("metta_to_swip_wrapper(X, 'example').")))

@export_flags(MeTTa=True)
def metta_to_swip_tests2():
    # Register the methods as foreign predicates
    registerForeign(swip_to_metta_wrapper, arity=2)
    registerForeign(metta_to_swip_wrapper, arity=2)

    circles = Circles()
    # Now you can use the methods in PySwip queries
    println(list(swip.query("swip_to_metta_wrapper('example', X).")))
    println(list(swip.query("metta_to_swip_wrapper(X, 'example').")))

    # Usage:
    swip_list = ["a", "b", 3]
    metta_expr = s2m(circles,swip_list)
    converted_back_to_swip = m2s(circles,metta_expr)
    swip_functor = Functor(PySwipAtom("example"), 2, [PySwipAtom("sub1"), 3.14])
    metta_expr = s2m(circles,swip_functor)
    converted_back_to_swip = m2s(circles,metta_expr)

NeedNameSpaceInSWIP = True
@export_flags(MeTTa=True, unwrap=True)
def load_vspace():
   global NeedNameSpaceInSWIP
   swip_exec(f"ensure_loaded('{os.path.dirname(__file__)}/pyswip/flybase_main')")
   if NeedNameSpaceInSWIP:
       NeedNameSpaceInSWIP = False
       swip.retractall("was_asserted_space('&self')")
       swip.assertz("py_named_space('&self')")

@export_flags(MeTTa=True, CallsVSpace=True)
def mine_overlaps():
   load_vspace()
   swip_exec("mine_overlaps")
   #readline_add_history("!(try-overlaps)")


@export_flags(MeTTa=True, CallsVSpace=True)
def try_overlaps():
   load_vspace()
   swip_exec("try_overlaps")

@export_flags(MeTTa=True, CallsVSpace=True)
def learn_vspace():
   load_vspace()
   swip_exec("learn_vspace(60)")

@export_flags(MeTTa=True, CallsVSpace=True)
def mettalog():
   load_vspace()
   swip_exec("repl")


@export_flags(MeTTa=True)
def register_mettalog_op_new(fn, n):
   arg_types = [AtomType.ATOM] * (n) + [AtomType.UNDEFINED]
   op = OperationAtom(fn, lambda *args:
                  print_cmt(f"eval_mettalog('{fn}', {args})") +
                  eval_mettalog(fn, *args),
                              type_names=arg_types,
                              unwrap=True)
   the_python_runner.register_atom(fn, op)
   return op


@export_flags(MeTTa=True, CallsVSpace=True)
def use_mettalog():
   load_vspace()
   register_mettalog_op("pragma!",2)
   register_mettalog_op("match",3)
   return register_mettalog_op("import!",2)

@export_flags(MeTTa=True)
def register_mettalog_op(fn, n):
    arg_types = [AtomType.ATOM] * (n) + [AtomType.UNDEFINED]
    n_args = ', '.join(['arg' + str(i) for i in range(n)])
    local_vars = {}
    src = f'lop = lambda {n_args}: eval_mettalog("{fn}", {n_args})'
    exec(src, globals(), local_vars)
    lop = local_vars['lop']
    #print_cmt(src) print_cmt(type(the_python_runner))
    op = OperationAtom(fn, lop, type_names=arg_types, unwrap=False)
    oper_dict[fn]=op
    the_python_runner.register_atom(fn, op)

    return op

def eval_mettalog(fn, *args):
    print_cmt(f"eval_mettalog('{fn}', {args})")
    return list(_eval_mettalog(fn,args))

def _eval_mettalog(fn, *args):
    circles = Circles()
    expr = [fn] + list(args) # Prepend fn to args list
    swip_obj = m2s(circles,expr)
    flush_console()
    call_sexpr = Functor("call_sexpr", 5)
    #user = newModule("user")
    X = Variable()
    q = PySwipQ(call_sexpr(argmode,selected_space_name, str(expr), swip_obj, X))
    while q.nextSolution():
      flush_console()
      r = X.value
      println(r)
      yield s2m(circles,r)
    q.closeQuery()
    flush_console()

@export_flags(MeTTa=True, CallsVSpace=True)
def mettalog_pl():
   load_vspace()
   swip_exec("break")

@export_flags(CallsVSpace=True)
def load_flybase(size):
   load_vspace()
   swip_exec(f"load_flybase({size})")
   #readline_add_history("!(mine-overlaps)")

@export_flags(MeTTa=True)
def swip_exec(qry):
    swip_exec_ff(qry)

@foreign_framed
def swip_exec_ff(qry):
    #from metta_vspace import swip
    #if is_init==True:
    #   print_cmt("Not running Query: ",qry)
    #   return
    for r in swip.query(qry):
        print_cmt(r)

@export_flags(MeTTa=True)
def test_custom_m_space():

    class TestSpace(AbstractSpace):

        def __init__(self, unwrap=False):
            super().__init__()
            self.atoms_list = []
            self.unwrap = unwrap

        # NOTE: this is a naive implementation barely good enough to pass the tests
        # Don't take this as a guide to implementing a space query function
        def query(self, query_atom):

            # Extract only the variables from the query atom
            circles = list(filter(lambda atom: atom.get_type() == AtomKind.VARIABLE, query_atom.iterate()))

            # Match the query atom against every atom in the space
            # BindingsSet() creates a binding set with the only matching result
            # We use BindingsSet.empty() to support multiple results
            new_bindings_set = BindingsSet.empty()
            for space_atom in self.atoms_list:
                match_results = space_atom.match_atom(query_atom)

                # Merge in the bindings from this match, after we narrow the match_results to
                # only include variables vars in the query atom
                for bindings in match_results.iterator():
                    bindings.narrow_vars(circles)
                    if not bindings.is_empty():
                        # new_bindings_set.merge_into(bindings) would work with BindingsSet(), but
                        # it would return an empty result for multiple alternatives and merge bindings
                        # for different variables from alternative branches, which would be a funny
                        # modification of query, but with no real use case
                        # new_bindings_set.push(bindings) adds an alternative binding to the binding set
                        new_bindings_set.push(bindings)

            return new_bindings_set

        def add(self, atom):
            self.atoms_list.append(atom)

        def remove(self, atom):
            if atom in self.atoms_list:
                self.atoms_list.remove(atom)
                return True
            else:
                return False

        def replace(self, from_atom, to_atom):
            if from_atom in self.atoms_list:
                self.atoms_list.remove(from_atom)
                self.atoms_list.append(to_atom)
                return True
            else:
                return False

        def atom_count(self):
            return len(self.atoms_list)

        def atoms_iter(self):
            return iter(self.atoms_list)

    test_custom_space(lambda: TestSpace())



# Borrowed impl from Adam Vandervorst
class InteractiveMeTTa(ExtendedMeTTa): # LazyMeTTa ExtendedMeTTa

    def __init__(self):
        super().__init__()
        # parent == self
        #   means no parent MeTTa yet
        self.parent = self
        self.mode = "metta"

    def maybe_submode(self, line):
        lastchar = line[-1]
        if "+-?!^".find(lastchar)>=0:
            self.submode=lastchar

    def repl_loop(self, get_sexpr_input=get_sexpr_input, print_cmt=print_cmt, mode=None):

        global the_new_runner_space
        global selected_space_name
        global verbose
        global argmode

        if mode:
            self.mode = mode
        self.submode = "!"
        #self.history = []
        #load_vspace()

        while True:
            try:
                flush_console()
                # Use the input function to get user input
                prmpt = "; "+self.mode + "@" + selected_space_name+ " " + self.submode + "> "

                line = get_sexpr_input(prmpt)

                #print_cmt(f"You entered: {line}\n")

                if line:
                    sline = line.lstrip().rstrip()
                    add_to_history_if_unique(line, position_from_last=1)
                else:
                    continue

                if len(sline)==1:
                    if "+-?!^".find(sline)>=0:
                        self.maybe_submode(sline)
                        continue

                if sline.endswith(".") and not sline.startswith(";") and not sline.startswith("%"):
                    swip_exec(line)
                    continue


                if not line.startswith(" "):
                        line = " " + line

                if sline.rstrip() == '?':
                    expr = self.parse_single("(match &self $ $)")
                    yield expr, interpret(the_running_metta_space(), expr)
                    continue

                # Check for history commands
                if sline.rstrip() == '.h':
                    for idx, item in enumerate(self.history):
                        print_cmt(f"{idx + 1}: {item}")
                    continue

                # Switch to python mode
                elif sline.startswith("@p"):
                    self.mode = "python"
                    print_cmt("Switched to Python mode.")
                    self.maybe_submode(line.rstrip())
                    add_to_history_if_unique("@swip")
                    add_to_history_if_unique("@metta")
                    continue

                elif sline.startswith("@space"):
                    cmd_, named = split_or_none(sline, " ")
                    if named is None:
                        print_cmt("@spaces: " + " ".join(space_refs))
                        shownAlready = {}
                        for n in space_refs:
                            v = space_refs[n]
                            if v:
                                s=v()
                                if s:
                                    print_cmt(f"==============================================================")
                                    was = shownAlready.get(id(s))
                                    if was:
                                        print_cmt(f"ALREADY {n} SHOWN as {was}")
                                        continue
                                    n= f"Name: {n}"
                                    shownAlready[id(s)]=n
                                    print_cmt(n)
                                    print_cmt(s)

                        print_cmt(f"==============================================================")

                    else:
                        found = getSpaceByName(named)
                        if found is not None:
                            selected_space_name = named
                            the_new_runner_space = found
                            print_cmt(f"switching to {named}")
                        else:
                            print_cmt(f"Space not found: '{named}'")
                            print_cmt("try:" + " ".join(space_refs))
                    continue

                # Switch to MeTTaLog mode
                elif sline.startswith("@sm") or sline.startswith("@mettal") or sline.startswith("@ml"):
                    self.mode = "mettalog"
                    print_cmt("Switched to MeTTaLog mode.")
                    continue

                # Switch to swip mode
                elif sline.startswith("@s"):
                    self.mode = "swip"
                    print_cmt("Switched to Swip mode.")
                    self.maybe_submode(line.rstrip())
                    add_to_history_if_unique("break")
                    add_to_history_if_unique("listing(maybe_corisponds/2)")
                    add_to_history_if_unique("synth_query(4,Query)")
                    continue

                # Switch to metta mode
                elif sline.startswith("@m"):
                    self.mode = "metta"
                    print_cmt("Switched to MeTTa mode.")
                    self.maybe_submode(line.rstrip())
                    add_to_history_if_unique("!(match &self $ $)")
                    continue

                elif sline.startswith("@v"):
                    verbose = int(sline.split()[1])
                    os.environ["VSPACE_VERBOSE"] = str(verbose)
                    print_cmt(f"Verbosity level set to {verbose}")
                    continue

                elif sline.startswith("@a"): # @arg
                    argmode = self.mode
                    arg = sline.split()[1]
                    handle_arg(arg)
                    self.mode = argmode
                    continue

                elif sline.startswith("@l"): # @load
                    argmode = self.mode
                    arg = sline.split()[1]
                    handle_arg(arg)
                    self.mode = argmode
                    continue

                # Show help
                elif sline.startswith("@h"):
                    print_cmt("Help:")
                    print_cmt("@m       - Switch to MeTTa mode")
                    print_cmt("@m +     -   changes to: Add bare atoms (default)")
                    print_cmt("@m !     -               Interpret bare atoms")
                    print_cmt("@m -     -               Remove bare atoms")
                    #print_cmt("@m ?     -               Query bare atoms")
                    print_cmt("@m ^     - Interpret atoms as if there are in files (+)")
                    print_cmt("@p       - Switch to Python mode")
                    print_cmt("@s       - Switch to Swip mode")
                    print_cmt("@sm,ml   - Switch to MeTTaLog mode")
                    print_cmt("@space   - Change the &self of the_runner_space")
                    print_cmt("@v ###   - Verbosity 0-3")
                    print_cmt("@h       - Display this help message")
                    print_cmt("@arg     - Act as if this arg was passed to the command")
                    print_cmt("           example: '@arg 1-VSpaceTest.metta'  loads and runs this metta file")
                    print_cmt("Ctrl-D   - Exit interpreter")
                    print_cmt(".s       - Save session")
                    print_cmt(".l       - Load the latest session")
                    print_cmt(".q       - Quit the session")
                    print_cmt(".h       - Display command history")
                    print_cmt("\nFrom your shell you can use..")
                    print_cmt("\texport VSPACE_VERBOSE=2")
                    flush_console()
                    continue

                prefix = sline[0]

                if self.mode == "swip":
                    if prefix == "%":
                        print_cmt(line) # comment
                        continue
                    else:
                        swip_exec(line)
                        continue

                elif self.mode == "mettalog":
                   if prefix == ";":
                        print_cmt(line) # comment
                        continue
                   else:

                        if "+-?!^".find(prefix)<0:
                           prefix = self.submode
                           line = sline
                        else:
                           line = line[2:].strip()

                        if prefix=='!':
                            expr = self.parse_single(line)
                            expr = E(S("!"),expr)
                        else:
                            expr = self.parse_single(line)

                        if verbose>1: print_cmt(f"% S-Expr {line}")
                        if verbose>1: print_cmt(f"% M-Expr {expr}")
                        circles = Circles()
                        swipl_fid = PL_open_foreign_frame()
                        t0 = monotonic_ns()
                        try:
                            swip_obj = m2s(circles,expr);
                            if verbose>1: print_cmt(f"% P-Expr {swip_obj}")
                            call_sexpr = Functor("call_sexpr", 5)
                            user = newModule("user")
                            X = Variable()
                            try:
                                print("mettalog...")
                                q = PySwipQ(call_sexpr(prefix,selected_space_name, str(line), swip_obj, X))
                                while q.nextSolution():
                                  print("mettalog...sol")
                                  flush_console()
                                  yield expr, s2m1(circles, X.value)
                            finally:
                                q.closeQuery()
                                flush_console()
                                continue
                        finally:
                            if verbose>0: timeFrom("MeTTaLog",t0)
                            PL_discard_foreign_frame(swipl_fid)
                   continue

                elif self.mode == "python":
                    if prefix == "#":
                        print_cmt(line) # comment
                        continue
                    try:
                     t0 = monotonic_ns()
                     result = eval(line)
                     println(result)
                     continue
                    finally:
                     if verbose>0: timeFrom("python",t0)
                elif self.mode == "metta":
                   try:
                    t0 = monotonic_ns()
                    rest = line[2:].strip()
                    if prefix == ";":
                        print_cmt(line) # comment
                        continue
                    elif sline.startswith(".s"):
                        name = f"session_{round(time())}.mettar" if rest == "" else (
                            rest if rest.endswith("mettar") else rest + ".mettar")
                        with open(os.sep.join(self.cwd + name), 'w') as f:
                            f.writelines(history)
                        continue
                    elif sline.startswith(".l"):
                        name = max(glob("session_*.mettar")) if rest == "" else (
                            rest if rest.endswith("mettar") else rest + ".mettar")
                        self.lazy_import_file(name)
                        continue
                    elif sline.startswith(".q"):
                        break

                    if "+-?!^".find(prefix)<0:
                        prefix = self.submode
                        rest = line

                    #print_cmt(f"submode={self.submode} rest={rest} ")

                    if prefix == "!":
                        expr = self.parse_single(rest)
                        yield expr, interpret(the_running_metta_space(), expr)
                        continue
                    elif prefix == "?":
                        expr = self.parse_single(rest)
                        yield expr, the_running_metta_space().subst(expr, expr)
                        continue
                    elif prefix == "+":
                        expr = self.parse_single(rest)
                        println(the_running_metta_space().add_atom(expr))
                        continue
                    elif prefix == "-":
                        expr = self.parse_single(rest)
                        println(the_running_metta_space().remove_atom(expr))
                        continue
                    elif prefix == "^":
                        println(the_python_runner.run(line));
                        continue
                    else:
                        expr = self.parse_single(rest)
                        yield expr, interpret(the_running_metta_space(), expr)
                        continue
                   finally:
                    if verbose>0: timeFrom("MeTTa",t0)

            except EOFError:
                    sys.stderr = sys.__stderr__
                    if verbose>0: print_cmt("\nCtrl^D EOF...")
                    flush_console()
                    return [True] #sys.exit(0)

            except KeyboardInterrupt as e:
                if verbose>0: print_cmt(f"\nCtrl+C: {e}")
                if verbose>0:
                    buf = io.StringIO()
                    sys.stderr = buf
                    traceback.print_exc()
                    sys.stderr = sys.__stderr__
                    print_cmt(buf.getvalue().replace('rolog','ySwip'))
                #sys.exit(3)
                continue

            except Exception as e:
                if verbose>0: print_cmt(f"Error: {e}")
                if verbose>0:
                    buf = io.StringIO()
                    sys.stderr = buf
                    traceback.print_exc()
                    sys.stderr = sys.__stderr__
                    print_cmt(buf.getvalue().replace('rolog','ySwip'))
                continue

    def repl(self, get_sexpr_input=get_sexpr_input, print_cmt=print_cmt, mode=None):
        load_vspace()
        for i, (expr, result_set) in enumerate(self.repl_loop(get_sexpr_input=get_sexpr_input, print_cmt=print_cmt, mode=mode)):
            if result_set:
                try:
                    for result in result_set:
                        print_cmt(color_expr(result))
                        flush_console()
                except TypeError:
                    print_cmt(color_expr(result_set))
                    flush_console()
            else:
                print_cmt(f"[/]")
                flush_console()

    def copy(self):
        return self

def timeFrom(w, t0):
    elapsed_ns = monotonic_ns() - t0
    elapsed_s = elapsed_ns / 1e9
    elapsed_ms = elapsed_ns / 1e6
    elapsed_us = elapsed_ns / 1e3

    if elapsed_s >= 1:
        print_cmt(f"{w} took {elapsed_s:.5f} seconds")
    elif elapsed_ms >= 1:
        print_cmt(f"{w} took {elapsed_ms:.5f} milliseconds")
    else:
        print_cmt(f"{w} took {elapsed_us:.5f} microseconds")



def call_mettalog(line, parseWithRust = False):

    if parseWithRust:
        expr = self.parse_single(sline)
        if verbose>1: print_cmt(f"% S-Expr {line}")
        if verbose>1: print_cmt(f"% M-Expr {expr}")
        circles = Circles()
        swip_obj = m2s(circles,expr);
        if verbose>1: print_cmt(f"% P-Expr {swip_obj}")
    else:
        swip_obj = line

    flush_console()
    call_sexpr = Functor("call_sexpr", 5)
    user = newModule("user")
    X = Variable()
    q = PySwipQ(call_sexpr(argmode,selected_space_name, str(line), swip_obj, X))
    while q.nextSolution():
      flush_console()
      yield X.value
    q.closeQuery()
    flush_console()

def redirect_stdout(inner_function):
    old_stdout = sys.stdout # Save the current stdout stream
    new_stdout = io.StringIO() # Create a new StringIO buffer
    sys.stdout = new_stdout # Redirect stdout to the new buffer
    try:
        inner_function() # Execute the inner function
    finally:
        sys.stdout = old_stdout # Restore the original stdout stream
    output = new_stdout.getvalue() # Retrieve the output from the new buffer
    new_stdout.close() # Close the new buffer
    return output

@staticmethod
def vspace_init():
    if getattr(vspace_init,"is_init_ran", False) == True:
        return
    vspace_init.is_init_ran = True

    t0 = monotonic_ns()
    #os.system('clear')
    print_cmt(underline(f"Version-Space Init: {__file__}\n"))
    #import site
    #print_cmt ("Site Packages: ",site.getsitepackages())
    #test_nondeterministic_foreign()
    #if os.path.isfile(f"{the_python_runner.cwd}autoexec.metta"):
    #    the_python_runner.lazy_import_file("autoexec.metta")
    # @TODO fix this metta_to_swip_tests1()
    #load_vspace()
    add_janus_methods(sys.modules[__name__], dict = oper_dict)
    f = reg_pyswip_foreign
    redirect_stdout(f)
    #f()
    if verbose>0: timeFrom("init", t0)
    flush_console()

def flush_console():
    try:
      if sys.__stdout__ is not None: sys.__stdout__.flush()
    except Exception: ""
    try:
      if sys.__stderr__ is not None: sys.__stderr__.flush()
    except Exception: ""
    try:
      if sys.stderr is not None and not (sys.stderr is sys.__stderr__): sys.sys.stderr.flush()
    except Exception: ""
    try:
      if sys.stdout is not None and not (sys.stdout is sys.__stdout__): sys.sys.stdout.flush()
    except Exception: ""


# Exporting to another CSV (for demonstration)
#df.to_csv("exported.csv", index=False)
#print_cmt("\n### Data Exported to 'exported.csv' ###")


import os
import pandas as pd
import re
import sys
import chardet

def detect_encoding(file_path, sample_size=20000):
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
    return chardet.detect(raw)['encoding']

from collections.abc import Iterable

def is_lisp_dashed(s):
    pattern = re.compile('^[A-Za-z0-9-_:]+$')
    return bool(pattern.match(s))

def item_string(lst, functor=""):
    if isinstance(lst, str):
        if len(lst) == 0:
            return '""'
        if any(char in lst for char in [' ', '"', "'", "(", ")", ".", "\\"]):
            return json.dumps(lst)
        if isinstance(lst, (int, float)):
            return repr(lst)
        if is_float_string(lst):
            return repr(float(lst))
        if lst.isdigit():
            return repr(int(lst))
        if lst.isalnum():
            if lst[0].isdigit(): return json.dumps(lst)
            return lst
        if lst=="#":
            return lst
        if is_lisp_dashed(lst):
            return lst
        return json.dumps(lst)

    try:
        if isinstance(lst, Iterable):
            return '(' + functor + ' '.join([item_string(vv) for vv in lst]) + ')'
        else:
            return str(lst)
    except TypeError:
        return str(lst)

def list_string(lst, functor="# "):
    try:
        if isinstance(lst, Iterable) and not isinstance(lst, str):
            if len(lst) == 0:
                return '()'
            return '(' + functor + ' '.join([item_string(vv) for vv in lst]) + ')'
        else:
            return item_string(lst)
    except TypeError:
        return item_string(lst)

def is_float_string(s):
    return bool(re.fullmatch(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', s))

import pandas as pd

def update_dataframe_skipping_first_row(df):
    """
    Takes a DataFrame, skips the first row, recalculates unique counts and value counts,
    and infers the most appropriate datatypes for each column.

    Parameters:
    df (pandas.DataFrame): The original DataFrame.

    Returns:
    pandas.DataFrame: A DataFrame with the first row removed, updated with recalculated
    uniqueness and value counts, and with inferred datatypes.
    """
    # Check if the DataFrame is empty or has only one row
    if df.empty or df.shape[0] == 1:
        raise ValueError("DataFrame is empty or has only one row, which cannot be skipped.")

    # Skip the first row and reset the index
    updated_df = df.iloc[1:].reset_index(drop=True)

    # Attempt to infer better dtypes for object columns
    updated_df = updated_df.infer_objects()
    updated_df = updated_df.convert_dtypes()

    # Update DataFrame with uniqueness and value counts for each column
    for col in updated_df.columns:
        updated_df[f'{col}_unique_count'] = updated_df[col].nunique()
        updated_df[f'{col}_value_counts'] = updated_df[col].value_counts().to_dict().__str__()

    return updated_df



def analyze_csv_basename(file_path, sep=None):
    base_name = os.path.basename(file_path)
    base_name = base_name.replace('.tsv', '')
    base_name = base_name.replace('.fb', '')
    # Remove a sequence like _fb_####_## (where # represents a digit)
    base_name = re.sub(r'_fb_\d{4}_\d{2}', '', base_name)
    # Remove a sequence like ####_## at any place
    base_name = re.sub(r'\d{4}_\d{2}', '', base_name)
    # Replace periods with underscores, if not part of a file extension
    base_name = re.sub(r'\.(?=.*\.)', '_', base_name)
    analyze_csv(base_name, file_path, sep=sep)

needed_Skip = 0

def analyze_csv(base_name, file_path, sep=None):
    print_cmt(";;------------------------------------------------------------------------------------------------------------------")
    print_cmt(f"Analyzing file: {file_path}")
    missing_values_list = ["","-"," ","|",",","#",  "*",  "\"\"",  "+", "NULL", "N/A", "--",
                         "NaN","EMPTY","None","n/a","(none)",
                         # "0","Dmel","-1",
                          "MISSING", "?", "undefined", "unknown", "none", "[]", "."]

    def read_csv(enc, skip_rows, header_option):
            false_values_list = ["F", "f", "False", "false", "N", "No", "no", "FALSE"]
            true_values_list = ["T", "t", "True", "true", "Y", "Yes", "yes", "TRUE"]

            engine = 'python' if sep is None else 'c'
            return pd.read_csv(
                file_path,
                sep=sep,
                encoding=enc,
                comment='#',
                compression='infer',
                true_values=true_values_list,
                false_values=false_values_list,
                engine=engine,
                header=header_option,
                skiprows=skip_rows,
                #names=header_names,
                keep_default_na=False,
                skip_blank_lines=True,
                on_bad_lines='skip'
            )

    def read_csv_both_encodings(skip_rows=None, header_option=None):
        try:
            return read_csv('utf-8', skip_rows, header_option)
        except UnicodeDecodeError:
            encoding = detect_encoding(file_path)
            if encoding=='uft-8':
                print_cmt(";; Trying '{encoding}' encoding...")
                try:
                    return read_csv(encoding, skip_rows)
                except Exception as e:
                    print_cmt(f";; Error reading the file with 'utf-8' encoding: {e}")
                    return None
        except Exception as e:
            print_cmt(f";; Error reading the file: {e}")
            return None

    df = read_csv_both_encodings()

    # Function to check if a string contains any digits
    def contains_digit(s):
        return any(char.isdigit() for char in s)

    # Read the first few rows to check for digits
    header_candidates = df.head(3)
    first_row_has_no_digits = all(not contains_digit(str(value)) for value in header_candidates.iloc[0])
    second_third_row_has_digits = any(contains_digit(str(value)) for value in header_candidates.iloc[1]) and any(contains_digit(str(value)) for value in header_candidates.iloc[2])
    global needed_Skip
    # If the first row has no digits but the second and third do, treat the first row as a header
    if first_row_has_no_digits and second_third_row_has_digits:
        print_cmt("First row is set as header based on the digit check.")
        df = read_csv_both_encodings(skip_rows=1, header_option=None)
        needed_Skip = 1
        old_columns = header_candidates.iloc[0]
    else:
        print_cmt("Digit check is inconclusive for determining a header row. No header set.")
        old_columns = df.columns
        # If the columns should be anonymized or kept as is, handle that here


    need_anon_columns = False
    for col in old_columns:
        if not re.match("^[a-zA-Z]+$", str(col)):
            need_anon_columns = True
            break

    new_columns = [f'{i+1}' for i in range(df.shape[1])] if need_anon_columns else old_columns.tolist()
    if need_anon_columns:
        df.columns = new_columns

    col_names = ' '.join([f"{col}" for col in new_columns])

    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

    def metta_read(str):
        print(str)
        res = the_python_runner.run(json.dumps(str))
        if len(res) != 0: print_cmt(";;;="+ repr(res))

    metta_read(f"!(file-name {base_name}  {file_path})")
    metta_read(f"(num-columns {base_name} {df.shape[1]})")
    metta_read(f"(column-names {base_name} {list_string(old_columns)})")
    metta_read(f"(column-names-maybe {base_name} {list_string(df.columns)})")
    metta_read(f"(duplicated-rows {base_name} {df.duplicated().sum()})")
    metta_read(f"(total-rows {base_name} {len(df)})")
    for col in new_columns:
        metta_read(f"(unique-values {base_name} {col} {df[col].nunique()} {df[col].dtype})")

    # Print the unique NA values and their frequencies for each column
    for col in df.columns:
        missing_count = 0
        unique_na_values = []
        frequency_of_unique_na = []
        for na_val in missing_values_list:
            count = df[df[col] == na_val].shape[0]
            if count > 0:
                missing_count += count
                metta_read(f"(null-value-count {base_name} {col} \"{na_val}\" {count})")
                unique_na_values.append(na_val)
                frequency_of_unique_na.append(count)

        metta_read(f"(missing-values {base_name} {col} {missing_count} {list_string(unique_na_values)} {list_string(frequency_of_unique_na)})")

    hl =7
    for col in df.columns:
        if len(df) != df[col].nunique():

            isfrequents = [["#", val, cnt] for val, cnt in df[col].value_counts(ascending=False).head(hl).items() if val not in missing_values_list or len(val)>3]
            isfrequents.reverse()
            metta_read(f"(most-frequent {base_name} {col} {list_string(isfrequents)})\n")
            infrequents = [["#", val, cnt] for val, cnt in df[col].value_counts(ascending=True).head(hl).items() if val not in missing_values_list or len(val)>3]
            #infrequents.reverse()
            #infrequents.reverse()  # Since we can't use slicing on a generator, we reverse it here
            metta_read(f"(less-frequent {base_name} {col} {list_string(infrequents)})\n")

    #metta_read(f"(data-types {base_name} {col} {col.dtype} )")



def import_metta_file(string):
    global argmode
    if argmode=="mettalog":
        load_vspace()
        swip_exec(f"load_metta_file('{selected_space_name}','{string}')")
    else: the_python_runner.import_file(string)



import os
import sys

@export_flags(MeTTa=True)
def vspace_main(*args):
    is_init=False
    #os.system('clear')
    t0 = monotonic_ns()
    if verbose>0: print_cmt(underline("Version-Space Main\n"))
    flush_console()
    #if is_init==False: load_vspace()
    #if is_init==False: load_flybase()
    #if is_init==False:

    if isinstance(args, str):
        handle_arg(args)
    elif isinstance(args, list):
        for arg in args:
            if isinstance(arg, str):
                if len(arg) > 1: handle_arg(arg)

    flush_console()
    global argmode
    the_python_runner.repl(mode=argmode)
    flush_console()
    if verbose>1: timeFrom("main", t0)
    flush_console()

def vspace_main_from_python(sysargv1toN):
    vspace_main(sysargv1toN)

def handle_arg(string, skip_filetypes=['.metta', '.md','.pl', '.png', '.jpg', '.obo']):

        lower = string.lower()

        if lower in ["--metta","--mettalog","--python"]:
            global argmode
            argmode = lower.lstrip('-')
            if verbose>0: print("; argmode=", argmode)
            return

        if os.path.isfile(string):
            if lower.endswith('.metta'):
                if verbose>0: print("; import_metta_file=", string)
                import_metta_file(string)
                return

        global needed_Skip
        if string=="--analyze": sys.exit(needed_Skip)

        if os.path.isdir(string):
            # If it's a directory, traverse it
            for root, _, files in os.walk(string):
                for file in files:
                    try:
                        if any(file.endswith(ext) for ext in skip_filetypes):
                            if verbose>0: print_cmt(f"Skipping file: {file}")
                            continue
                        handle_arg([os.path.join(root, file)], skip_filetypes)
                    except Exception as e:
                        print_cmt(f"An error occurred while processing {string}: {e}")
            return

        elif os.path.isfile(string):
            if lower.endswith('.csv'):
                analyze_csv_basename(string, sep=',')
                return
            elif lower.endswith('.tsv'):
                analyze_csv_basename(string, sep='\t')
                return
            else:
                # Read only the first few lines
                try:
                    analyze_csv_basename(string)
                except UnicodeDecodeError:
                    print_cmt(f"Passing in file: {string}")
                    with open(string, 'r') as file:
                        for i, line in enumerate(file):
                            if i >= 10:
                                break
                            print_cmt(line.strip())
                return

        print_cmt(f"Skipping: {string}")



# All execution happens here
swip = globals().get('swip') or PySwip()
the_verspace = globals().get('the_verspace') or VSpace("&verspace")
the_flybase = globals().get('the_flybase') or VSpace("&flybase")
the_nb_space = globals().get('the_nb_space') or VSpace("&nb")
the_gptspace = globals().get('the_gptspace') or GptSpace()
the_python_runner = globals().get('the_python_runner') or None
selected_space_name = globals().get('selected_space_name') or "&self"
argmode = None
sys_argv_length = len(sys.argv)

if the_python_runner is None:  #MakeInteractiveMeTTa() #def MakeInteractiveMeTTa(): #global the_python_runner,the_old_runner_space,the_new_runner_space,sys_argv_length
    the_python_runner = InteractiveMeTTa()
    #the_python_runner = MeTTa()
    the_python_runner.cwd = [os.path.dirname(os.path.dirname(__file__))]
    the_old_runner_space = the_python_runner.space()
    the_new_runner_space = the_python_runner.space()
    print_cmt("The sys.argv list is:", sys.argv)
    vspace_init()
    the_python_runner.run("!(extend-py! metta_space/metta_learner)")


is_init=False

if __name__ == "__main__":
    vspace_main_from_python(sys.argv[1:])

