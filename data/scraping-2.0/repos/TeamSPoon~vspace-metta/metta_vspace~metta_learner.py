# Version Space Candidate Elimination inside of MeTTa
# This implementation focuses on bringing this machine learning algorithm into the MeTTa relational programming environment.
# Douglas R. Miles 2023
from time import monotonic_ns, time
import atexit, os, subprocess, sys, pip, io, re, traceback, inspect
try:
 import readline
except ImportError:
  import pyreadline3 as readline
from collections import Counter
from glob import glob
import hyperonpy as hp
from hyperon.atoms import V, S, E, ValueAtom, GroundedAtom, ExpressionAtom, G, AtomType, MatchableObject, OperationAtom, OperationObject, BindingsSet
from hyperon.runner import MeTTa
from hyperon.ext import register_atoms, register_tokens
from hyperon.base import AbstractSpace, SpaceRef, GroundingSpace, interpret
# Avoid conflict for "Atom"
from pyswip import Atom as PySwipAtom
from pyswip import Term
from hyperon.atoms import Atom as MeTTaAtom
from pyswip import (call, Functor, PL_discard_foreign_frame, PL_new_term_ref, PL_open_foreign_frame, registerForeign, PL_PRUNED, PL_retry, PL_FA_NONDETERMINISTIC, PL_foreign_control, PL_foreign_context, PL_FIRST_CALL, PL_REDO, Variable, Prolog as PySwip)
from pyswip.easy import newModule, Query
from hyperon.atoms import *
from janus import *
import openai
import hyperon
try:
 openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
 ""

VSPACE_VERBOSE = os.environ.get("VSPACE_VERBOSE")
# 0 = for scripts/demos
# 1 = developer
# 2 = debugger
verbose = 1
if VSPACE_VERBOSE is not None:
    try:
        # Convert it to an integer
        verbose = int(VSPACE_VERBOSE)
    except ValueError: ""
histfile = os.path.join(os.path.expanduser("~"), ".metta_history")
is_init = True
oper_dict = {}
janus_dict = {}
syms_dict = {}
space_refs = {
    #'&vspace': lambda: the_vspace,
    '&gptspace': lambda: the_gptspace,
    #'&flybase': lambda: the_flybase,
    '&parent': lambda: parent_space(),
    '&child': lambda: child_space(),
    '&self': lambda: self_space()}

def  parent_space():
    return the_python_runner.parent.space()

def  child_space():
    return the_python_runner.space()

def  self_space():
    return the_new_runner_space

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

def save(prev_h_len, histfile):
    new_h_len = readline.get_current_history_length()
    readline.set_history_length(300)
    readline.append_history_file(new_h_len - prev_h_len, histfile)
atexit.register(save, h_len, histfile)

def export_to_metta(func):
    setattr(func, 'metta', True)
    if verbose>3: print(f"{func}={getattr(func, 'export_to_metta', False)}")
    return func

def export_flags(**kwargs):
    def decorator(func):
        if verbose > 1: print(f"export_flags({repr(func)})", end=" ")
        for n in kwargs:
            setattr(func, n, kwargs[n])
        if verbose > 1:
            for n in kwargs:
                print(f"{repr(n)}={repr(getattr(func, n, None))}", end=" ")
            print()
        return func
    return decorator

@export_flags(MeTTa=True)
def add_exported_methods(module, dict = oper_dict):
    for name, func in inspect.getmembers(module):
        if inspect.isfunction(func):
            if getattr(func, 'MeTTa', False):
                suggestedName = getattr(func, 'name', None)
                if suggestedName is not None:
                    use_name = suggestedName
                else: use_name = name
                sig = inspect.signature(func)
                params = sig.parameters
                num_args = len([p for p in params.values() if p.default == p.empty and p.kind == p.POSITIONAL_OR_KEYWORD])
                add_pyop(use_name, num_args, getattr(func, 'op', "OperationAtom"), getattr(func, 'unwrap', True), dict)

def add_janus_methods(module, dict = janus_dict):
    for name, func in inspect.getmembers(module):
        if inspect.isfunction(func):
            if getattr(func, 'Janus', False):
                suggestedName = getattr(func, 'name', None)
                if suggestedName is not None:
                    use_name = suggestedName
                else: use_name = name
                for key, item in dict.items():
                    if key==use_name:
                        return
                dict[use_name]=func

                suggestedFlags = getattr(func, 'flags', None)
                if suggestedFlags is None:
                    suggestedFlags = 0

                suggestedArity = getattr(func, 'arity', None)
                if suggestedArity is None:
                    sig = inspect.signature(func)
                    params = sig.parameters
                    num_args = len([p for p in params.values() if p.default == p.empty and p.kind == p.POSITIONAL_OR_KEYWORD])
                    suggestedArity = num_args
                    func.arity = suggestedArity

                registerForeign(func, arity = suggestedArity, flags = suggestedFlags )

@export_flags(MeTTa=True)
def add_pyop(name, length, op_kind, unwrap, dict = oper_dict):
    hyphens, underscores = name.replace('_', '-'), name.replace('-', '_')
    mettavars, pyvars = (' '.join(f"${chr(97 + i)}" for i in range(length))).strip(), (', '.join(chr(97 + i) for i in range(length))).strip()
    s = f"!({hyphens})" if mettavars == "" else f"!({hyphens} {mettavars})"
    add_to_history_if_unique(s); #print(s)
    if hyphens not in dict:
        src, local_vars = f'op = {op_kind}( "{hyphens}", lambda {pyvars}: [{underscores}({pyvars})], unwrap={unwrap})', {}
        if verbose>1: print(f"add_pyop={src}")
        exec(src, globals(), local_vars)
        dict[hyphens] = local_vars['op']
        dict[underscores] = local_vars['op']

@export_flags(MeTTa=True)
def add_from_swip(name, dict = oper_dict):
    hyphens, underscores = name.replace('_', '-'), name.replace('-', '_')
    add_to_history_if_unique(f"!({hyphens})")
    if hyphens not in dict:
        src, local_vars = f'op = lambda : [swip_exec("{underscores}")]', {}
        exec(src, {}, local_vars)
        if verbose>1: print(f"swip: {hyphens}")
        dict[hyphens] = OperationAtom(hyphens, local_vars['op'], unwrap=False)

def addSpaceName(name, space):
    global syms_dict, space_refs
    name = str(name)
    syms_dict[name] = lambda _: G(VSpaceRef(space))
    space_refs[name] = lambda : space

def getSpaceByName(name):
    if name is ValueAtom:
        name = name.get_value()
    if name is GroundingSpace:
        return name
    global space_refs
    found = space_refs.get(str(name), None)
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
            result.append(MeTTaAtom._from_catom(r))
        return result


    def __del__(self):
        """Free the underlying CSpace object """
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
        return VSpaceRef(cspace)

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
        return [MeTTaAtom._from_catom(catom) for catom in
                hp.space_subst(cspace, pattern.catom,
                                         templ.catom)]


def foreign_framed(func):
    def wrapper(*args, **kwargs):
        swipl_fid = PL_open_foreign_frame()
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            if verbose>0: print(f"Error: {e}")
            if verbose>0: traceback.print_exc()
        finally:
            PL_discard_foreign_frame(swipl_fid)
        return result
    return wrapper


@export_flags(MeTTa=True)
class VSpace(AbstractSpace):

    def from_space(self, cspace):
        self.gspace = GroundingSpaceRef(cspace)

    def __init__(self, space_name=None, unwrap=True):
        super().__init__()
        global vspace_ordinal
        ispace_name = f"&vspace_{vspace_ordinal}"
        vspace_ordinal=vspace_ordinal+1
        addSpaceName(ispace_name,self)
        if space_name is None:
            space_name = ispace_name
        self.sp_name = PySwipAtom(space_name)
        self.sp_module = newModule("user")
        self.unwrap = unwrap
        addSpaceName(space_name,self)

    def __del__(self):
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
        if verbose>1: print(f"circles={circles}")
        #if verbose>1: print(f"metta_vars={metta_vars}, swivars={swivars}")
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
            if verbose>0: print(f"Error: {e}")
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
        if verbose>1: print(result)
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
        if verbose>1: print(f"get_atoms={type(C)}")
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
    def __init__(self, space_name=None, unwrap=True):
        super().__init__()



class PySwipQ(Query):

    def __init__(self, *terms, **kwargs):
        if verbose > 1:
            for obj in terms:
                println(obj)
        super().__init__(*terms, **kwargs)

access_error = True

@export_flags(MeTTa=True)
class FederatedSpace(VSpace):

    def __init__(self, space_name, unwrap=True):
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
    test_custom_space(lambda: VSpace())

@export_flags(MeTTa=True)
def test_custom_v_space2():
    test_custom_space(lambda: the_other_space)

@export_flags(MeTTa=True)
def test_custom_v_space3():
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

    kb = VSpaceRef(test_space)


    kb.add_atom(S("a"))
    kb.add_atom(S("b"))
    #kb.add_atom(E(S("a"),S("b")))

    self_assertEqual(kb.atom_count(), 2)
    self_assertEqual(kb.get_payload().test_attrib, "Test Space Payload Attrib")
    self_assertEqualNoOrder(kb.get_atoms(), [S("a"), S("b")])

    kb = VSpaceRef(LambdaSpaceFn())
    kb.add_atom(S("a"))
    kb.add_atom(S("b"))
    kb.add_atom(S("c"))

    self_assertTrue(kb.remove_atom(S("b")),"remove_atom on a present atom should return true")
    self_assertFalse(kb.remove_atom(S("bogus")),"remove_atom on a missing atom should return false")
    self_assertEqualNoOrder(kb.get_atoms(), [S("a"), S("c")])

    kb = VSpaceRef(LambdaSpaceFn())
    kb.add_atom(S("a"))
    kb.add_atom(S("b"))
    kb.add_atom(S("c"))

    self_assertTrue(kb.replace_atom(S("b"), S("d")))
    self_assertEqualNoOrder(kb.get_atoms(), [S("a"), S("d"), S("c")])

    kb = VSpaceRef(LambdaSpaceFn())
    kb.add_atom(E(S("A"), S("B")))
    kb.add_atom(E(S("C"), S("D")))
    # Checking that multiple matches can be returned
    kb.add_atom(E(S("A"), S("E")))

    result = kb.query(E(S("A"), V("XX")))
    self_assertEqualNoOrder(result, [{"XX": S("B")}, {"XX": S("E")}])

    m = MeTTa()

    # Make a little space and add it to the MeTTa interpreter's space
    little_space = VSpaceRef(LambdaSpaceFn())
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
    if verbose>1: print("mspace")
    mspace = m.space()
    gmspace = G(mspace)
    A = E(S("big-space"), gmspace)
    if verbose>1: print("little_space.add_atom")
    little_space.add_atom(A)
    if verbose>1: print("Next Space")
    nested = VSpaceRef(LambdaSpaceFn())
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
    print(f"r({type(r)})={str(r)}/{repr(r)}")
    return r

def s2m1(circles,swip_obj, depth=0):

    if verbose>1:
        for i in range(depth):
            print("   ",end='')
        print(f's2m({len(circles)},{type(swip_obj)}): {str(swip_obj)}/{repr(swip_obj)}')

    # Already converted
    if isinstance(swip_obj, (VariableAtom, GroundedAtom, MeTTaAtom, ExpressionAtom)):
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
           V = V(mname)
           circles[mname] = swip_obj
           circles[id(V)] = swip_obj
           circles[swip_obj] = V
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

    V = None

    if isinstance(metta_obj, VariableAtom):
        oid = mv2svn(metta_obj)
        var = circles.get("$" + oid, None)
        # We are in a circluar reference?
        if var is not None:
            #print(f"{oid}={len(circles)}={type(circles)}={type(metta_obj)}")
            return var

        V = Variable(name = oid)
        circles["$" + oid] = V
        circles[metta_obj] = V
        circles[V] = metta_obj
        return V

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
    else:
        raise ValueError(f"Unknown MeTTa object type: {type(metta_obj)}")

    if depth==0:
        V = Variable()
        V.unify(L)
        circles[oid] = V
        circles[V] = metta_obj
        circles[metta_obj] = V
        return V

    circles[L] = metta_obj
    circles[metta_obj] = L
    return L

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

    if verbose>0: print(f"Unknown MeTTa object type: {type(metta_obj)}={metta_obj}")

    raise ValueError(f"Unknown MeTTa object type: {type(metta_obj)}")

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

    def match_(self, metta_obj):
        if isinstance(metta_obj, GroundedAtom):
            metta_obj = metta_obj.get_object().content
        if not isinstance(metta_obj, VSPatternValue):
            return metta_obj.match_(self)
        # TODO: match to patterns
        return []


class VSPatternOperation(OperationObject):

    def __init__(self, name, op, unwrap=False, rec=False):
        super().__init__(name, op, unwrap)
        self.rec = rec

    def execute(self, *args, res_typ=AtomType.UNDEFINED):
        if self.rec:
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
        if self.rec:
            args = args[0].get_children()
            args = [self.execute(arg)[0] if isinstance(arg, ExpressionAtom) else arg for arg in args]
        # If there is a variable or VSPatternValue in arguments, create VSPatternValue
        # instead of executing the operation
        for arg in args:
            if isinstance(arg, GroundedAtom) and isinstance(arg.get_object(), VSPatternValue):
                return [G(VSPatternValue([self, args]))]
            if isinstance(arg, VariableAtom):
                return [G(VSPatternValue([self, args]))]

        # type-check?
        if self.unwrap:
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
    return unwrap_pyobjs(arg)

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
        # if verbose>0: print(type(obj))
        print(obj)
    return obj


@export_flags(MeTTa=True, name="print")
def println(orig):
    """
    Prints the given object and returns it.

    Args:
        obj: The object to be printed.

    Returns:
        The same object that was passed in.
    """
    obj = unwrap_pyobjs(orig)

    if not isinstance(obj, (Term, Variable)):
	    print(repr(obj))
    else:
        fn = Functor("pp_ilp")
        call(fn(obj))

    flush_console()
    return orig





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
        fn = Functor("pp_ilp")
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
    #if the_python_runner.parent!=the_python_runner:
    #    return the_python_runner.parent.space()
    return the_new_runner_space

# Borrowed impl from Adam Vandervorst
import os
from importlib import import_module
import hyperonpy as hp
from hyperon.atoms import Atom, AtomType, OperationAtom
from hyperon.base import GroundingSpaceRef, Tokenizer, SExprParser

class ExtendedMeTTa:

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


    #def __del__(self):
        #hp.metta_free(self.cmetta)

    def space(self):
        return GroundingSpaceRef._from_cspace(hp.metta_space(self.cmetta))

    def tokenizer(self):
        return Tokenizer._from_ctokenizer(hp.metta_tokenizer(self.cmetta))

    def register_token(self, regexp, constr):
        self.tokenizer().register_token(regexp, constr)

    def register_atom(self, name, symbol):
        self.register_token(name, lambda _: symbol)

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
        if not isinstance(name, str):
            name = repr(name)
        mod = import_module(name)
        for n in dir(mod):
            obj = getattr(mod, n)
            if '__name__' in dir(obj) and obj.__name__ == 'metta_register':
                obj(self)

    def import_file(self, fname):
        path = fname.split(os.sep)
        if len(path) == 1:
            path = ['.'] + path
        f = open(os.sep.join(path), "r")
        program = f.read()
        f.close()
        # changing cwd
        prev_cwd = os.getcwd()
        os.chdir(os.sep.join(path[:-1]))
        result = self.run(program)
        # restoring cwd
        os.chdir(prev_cwd)
        return result

    def run(self, program, flat=False):
        parser = SExprParser(program)
        results = hp.metta_run(self.cmetta, parser.cparser)
        if flat:
            return [MeTTaAtom._from_catom(catom) for result in results for catom in result]
        else:
            return [[MeTTaAtom._from_catom(catom) for catom in result] for result in results]

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
    return parts[0], parts[1] if len(parts) > 1 else None

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
    print(args)
    flush_console()

def pl_insert(*args):
    print(args)
    flush_console()

#pass_metta=True
@register_atoms
#@register_atoms(pass_metta=True)
def register_vspace_atoms():

    metta = None
    global oper_dict

    if not metta is None: the_python_runner.set_cmetta(metta)

    counter = 0
    if verbose>1: print(f"register_vspace_atoms metta={metta} {self_space_info()}")

    if not isinstance(metta, VSpace):
        the_python_runner.parent = metta

    def new_value_atom_func():
        nonlocal counter
        counter += 1
        return [ValueAtom({'A': counter, 6: 'B'})]

    # We don't add types for operations, because numpy operations types are too loose
    nmVectorAtom = G(VSPatternOperation('np.vector', wrapnpop(lambda *args: np.array(args)), unwrap=False))
    nmArrayAtom = G(VSPatternOperation('np.array', wrapnpop(lambda *args: np.array(args)), unwrap=False, rec=True))
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
    add_exported_methods(sys.modules[__name__], dict = oper_dict)
    oper_dict.update({
        r"TupleCount": OperationAtom( 'TupleCount', lambda atom: [ValueAtom(len(atom.get_children()), 'Number')], [AtomType.ATOM, "Number"], unwrap=False),
        r"np\.vector": nmVectorAtom,
        r"np\.array": nmArrayAtom,
        r"np\.add": nmAddAtom,
        r"np\.sub": nmSubAtom,
        r"np\.mul": nmMulAtom,
        r"np\.matmul": nmMMulAtom,
        r"test_s": testS,

        r"np\.div": nmDivAtom,

        r"new-gpt-space": OperationAtom('new-gpt-space', lambda: [G(SpaceRef(GptSpace()))], unwrap=False),
        r"new-gpt-intent-space": OperationAtom('new-gpt-intent-space', lambda: [G(SpaceRef(GptIntentSpace()))], unwrap=False),

        r"new-v-space": OperationAtom('new-v-space', lambda: [G(SpaceRef(VSpace()))], unwrap=False),
        r"the-v-space": OperationAtom('new-v-space', lambda: [G(SpaceRef(the_vspace))], unwrap=False),


        r"new-value-atom": newValueAtom,
        #'&self': runnerAtom,
        #'&swip': ValueAtom(swip),



       'pl_select': G(OperationObject('pl_select', wrapsqlop(pl_select), unwrap=False)),
       'pl_insert': G(OperationObject('pl_insert', wrapsqlop(pl_insert), unwrap=False)),


        '&my-dict': ValueAtom({'A': 5, 6: 'B'}),
        'get-by-key': OperationAtom('get-by-key', lambda d, k: d[k]),

        # Our FFI to PySWIP
        'load-vspace': OperationAtom('load-vspace', lambda: [load_vspace()]),
        'mine-overlaps': OperationAtom('mine-overlaps', lambda: [mine_overlaps()]),
        'try-overlaps': OperationAtom('try-overlaps', lambda: [try_overlaps()]),
        'load-flybase-full': OperationAtom('load-flybase-full', lambda: [load_flybase("inf")]),
        'load-flybase-tiny': OperationAtom('load-flybase-tiny', lambda: [load_flybase(20000)]),

        r"fb.test-nondeterministic-foreign": OperationAtom('test-nondeterministic-foreign', lambda: test_nondeterministic_foreign, unwrap=False),

        'vspace-main': OperationAtom('vspace-main', lambda: [vspace_main()]),
        'metta_learner::vspace-main': OperationAtom('vspace-main', lambda: [vspace_main()]),
        'swip-exec': OperationAtom('swip-exec', lambda s: [swip_exec(s)]),
        'py-eval': OperationAtom('py-eval', lambda s: [eval(s)]) })

    return oper_dict


# For now lets test with only  Atoms
@register_tokens(pass_metta=True)
def register_vspace_tokens(metta):

    if verbose>1: print(f"register_vspace_tokens metta={metta} {self_space_info()}")

    the_python_runner.set_cmetta(metta.cmetta)

    if not isinstance(metta, VSpace):
        the_python_runner.parent = metta

    def run_resolved_symbol_op(the_python_runner, atom, *args):
        expr = E(atom, *args)
        if verbose>1: print(f"run_resolved_symbol_op: atom={atom}, args={args}, expr={expr} metta={metta} {self_space_info()}")
        result1 = hp.metta_evaluate_atom(the_python_runner.cmetta, expr.catom)
        result = [MeTTaAtom._from_catom(catom) for catom in result1]
        if verbose>1: print(f"run_resolved_symbol_op: result1={result1}, result={result}")
        return result

    def resolve_atom(metta, token):
        # TODO: nested modules...

        if token is None: return token

        if verbose>1: print(f"resolve_atom: token={token}/{type(token)} metta={metta}")
        runner_name, atom_name = token.split('::')

        if atom_name in oper_dict:
            if verbose>1: print(f"resolve_atom: token={token} metta={metta}")
            return oper_dict[atom_name]

        atom_name2 = atom_name.replace('_', '-')

        if atom_name2 in oper_dict:
            if verbose>0: print(f"resolve_atom: token={token} metta={metta}")
            return oper_dict[atom_name2]

        if atom_name=="vspace-main":
            vspace_main()
            return
        # FIXME: using `run` for this is an overkill
        ran = metta.run('! ' + runner_name)[0][0];
        if verbose>1: print(f"resolve_atom: token={token} ran={type(ran)} metta={metta} {self_space_info()}")
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
            if verbose>1: print(f"resolve_atom: token={token} metta={metta}")
            return oper_dict[atom_name]

    syms_dict.update({
        '&gptspace': lambda _: G(VSpaceRef(the_gptspace)),
        '&flybase': lambda _: G(VSpaceRef(the_flybase)),
        '&vspace': lambda _: G(VSpaceRef(the_vspace)),
        '&vbase_class': lambda _: G((the_vspace)),
        '&parent_ref': lambda _: G(VSpaceRef(the_python_runner.parent.space())),
        '&parent': lambda _: G(the_python_runner.parent.space()),
        '&child': lambda _: G(the_python_runner.space()),
        '&child_ref': lambda _: G(VSpaceRef(the_python_runner.space())),
        '&the_runner': lambda _: ValueAtom(the_python_runner),
        '&the_metta': lambda _: ValueAtom(the_python_runner.parent),
        r"[^\s]+::[^\s]+": lambda token: resolve_atom(metta, token)
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
def add_from_space(space_name, atom):
    space = getSpaceByName(space_name)
    if space:
        atom = s2m(circles,atom)
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
def replace_from_space(space_name, from_atom, to_atom):
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
@export_flags(Janus=True, arity=2, flags=PL_FA_NONDETERMINISTIC)
def atoms_iter_from_space(space_name, result, context):
    global idKey, context_atom_iters
    control = PL_foreign_control(context)
    context = PL_foreign_context(context)
    id = context

    if control == PL_FIRST_CALL:
        id = idKey
        idKey= idKey+1
        space = getSpaceByName(space_name)
        if space:
            iterator =  getattr(space,"") # Create a new iterator
            context_atom_iters[id] = iterator  # Store it in the dictionary
            try:
                value = next(iterator)
                a.unify((value))
                context = id
                return PL_retry(context)
            except StopIteration:
                del context_atom_iters[id]  # Clean up
        return False

    elif control == PL_REDO:
        iterator = context_atom_iters.get(id)
        if iterator is not None:
            try:
                value = next(iterator)
                a.unify((value))
                return PL_retry(context)
            except StopIteration:
                del context_atom_iters[id]  # Clean up
                return False
        pass

    elif control == PL_PRUNED:
        # Clean up the iterator when we're done
        if id in context_atom_iters:
            del context_atom_iters[id]
        pass


def reg_pyswip_foreign():

    test_nondeterministic_foreign()


    def py_eval(e, result):
        return res_unify(result,eval(str(e)))
    py_eval.arity = 2
    registerForeign(py_eval)

    # Register the foreign functions in PySwip
    #registerForeign(new_rust_space, arity=1)
    #registerForeign(query_from_space, arity=3)
    #registerForeign(add_from_space, arity=2)
    #registerForeign(remove_from_space, arity=2)
    #registerForeign(replace_from_space, arity=3)
    #registerForeign(atom_count_from_space, arity=2)
    #registerForeign(atoms_iter_from_space, arity=2)
    #registerForeign(get_atoms_from_space, arity=2)
    add_to_history_if_unique.arity=1
    registerForeign(add_to_history_if_unique)

    add_janus_methods(sys.modules[__name__], dict = oper_dict)

    #?- query_from_space('example', 'my_atom', Result).
    #?- add_from_space('example', 'new_atom').
    #?- remove_from_space('example', 'some_atom').
    #?- replace_from_space('example', 'old_atom', 'new_atom').
    #?- atom_count_from_space('example', Count).
    #?- atoms_iter_from_space('example', Atoms).


@export_flags(Janus=True)
def find_rust_space(space_name, result):
    space = getSpaceByName(space_name)
    named = getNameBySpace(space)
    if space:
        return res_unify(result,named)
    return False

@export_flags(Janus=True)
def new_rust_space(result):
    space = VSpace()
    named = getNameBySpace(space)
    if space:
        res_unify(result,swipAtom(named))
        return True
    return False


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

    print(result)

    if len(result) != 10:
        print('Query should return 10 results')

    for i in range(10):
        if {'X': i} not in result:
            print('Expected result X:{} not present'.format(i))


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

    print(result)

    if len(result) != 10:
        print('Query should return 10 results')

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

    print(result)

    if len(result) != 4:
        print('nondet3 should return 4 results')


@export_flags(MeTTa=True, Janus=True)
def test_nondeterministic_foreign():

    test_nondeterministic_foreign1()
    test_nondeterministic_foreign2()
    test_nondeterministic_foreign3()


    def hello(t):
        print("Hello,", t)

    hello.arity = 1

    registerForeign(hello, arity=1)


    def hello1(t):
        readline.replace_history_item(0, t)
        print("Hello1,", t)


    hello1.arity = 1

    registerForeign(hello1, arity=1)




    swip.assertz("father(michael,john)")
    swip.assertz("father(michael,gina)")

    result = list(swip.query("father(michael,X), hello(X)"))

    print(result)

    if len(result) != 2:
        print('Query should return two results')
    for name in ('john', 'gina'):
        if {'X': name} not in result:
            print('Expected result X:{} not present'.format(name))


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

    print(result)

    if result[0]['Result'] != 'true':
          print('A string return value should not be converted to an atom.')

    print()
    print()
    print()
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
    print(f"swip_functor={swip_functor}"),
    metta_expr = s2m(circles,swip_functor)
    print(f"metta_expr={metta_expr}"),
    converted_back_to_swip = m2s(circles,metta_expr)
    print(f"converted_back_to_swip={converted_back_to_swip}"),


    # Now you can use the methods in PySwip queries
    print(list(swip.query("swip_to_metta_wrapper('example', X).")))
    print(list(swip.query("metta_to_swip_wrapper(X, 'example').")))

@export_flags(MeTTa=True)
def metta_to_swip_tests2():
    # Register the methods as foreign predicates
    registerForeign(swip_to_metta_wrapper, arity=2)
    registerForeign(metta_to_swip_wrapper, arity=2)

    circles = Circles()
    # Now you can use the methods in PySwip queries
    printl(list(swip.query("swip_to_metta_wrapper('example', X).")))
    printl(list(swip.query("metta_to_swip_wrapper(X, 'example').")))

    # Usage:
    swip_list = ["a", "b", 3]
    metta_expr = s2m(circles,swip_list)
    converted_back_to_swip = m2s(circles,metta_expr)
    swip_functor = Functor(PySwipAtom("example"), 2, [PySwipAtom("sub1"), 3.14])
    metta_expr = s2m(circles,swip_functor)
    converted_back_to_swip = m2s(circles,metta_expr)

@export_flags(MeTTa=True)
def load_vspace():
   swip_exec(f"ensure_loaded('{os.path.dirname(__file__)}/pyswip/swi_flybase')")

@export_flags(MeTTa=True)
def mine_overlaps():
   load_vspace()
   swip_exec("mine_overlaps")
   #readline_add_history("!(try-overlaps)")


@export_flags(MeTTa=True)
def try_overlaps():
   load_vspace()
   swip_exec("try_overlaps")

@export_flags(MeTTa=True)
def learn_vspace():
   load_vspace()
   swip_exec("learn_vspace(60)")

@export_flags(MeTTa=True)
def mettalog():
   load_vspace()
   swip_exec("repl")

@export_flags(MeTTa=True)
def mettalog_pl():
   load_vspace()
   swip_exec("break")

def load_flybase(size):
   load_vspace()
   swip_exec(f"load_flybase({size})")
   #readline_add_history("!(mine-overlaps)")

@export_flags(MeTTa=True)
@foreign_framed
def swip_exec(qry):
    #from metta_vspace import swip
    #if is_init==True:
    #   print("Not running Query: ",qry)
    #   return
    for r in swip.query(qry):
        print(r)

@export_flags(MeTTa=True)
def test_custom_m_space():

    class TestSpace(AbstractSpace):

        def __init__(self, unwrap=True):
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
class InteractiveMeTTa(LazyMeTTa):

    def __init__(self):
        super().__init__()
        # parent == self
        #   means no parent MeTTa yet
        self.parent = self

    def maybe_submode(self, line):
        lastchar = line[-1]
        if "+-?!^".find(lastchar)>=0:
            self.submode=lastchar

    def repl_loop(self):

        global verbose
        self.mode = "metta"
        self.submode = "!"
        #self.history = []
        load_vspace()

        while True:
            try:
                flush_console()
                # Use the input function to get user input
                prmpt = self.mode + " "+ self.submode + "> "

                line = get_sexpr_input(prmpt)
                if line:
                    sline = line.lstrip()
                    add_to_history_if_unique(line, position_from_last=1)
                else:
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
                        print(f"{idx + 1}: {item}")
                    continue

                # Switch to python mode
                elif sline.startswith("@p"):
                    self.mode = "python"
                    print("Switched to Python mode.")
                    self.maybe_submode(line.rstrip())
                    add_to_history_if_unique("@swip")
                    add_to_history_if_unique("@metta")
                    continue

                elif sline.startswith("@space"):
                    global the_new_runner_space
                    global selected_space_name
                    cmd_, named = split_or_none(sline, " ")
                    if named is None:
                        print("; @spaces:", " ".join(space_refs))
                    elif named in space_refs:
                        print(f"; named={named}")
                        selected_space_name = named
                        found = getSpaceByName(named)
                        if found is not None:
                            selected_space_name = named
                            the_new_runner_space = found
                        else: print("Space not found {named}")
                    continue

                # Switch to MeTTaLog mode
                elif sline.startswith("@sm") or sline.startswith("@mettal") or sline.startswith("@ml"):
                    self.mode = "mettalog"
                    print("Switched to MettaLog mode.")
                    continue

                # Switch to swip mode
                elif sline.startswith("@s"):
                    self.mode = "swip"
                    print("Switched to Swip mode.")
                    self.maybe_submode(line.rstrip())
                    add_to_history_if_unique("break")
                    add_to_history_if_unique("listing(maybe_corisponds/2)")
                    add_to_history_if_unique("synth_query(4,Query)")
                    continue

                # Switch to metta mode
                elif sline.startswith("@m"):
                    self.mode = "metta"
                    print("Switched to MeTTa mode.")
                    self.maybe_submode(line.rstrip())
                    add_to_history_if_unique("!(match &self $ $)")
                    continue

                elif sline.startswith("@v"):
                    verbose = int(sline.split()[1])
                    os.environ["VSPACE_VERBOSE"] = str(verbose)
                    print(f"Verbosity level set to {verbose}")
                    continue

                # Show help
                elif sline.startswith("@h"):
                    print("Help:")
                    print("@m       - Switch to MeTTa mode")
                    print("@m +     -   changes to: Add bare atoms (default)")
                    print("@m !     -               Interpret bare atoms")
                    print("@m -     -               Remove bare atoms")
                    #print("@m ?     -               Query bare atoms")
                    print("@m ^     - Interpret atoms as if there are in files (+)")
                    print("@p       - Switch to Python mode")
                    print("@s       - Switch to Swip mode")
                    print("@sm      - Switch to MeTTaLog mode")
                    print("@space   - Change the &self of the_runner_space")
                    print("@v ###   - Verbosity 0-3")
                    print("@h       - Display this help message")
                    print("Ctrl-D   - Exit interpreter")
                    print(".s       - Save session")
                    print(".l       - Load the latest session")
                    print(".q       - Quit the session")
                    print(".h       - Display command history")
                    print("\nFrom your shell you can use..")
                    print("\texport VSPACE_VERBOSE=2")
                    flush_console()
                    continue

                prefix = sline[0]

                if sline.endswith("."):
                    swip_exec(line)
                    continue

                elif self.mode == "swip":
                    if prefix == "%":
                        print(line) # comment
                        continue
                    else:
                        swip_exec(line)
                        continue

                elif self.mode == "mettalog":
                   if prefix == ";":
                        print(line) # comment
                        continue
                   else:
                        if sline.startswith("!"):
                            rest = line[2:].strip()
                            expr = self.parse_single(rest)
                            expr = E(S("!"),expr)
                        else:
                            expr = self.parse_single(sline)

                        if verbose>1: print(f"% S-Expr {line}")
                        if verbose>1: print(f"% M-Expr {expr}")
                        circles = Circles()
                        swipl_fid = PL_open_foreign_frame()
                        try:
                            swip_obj = m2s(circles,expr);
                            if verbose>1: print(f"% P-Expr {swip_obj}")
                            call_sexpr = Functor("call_sexpr", 4)
                            user = newModule("user")
                            X = Variable()
                            try:
                                q = PySwipQ(call_sexpr(selected_space_name, line, swip_obj, X))
                                while q.nextSolution():
                                  flush_console()
                                  yield expr, s2m1(circles, X.value)
                            finally:
                                q.closeQuery()
                                flush_console()
                                continue
                        finally:
                            PL_discard_foreign_frame(swipl_fid)
                   continue

                elif self.mode == "python":
                    if prefix == "#":
                        print(line) # comment
                        continue
                    result = eval(line)
                    printl(result)
                    continue

                elif self.mode == "metta":
                    rest = line[2:].strip()
                    if prefix == ";":
                        print(line) # comment
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

                    #print(f"submode={self.submode} rest={rest} ")

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
                        the_running_metta_space().add_atom(expr)
                        continue
                    elif prefix == "-":
                        expr = self.parse_single(rest)
                        the_running_metta_space().remove_atom(expr)
                        continue
                    elif prefix == "^":
                        printl(the_python_runner.run(line));
                        continue
                    else:
                        expr = self.parse_single(rest)
                        yield expr, interpret(the_running_metta_space(), expr)
                        continue

            except EOFError:
                    sys.stderr = sys.__stderr__
                    if verbose>0: print("\nCtrl^D EOF...")
                    flush_console()
                    return [True] #sys.exit(0)

            except KeyboardInterrupt as e:
                if verbose>0: print(f"\nCtrl+C: {e}")
                if verbose>0:
                    buf = io.StringIO()
                    sys.stderr = buf
                    traceback.print_exc()
                    sys.stderr = sys.__stderr__
                    print(buf.getvalue().replace('rolog','ySwip'))
                #sys.exit(3)
                continue

            except Exception as e:
                if verbose>0: print(f"Error: {e}")
                if verbose>0:
                    buf = io.StringIO()
                    sys.stderr = buf
                    traceback.print_exc()
                    sys.stderr = sys.__stderr__
                    print(buf.getvalue().replace('rolog','ySwip'))
                continue

    def repl(self):
        for i, (expr, result_set) in enumerate(self.repl_loop()):
            if result_set:
                try:
                    for result in result_set:
                        print(color_expr(result))
                        flush_console()
                except TypeError:
                    print(color_expr(result_set))
                    flush_console()
            else:
                print(f"[/]")
                flush_console()

    def copy(self):
        return self

def call_mettalog(line, parseWithRust = False):

    if parseWithRust:
        expr = self.parse_single(sline)
        if verbose>1: print(f"% S-Expr {line}")
        if verbose>1: print(f"% M-Expr {expr}")
        circles = Circles()
        swip_obj = m2s(circles,expr);
        if verbose>1: print(f"% P-Expr {swip_obj}")
    else:
        swip_obj = line

    flush_console()
    call_sexpr = Functor("call_sexpr", 3)
    user = newModule("user")
    X = Variable()
    q = PySwipQ(call_sexpr(selected_space_name, swip_obj, X))
    while q.nextSolution():
      flush_console()
      yield X.value
    q.closeQuery()
    flush_console()

@export_flags(MeTTa=True)
def vspace_main():
    is_init=False
    #os.system('clear')
    t0 = monotonic_ns()
    if verbose>0: print(underline("Version-Space Main\n"))
    flush_console()
    #if is_init==False: load_vspace()
    #if is_init==False: load_flybase()
    #if is_init==False:

    flush_console()
    the_python_runner.repl()
    flush_console()
    if verbose>1: print(f"\nmain took {(monotonic_ns() - t0)/1e9:.5} seconds in walltime")
    flush_console()

def vspace_init():
    t0 = monotonic_ns()
    #os.system('clear')
    print(underline(f"Version-Space Init: {__file__}\n"))
    #import site
    #print ("Site Packages: ",site.getsitepackages())
    #test_nondeterministic_foreign()
    swip.assertz("py_named_space('&self')");
    #if os.path.isfile(f"{the_python_runner.cwd}autoexec.metta"):
    #    the_python_runner.lazy_import_file("autoexec.metta")
    # @TODO fix this metta_to_swip_tests1()
    #load_vspace()
    reg_pyswip_foreign()
    print(f"\nInit took {(monotonic_ns() - t0)/1e9:.5} seconds")
    flush_console()

def flush_console():
    try:
      sys.stdout.flush()
    except Exception: ""
    try:
      sys.stderr.flush()
    except Exception: ""



# All execution happens here
swip = PySwip()
the_gptspace = GptSpace()
the_vspace = VSpace("&vspace")
the_flybase = VSpace("&flybase")
the_nb_space = VSpace("&nb")
the_other_space = VSpace("&self2")
the_python_runner = InteractiveMeTTa();
the_python_runner.cwd = [os.path.dirname(os.path.dirname(__file__))]
the_old_runner_space = the_python_runner.space()
the_python_runner.run("!(extend-py! metta_learner)")
the_new_runner_space = the_python_runner.space()
selected_space_name = "&self"
#the_python_runner.run("!(extend-py! VSpace)")
#the_python_runner.run("!(extend-py! GptSpace)")
is_init_ran = False
if is_init_ran == False:
    is_init_ran = True
    vspace_init()

if __name__ == "__main__":
    vspace_main()

#from . import metta_learner

