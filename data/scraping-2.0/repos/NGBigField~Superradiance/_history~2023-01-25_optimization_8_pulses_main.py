# ==================================================================================== #
        
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np
from numpy import pi

# For typing hints:
from typing import (
    Any,
    Tuple,
    List,
    Union,
    Dict,
    Final,
    Optional,
    Callable,
    Generator,
    NamedTuple,
)

# import our helper modules
from utils import (
    visuals,
    saveload,
    types,
    decorators,
    strings,
    assertions,
    sounds,
    lists,
    errors,
)

# For coherent control
from coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
    Operation,
)
        
# For OOP:
from dataclasses import dataclass, field
from enum import Enum, auto

# Import optimization options and code:
from optimization import (
    LearnedResults,
    add_noise_to_vector,
    learn_custom_operation,
    ParamLock,
    BaseParamType,
    FreeParam,
    FixedParam,
    CostFunctions,
    _initial_guess,
    fix_random_params,
)
import metrics

# For operations:
import coherentcontrol
from fock import Fock, cat_state

# For managing saved data:
from utils.saved_data_manager import NOON_DATA, exist_saved_noon, get_saved_noon, save_noon

from optimization_and_operations import pair_custom_operations_and_opt_params_to_op_params, free_all_params

# For timing and random seeds:
import time


# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final[str] = "Nelder-Mead" #'SLSQP' # 'Nelder-Mead'
NUM_PULSE_PARAMS : Final = 4  

TOLERANCE : Final[float] = 1e-16  # 1e-12
MAX_NUM_ITERATION : Final[int] = int(1e6)  

T4_PARAM_INDEX : Final[int] = 5

# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #


def _rand(n:int, sigma:float=1)->list:
    return list(np.random.randn(n)*sigma)

def _load_or_find_noon(num_moments:int, print_on:bool=True) -> NOON_DATA:
    if exist_saved_noon(num_moments):
        noon_data = get_saved_noon(num_moments)
    else:
        
        ## Define operations:
        coherent_control = CoherentControl(num_moments=num_moments)
        standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=0)

        ## Define initial state and guess:
        initial_state = Fock.excited_state_density_matrix(num_moments)
        # Noon Operations:
        noon_creation_operations = [
            standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
            standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0]),
            standard_operations.stark_shift_and_rot(stark_shift_indices=[] , rotation_indices=[0, 1]),
            standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0, 1]),
        ]

        initial_guess = _initial_guess()
        
        ## Learn how to prepare a noon state:
        # Define cost function
        cost_function = CostFunctions.fidelity_to_noon(initial_state)            
        noon_results = learn_custom_operation(
            num_moments=num_moments, 
            initial_state=initial_state, 
            cost_function=cost_function, 
            operations=noon_creation_operations, 
            max_iter=MAX_NUM_ITERATION, 
            initial_guess=initial_guess
        )
        sounds.ascend()
        # visuals.plot_city(noon_results.final_state)
        # visuals.draw_now()
        fidelity =  -1 * noon_results.score
        if print_on:
            print(f"NOON fidelity is { fidelity }")
        
        # Save results:
        noon_data = NOON_DATA(
            num_moments=num_moments,
            state=noon_results.final_state,
            params=noon_results.theta,
            operation=[str(op) for op in noon_creation_operations],
            fidelity=fidelity
        )
        save_noon(noon_data)
        
    return noon_data

def _common_4_legged_search_inputs(num_moments:int, num_transition_frames:int=0):
    ## Check inputs:
    assertions.even(num_moments)
    
    ## Define operations:
    initial_state = Fock.excited_state_density_matrix(num_moments)
    coherent_control = CoherentControl(num_moments=num_moments)
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    Sp = coherent_control.s_pulses.Sp
    Sx = coherent_control.s_pulses.Sx
    Sy = coherent_control.s_pulses.Sy
    Sz = coherent_control.s_pulses.Sz
    noon_creation_operations : List[Operation] = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[] , rotation_indices=[0, 1]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0, 1]),
    ]
    rotation_operation = [standard_operations.power_pulse_on_specific_directions(power=1)]


    noon_data = _load_or_find_noon(num_moments)

    # Define target:
    target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    # visuals.plot_matter_state(target_4legged_cat_state, block_sphere_resolution=200)
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, target_4legged_cat_state)   
    
    # Define operations:    
    cat4_creation_operations = \
        noon_creation_operations + \
        rotation_operation + \
        noon_creation_operations + \
        rotation_operation + \
        noon_creation_operations + \
        rotation_operation 
            

    # Initital guess and the fixed params vs free params:
    num_noon_params = 8
    free  = ParamLock.FREE
    fixed = ParamLock.FIXED
    noon_data_params = [val for val in noon_data.params]
    noon_affiliation = list(range(1, num_noon_params+1))
    # noon_affiliation = [None]*num_noon_params
    noon_lockness    = [free]*num_noon_params  # [fixed]*8
    noon_bounds      = [None]*num_noon_params
    rot_bounds       = [(-pi, pi)]*3

    params_value       = noon_data_params + _rand(3)  + noon_data_params + _rand(3)  + noon_data_params + _rand(3)  
    params_affiliation = noon_affiliation + [None]*3  + noon_affiliation + [None]*3  + noon_affiliation + [None]*3  
    params_lockness    = noon_lockness    + [free]*3  + noon_lockness    + [free]*3  + noon_lockness    + [free]*3  
    params_bound       = noon_bounds      + rot_bounds+ noon_bounds      + rot_bounds+ noon_bounds      + rot_bounds
    assert lists.same_length(params_affiliation, params_lockness, params_value, params_bound)

    param_config : List[BaseParamType] = []
    for i, (affiliation, lock_state, initial_value, bounds) in enumerate(zip(params_affiliation, params_lockness, params_value, params_bound)):
        if lock_state == ParamLock.FREE:
            param_config.append(FreeParam(
                index=i, initial_guess=initial_value, affiliation=affiliation, bounds=bounds
            ))
        else:
            param_config.append(FixedParam(
                index=i, value=initial_value
            ))



    return initial_state, cost_function, cat4_creation_operations, param_config

def common_good_starts() -> Generator[list, None, None]:

    for item in [ \
        [ 
            2.78611668e+00,  
            8.78657591e-01, 
            -1.10548169e+01, 
            -9.17436114e-01,
            1.25958016e-01,  
            2.05399498e+00,  
            6.11934061e-02, 
            -1.14385562e+02,
            -7.42116525e-01, 
            2.28624127e+00,  
            1.44418193e-01, 
            -3.10637828e+00,
            2.74037410e+00,  
            3.14159265e+00, 
            -1.80498821e-02, 
            -5.26216960e-01,
            -3.73102342e-01
        ],
        [ 
            1.29715405e+00,  
            9.79621861e-01, 
            -1.18402567e+01,  
            5.85893730e-01,
            4.26152467e-01,  
            1.36222538e+00, 
            -2.23090306e+00,  
            -7.74818090e+01,
            -4.62497765e-02,  
            6.19195011e-03, 
            -2.87869076e-01, 
            -2.07285830e+00,
            3.14159265e+00,  
            2.20534006e+00,  
            6.02986735e-01,  
            9.82102284e-01,
            1.38114814e+00
        ],
        [  
            5.12334483,   
            2.08562615, 
            -14.46979106,  
            -1.75381598,   
            1.3261382,
            1.37128228,  
            -0.62730442,  
            -7.88884155,  
            -0.32050426,   
            3.13317347,
            1.57055123,  
            -1.6264514,    
            1.56699369,   
            3.14159265,  
            -1.56638219,
            -2.81588984,   
            0.82422727
        ],
        [ 
            3.59680480e+00,  
            8.70231489e-01, 
            -1.66371644e+01,  
            1.15964709e+00,
            2.77411784e+00,  
            1.61230528e+00, 
            -2.35460255e+00, 
            -9.05062544e+01,
            7.21027556e-01,  
            1.30767356e-02,  
            1.52088975e+00, 
            -1.75960138e+00,
            1.34089331e+00,  
            1.78832679e+00,  
            9.31994377e-01, 
            -7.45783960e-01,
            -8.12888428e-02
        ],
        [ 
            2.81235479e+00,  
            6.58500630e-01, 
            -1.91032004e+01, 
            -5.02577813e-02,
            7.56818763e-01,  
            2.40146756e+00, 
            -1.70876980e+00, 
            -9.43668349e+01,
            3.14065289e+00,  
            1.35396503e+00, 
            -6.96555278e-01, 
            -1.12360133e+00,
            1.47922973e+00,  
            2.54896639e+00,  
            1.44599870e+00, 
            -3.14159265e+00,
            9.64125752e-01
        ] 
    ]:
        yield item



# ==================================================================================== #
# |                                     main                                         | #
# ==================================================================================== #

class AttemptResult(NamedTuple):
    initial_guess : np.ndarray
    result : LearnedResults
    score : float

def exhaustive_search(
    base_guess          : np.ndarray,
    num_moments         : int = 40,
    num_tries           : int = 100,
    num_iter_per_try    : int = int(5*1e3),
    std                 : float = 0.5,
    plot_on             : bool = True
) -> List[AttemptResult]:

    # base_guess = np.array(
    #         [ 2.68167102e+00,  1.61405534e+00, -1.03042969e+01,  5.98736807e-02,
    #         1.26242432e+00,  1.47234240e+00, -1.71681054e+00, -8.64374806e+01,
    #         4.30847192e-01,  7.88459398e-01, -6.89081116e-02, -2.02854074e+00,
    #         2.23136298e+00,  3.14159265e+00,  3.60804145e-03, -2.18231897e+00,
    #         -5.95372440e-02]
    #    )

    all_attempts : List[AttemptResult] = []
    best_attempt : AttemptResult = AttemptResult(initial_guess=0, result=0, score=10)
    for i in range(num_tries):
        print("searching... "+strings.num_out_of_num(i+1, num_tries))
        try:
            guess = add_noise_to_vector(base_guess, std=std)
            res = _exhaustive_try(num_moments=num_moments, initial_guess=guess, num_iter=num_iter_per_try)
        except:
            print(f"Skipping try {i} due to an error.")
            continue
        score = res.score
        crnt_attempt = AttemptResult(initial_guess=guess, result=res, score=res.score)
        all_attempts.append(crnt_attempt)
        if crnt_attempt.score < best_attempt.score:
            best_attempt = crnt_attempt
            _print_progress(crnt_attempt)

    # Save results in a dict format that can be read without fancy containers:
    saved_var = dict(
        best_attempt= types.as_plain_dict(best_attempt),
        all_attempts= [types.as_plain_dict(attempt) for attempt in all_attempts],
        base_guess  = base_guess
    )
    file_name = "exhaustive_search "+strings.time_stamp()
    saveload.save(saved_var, name=file_name)
    print(f"Saving file with name '{file_name}'")
    
    # Plot best result:
    if plot_on:
        visuals.plot_matter_state(best_attempt.result.final_state)

    return all_attempts
    

def _print_progress(crnt_attempt:AttemptResult) -> None:
    print(f"New best result!  Fidelity = {crnt_attempt.score}")
    print(f"Theta = {crnt_attempt.result.theta}")
    print(f"Operation Params = {crnt_attempt.result.operation_params}")
    
    
@decorators.multiple_tries(3)
def _exhaustive_try(num_moments:int, initial_guess:np.ndarray, num_iter:int=MAX_NUM_ITERATION) -> LearnedResults:

    initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments)

    results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=initial_state, 
        cost_function=cost_function, 
        operations=cat4_creation_operations, 
        max_iter=num_iter, 
        parameters_config=param_config,
        initial_guess=initial_guess
    )
    
    return results


def creating_4_leg_cat_algo(
    num_moments:int=40
) -> LearnedResults:


    initial_guess = add_noise_to_vector( 
        np.array(
            [ 2.68167102e+00,  1.61405534e+00, -1.03042969e+01,  5.98736807e-02,
            1.26242432e+00,  1.47234240e+00, -1.71681054e+00, -8.64374806e+01,
            4.30847192e-01,  7.88459398e-01, -6.89081116e-02, -2.02854074e+00,
            2.23136298e+00,  3.14159265e+00,  3.60804145e-03, -2.18231897e+00,
            -5.95372440e-02]
       )
       , std=2.0
    )
        
    initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments)

    results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=initial_state, 
        cost_function=cost_function, 
        operations=cat4_creation_operations, 
        max_iter=MAX_NUM_ITERATION, 
        parameters_config=param_config,
        initial_guess=initial_guess
    )
    
    
    fidelity = -1 * results.score
    print(f"fidelity is { fidelity }")
    
    operation_params = results.operation_params
    print(f"operation params:")
    print(operation_params)
    
    final_state = results.final_state
    visuals.plot_matter_state(final_state)
    
    
    return results

    
    
def _load_all_search_data() -> Generator[dict, None, None]:
    for name, data in saveload.all_saved_data():
        splitted_name = name.split(" ")
        if splitted_name[0]=="exhaustive_search":
            yield data    
    

def _study():
    # results = exhaustive_search()
    # print("End")

    # search_results = saveload.load("key_saved_data//exhaustive_search 2022.12.31_08.37.25")
    # for attempt in search_results["all_attempts"]:
    #     score = attempt["score"]
    #     theta = attempt["result"]["theta"]
    #     if score < -0.6:
    #         print(theta)

    # all_results = []
    # for initial_guess in common_good_starts():
    #     results = exhaustive_search(
    #         base_guess=np.array(initial_guess),
    #         num_moments=40,
    #         num_tries=5,
    #         num_iter_per_try=int(1e5),
    #         plot_on=False,
    #         std=0.1
    #     )
    #     all_results += results
    # saveload.save(all_results, name="All best results "+strings.time_stamp())

    # print("Done.")
    
    # num_moments = 40
    
    # best_result : dict = {}
    # best_score = -0.7
    # count = 0
    # for data in _load_all_search_data():
    #     for result in data["all_attempts"]:
    #         count += 1
    #         if result["score"] < best_score:
    #             best_result = result
    
    # print(count)
    # print(best_result)
    # initial_guess = best_result["initial_guess"]
    # theta = best_result["result"]["theta"]
    
    
    # array([   3.03467614,    0.93387172,  -10.00699257,   -0.72388404,
    #       0.13744785,    2.11175319,    0.18788428, -118.69022356,
    #      -1.50210956,    2.02098048,   -0.21569011,    3.03467614,
    #       0.93387172,  -10.00699257,   -0.72388404,    0.13744785,
    #       2.11175319,    0.18788428, -118.69022356,   -2.9236711 ,
    #       3.01919738,    3.14159265,    3.03467614,    0.93387172,
    #     -10.00699257,   -0.72388404,    0.13744785,    2.11175319,
    #       0.18788428, -118.69022356,   -0.32642685,   -0.87976521,
    #      -0.83782409])
    
    # initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments)
    
    # for op, params in  pair_custom_operations_and_opt_params_to_op_params(cat4_creation_operations, theta, param_config):
    #     print(op.get_string(params))
    
    
    # results = _exhaustive_try(num_moments=num_moments, initial_guess=theta, num_iter=10)
    # print(results)
    
    
    num_moments = 40

    mode = "record_movie"
    
    # opt_theta = np.array(
    #     [   3.03467614,    0.93387172,  -10.00699257,   -0.72388404,
    #         0.13744785,    2.11175319,    0.18788428, -118.69022356,
    #         -1.50210956,    2.02098048,   -0.21569011,   -2.9236711 ,
    #         3.01919738,    3.14159265,   -0.32642685,   -0.87976521,
    #         -0.83782409])

    opt_theta = np.array(
      [   3.02985656,    0.89461558,  -10.6029319 ,   -0.75177908,
          0.17659927,    2.08111341,    0.30032648, -120.46353087,
         -1.51754475,    1.91694016,   -0.42664783,   -3.13543566,
          2.17021358,    3.14159224,   -0.26865575,   -0.92027109,
         -0.9889859 ])

    if mode=="optimize":

        results : LearnedResults = _exhaustive_try(num_moments=num_moments, initial_guess=opt_theta)        
        visuals.plot_matter_state(results.final_state)
        print(results)
    
    if mode=="record_movie":
        num_transition_frames = 20
        fps = 5


        initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments, num_transition_frames)
        
        operations = []
        theta = []
        for operation, oper_params in  pair_custom_operations_and_opt_params_to_op_params(cat4_creation_operations, opt_theta, param_config):
            print(operation.get_string(oper_params))
            theta.extend(oper_params)
            operations.append(operation)
            
        target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
        def _score_str_func(rho:_DensityMatrixType)->str:
            fidel = metrics.fidelity(rho, target_4legged_cat_state)
            return f"fidelity={fidel}"
        
            
        coherent_control = CoherentControl(num_moments=num_moments)
        movie_config = CoherentControl.MovieConfig(
            active=True,
            show_now=False,
            fps=fps,
            num_transition_frames=num_transition_frames,
            num_freeze_frames=fps,
            bloch_sphere_resolution=200,
            score_str_func=_score_str_func
        )
        final_state = coherent_control.custom_sequence(initial_state, theta=theta, operations=operations, movie_config=movie_config)
        print(_score_str_func(final_state))
    print("Done.")

def disassociate_affiliation()->LearnedResults:
    
    num_moments = 40
    num_transition_frames = 0
    
    # Copy best result with affiliated params:
    initial_state, cost_function, cat4_creation_operations, param_configs = _common_4_legged_search_inputs(num_moments, num_transition_frames)
    best_theta = np.array(
      [   3.02985656,    0.89461558,  -10.6029319 ,   -0.75177908,
          0.17659927,    2.08111341,    0.30032648, -120.46353087,
         -1.51754475,    1.91694016,   -0.42664783,   -3.13543566,
          2.17021358,    3.14159224,   -0.26865575,   -0.92027109,
         -0.9889859 ])
    '''
        ## Get operation params for best results:
        base_params = []
        for _, params in pair_custom_operations_and_opt_params_to_op_params(cat4_creation_operations, best_theta, param_configs):
            for param in params:
                base_params.append(param)        
            
        ## Disassociate affiliated params:
        param_configs : List[ParamConfigBase] 
        for config, value in zip(param_configs, base_params):
            if isinstance(config, FreeParam):
                config.affiliation = None
                config.initial_guess = value
            elif isinstance(config, FixedParam):
                config.value = value 
            print(config)
    '''
    param_configs = free_all_params(cat4_creation_operations, best_theta, param_configs)
     
    # Fix rotation params.
    # for i, param in enumerate( param_configs ): 
    #     print(f"i:{i:2}  {param.bounds}   ")
    #     if param.bounds != (None, None):
    #         param_configs[i] = param.fix()
    
    # Learn:
    results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=initial_state, 
        cost_function=cost_function, 
        operations=cat4_creation_operations, 
        max_iter=MAX_NUM_ITERATION, 
        parameters_config=param_configs
    )
    print(results)
    return results

    
    
    

def _sx_sequence_params(
    standard_operations:CoherentControl.StandardOperations, 
    sigma:float=0.0, 
    theta:Optional[List[float]]=None,
    num_free_params:Optional[int]=None
)-> Tuple[
    List[BaseParamType],
    List[Operation]
]:
    
    rotation    = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    p2_pulse    = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])
    stark_shift = standard_operations.stark_shift_and_rot()
        
    eps = 0.1    
        
    _rot_bounds   = lambda n : [(-pi-eps, pi+eps)]*n
    _p2_bounds    = lambda n : _rot_bounds(n) # [(None, None)]*n
    _stark_bounds = lambda n : [(None, None)]*n
    
    _rot_lock   = lambda n : [False]*n 
    _p2_lock    = lambda n : [False]*n
    _stark_lock = lambda n : [False]*n
    
    # previos_best_values = [
    #     -3.13496905e+00,  6.04779209e-01, -2.97065809e+00,  # rot
    #     -7.21249786e-01,  7.92523986e-02,                   # p2
    #      0.0           ,  0.0           ,  0.0,             # stark-shift
    #     -2.26480360e-01, -3.06453241e+00, -7.77837060e-01,  # rot
    #      1.89698575e-01,  1.44668992e-03,                   # p2
    #      0.0           ,  0.0           ,  0.0,             # stark-shift
    #      1.10893652e+00, -6.32039487e-02,  2.43629268e+00,  # rot
    #      1.39075989e-01, -5.08093640e-03,                   # p2
    #      2.03338557e+00,  3.54986211e-01,  1.23905514e+00   # rot
    # ]
    # previous_best_values = [
    #     -3.14132341e+00,  7.26499599e-01, -2.81640184e+00, 
    #     -7.21249786e-01,  0.0,
    #     -2.24533824e-01, -3.06451820e+00, -8.04970536e-01,
    #      1.89698575e-01,  0.0,
    #     -1.20910238e-03, -1.13211280e-03,  1.22899846e-03,
    #      1.39075989e-01,  0.0,
    #      2.03532868e+00,  3.53830170e-01,  1.23912615e+00
    # ]
    # previous_best_values = [-3.14159265e+00,  7.48816345e-01, -3.14158742e+00, -1.11845072e+00,
    #    -2.54551167e-01, -1.29370443e-04, -3.00227534e+00, -8.38431494e-01,
    #     5.62570928e-02, -1.00562122e-01,  5.38651426e-02,  1.86925119e-02,
    #     1.29525864e-01,  2.92131952e-01,  5.46879499e-02,  2.13122296e+00,
    #     3.05040262e-01,  8.45120583e-01]
    
    # previous_best_values = [
    #     -3.14030032,  1.56964558, -1.23099066, -1.06116912, 
    #     -0.20848513, -0.00737964, -2.79015456, -1.1894669 ,
    #      0.04517585, -0.10835456, -0.03379094, -0.12314313,
    #      0.17918845,  0.31359876,  0.07170169,  2.24711138,  0.36310499,  0.91055266
    # ]
    
    # previous_best_values = [ 
    #     -3.14030032,     1.56964558,     -1.23099066,     -1.06116912,     -0.20848513,
    #     3.61693959e-03, -2.75791882e+00, -1.13001522e+00,  4.37562070e-02,
    #    -1.09556315e-01,  3.00420606e-02, -1.89985052e-01,  1.90603431e-01,
    #     3.08800775e-01,  7.66210890e-02,  2.14303499e+00,  2.61838147e-01,
    #     8.67099173e-01,  7.57309369e-03,  2.43802801e-03, -3.69501207e-03,
    #    -9.91619336e-03,  2.54274091e-02
    # ]
    
    # previous_best_values = [-3.14030032,  1.56964558, -1.23099066, -1.06116912, -0.20848513,
    #     0.00361694, -2.75791882, -1.13001522,  0.04375621, -0.10955632,
    #     0.01501858, -0.18233922,  0.18501545,  0.31482114,  0.07721763,
    #     2.04140464,  0.17904715,  1.29808372,  0.01239275,  0.00664981,
    #    -0.11626285,  0.32799851, -0.14023262]
    
    # previous_best_values = [
    #     -3.11809877e+00,  1.80059370e+00, -1.55462574e+00, -1.08964755e+00,
    #    -2.14198793e-01,  8.51353082e-03, -2.62218220e+00, -1.11777407e+00,
    #     2.17978038e-02, -1.16416204e-01, -1.25477684e-02, -2.92207523e-01,
    #     2.38461118e-01,  3.13539220e-01,  7.49079999e-02,  1.97132229e+00,
    #     4.73772111e-02,  1.23930114e+00,  1.40898647e-02,  7.34119457e-03,
    #    -8.82234915e-02,  3.67593470e-01,  1.61897263e-03,
    #    0.0, 0.0, 0.0, 0.0, 0.0, 
    # ]
    
    # previous_best_values = [
    #     0.0, 0.0, 0.0, 0.0, 0.0, 
    #    -2.73964174e+00,  2.03423352e+00, -1.62928889e+00, -1.08470481e+00,
    #    -2.10877461e-01, -2.80748413e-02, -2.56002461e+00, -1.09108989e+00,
    #     2.29558918e-02, -1.15905236e-01, -7.05754005e-03, -3.56090360e-01,
    #     2.40560895e-01,  3.12555987e-01,  7.45061506e-02,  1.99923603e+00,
    #    -1.49483597e-02,  1.17152967e+00,  1.50856556e-02,  7.67289787e-03,
    #    -1.00619005e-01,  3.27342370e-01,  2.63205029e-02,  7.41929725e-04,
    #     9.55277316e-04,  3.46883173e-03,  4.86919756e-04,  2.12570641e-03
    # ]
    
    # previous_best_values = [ 2.64551298e-02, -1.16495825e-01, -9.04855043e-03, -3.87190166e-02,
    #    -1.77760170e-01, -2.27747555e+00,  2.30208055e+00, -1.65254288e+00,
    #    -1.09125669e+00, -2.13360234e-01, -2.12295400e-02, -2.56579479e+00,
    #    -1.07664641e+00,  2.19823025e-02, -1.16139925e-01, -4.95983723e-03,
    #    -3.62125331e-01,  2.38295456e-01,  3.11773381e-01,  7.49915536e-02,
    #     2.05874179e+00, -6.58375089e-02,  1.25318017e+00,  1.25353393e-02,
    #     3.72244913e-03, -1.94542231e-01,  3.47964229e-01,  6.49519029e-03,
    #     2.06871716e-03,  4.03951412e-03,  2.95672995e-03,  1.47839729e-02,
    #     2.66335759e-02]
    
    # previous_best_values =  [
    #     -6.11337359e-02, -4.48316330e-02, -2.82676017e-02, -3.25561302e-02,
    #    -1.98010475e-01, -2.31842421e+00,  2.36194099e+00, -1.70163429e+00,
    #    -1.08924675e+00, -2.10204448e-01, -2.75792429e-02, -2.55973218e+00,
    #    -1.05334435e+00,  2.12767325e-02, -1.16811652e-01,  4.36041716e-03,
    #    -3.71563640e-01,  2.29741367e-01,  3.13366952e-01,  7.62151375e-02,
    #     1.90917446e+00, -3.62066282e-01,  1.05373414e+00,  1.47731610e-02,
    #     1.99157089e-02, -1.08752683e-01,  2.72017441e-01, -2.72095759e-03,
    #     8.58682076e-04, -1.17743989e-02,  6.01113733e-02,  1.02590587e-01,
    #     3.39742620e-01]
    
    # previous_best_values = [
    #     -7.13588363e-02, -3.80137188e-02, -2.39536167e-02, -3.27949400e-02,
    #    -1.96901816e-01, -2.31905117e+00,  2.37206637e+00, -1.70061176e+00,
    #    -1.08833084e+00, -2.10303395e-01, -2.41519298e-02, -2.56865778e+00,
    #    -1.05087505e+00,  2.14462515e-02, -1.16775193e-01,  4.81643506e-03,
    #    -3.65193448e-01,  2.27217170e-01,  3.14479557e-01,  7.65798517e-02,
    #     1.90738399e+00, -4.26370890e-01,  1.05314249e+00,  1.61049232e-02,
    #     2.31567874e-02, -1.12499156e-01,  2.95158748e-01, -2.82196522e-03,
    #     5.08932313e-04, -1.38633239e-02,  4.88258818e-02,  9.38795946e-02,
    #     3.75208008e-01,
    #     0.0, 0.0, 0.0, 0.0, 0.0
    #     ]
    
    # previous_best_values = [-0.07135884, -0.03801372, -0.02395362, -0.03279494, -0.19690182,
    #    -2.31905117,  2.37206637, -1.70061176, -1.08833084, -0.2103034 ,
    #    -0.02415193, -2.56865778, -1.05087505,  0.02144625, -0.11684027,
    #     0.0026565 , -0.36551299,  0.22454328,  0.32622168,  0.07919275,
    #     1.77137189, -0.84950465,  1.12921608,  0.01628281,  0.03214572,
    #    -0.16109355,  0.24460846,  0.05098917,  0.00677852, -0.01135189,
    #     0.04941705,  0.36159298,  0.48542749, -0.00566101, -0.00541848,
    #    -0.0377712 ,  0.00684348,  0.12901692]
    
    # previous_best_values = [ 4.38619759e-02, -6.91810495e-02, -4.13109278e-02, -4.11018378e-02,
    #    -2.29033787e-01, -2.16721255e+00,  2.37077763e+00, -1.63013622e+00,
    #    -1.08201860e+00, -2.18180098e-01,  1.50771348e-02, -2.60030023e+00,
    #    -1.11920058e+00,  2.06021828e-02, -1.16682961e-01,  3.29579001e-03,
    #    -3.24155964e-01,  2.32194472e-01,  3.14474665e-01,  7.55065680e-02,
    #     1.70227218e+00, -8.24755200e-01,  1.10570026e+00,  3.88760181e-03,
    #     4.16635630e-02, -3.65548520e-02,  3.21201129e-01,  5.54161846e-02,
    #     6.45307529e-03, -3.56502024e-02, -6.42542775e-04,  3.38947625e-01,
    #     8.26304247e-01, -1.00921906e-02, -7.28238291e-03, -4.18837327e-03,
    #     6.01208928e-03,  8.70003401e-02]
    
    # previous_best_values = [-1.75183802e-02,  1.20829221e-02, -4.63423395e-02, -4.47289386e-02,
    #    -2.62899880e-01, -2.03732856e+00,  2.41325602e+00, -1.71053215e+00,
    #    -1.08140467e+00, -2.25058649e-01,  5.31529283e-02, -2.57407587e+00,
    #    -1.12473686e+00,  2.96311647e-02, -1.12157918e-01, -5.07889800e-03,
    #    -3.29336078e-01,  2.57622876e-01,  3.36550247e-01,  8.23519473e-02,
    #     1.69351080e+00, -9.66637051e-01,  8.43512892e-01, -1.12450440e-04,
    #     4.15594406e-02,  2.62156428e-02,  1.87023060e-01,  1.93246264e-01,
    #     1.18243861e-02, -1.63150806e-02, -1.44509345e-03,  4.05487863e-01,
    #     9.06370380e-01, -3.04902748e-02, -1.79484342e-02,  2.93459715e-03,
    #    -2.78165258e-03,  1.58611775e-01]
    
    
    # previous_best_values = [-1.01118870e-02, -1.17195746e-02, -1.80054590e-01, -4.24771380e-02,
    #    -2.60850023e-01, -2.06515131e+00,  2.39039228e+00, -1.74027661e+00,
    #    -1.08851647e+00, -2.25390084e-01,  4.25341691e-02, -2.56779503e+00,
    #    -1.11588157e+00,  3.15792476e-02, -1.11728279e-01, -2.32619327e-02,
    #    -3.39439405e-01,  2.50145720e-01,  3.44890936e-01,  8.71619748e-02,
    #     1.77633218e+00, -8.98407589e-01,  8.25653641e-01,  2.72808824e-03,
    #     4.77637403e-02, -3.45036330e-02,  6.22110587e-02,  2.66726277e-01,
    #     1.44363630e-02, -2.27761602e-02, -4.12024564e-03,  4.40959049e-01,
    #     7.38371824e-01, -4.13439740e-02, -2.13990400e-02,  1.60669395e-03,
    #    -3.84366145e-03,  2.93606588e-01]
    
    # previous_best_values = [
    #     0.0, 0.0, 0.0, 0.0, 0.0, 
    #     -5.21647351e-03,  3.64780064e-03,  1.20317122e-02, -4.12743795e-02,
    #    -2.65510012e-01, -2.07504236e+00,  2.38107513e+00, -1.74325902e+00,
    #    -1.08841124e+00, -2.26276465e-01,  5.04960332e-02, -2.59261655e+00,
    #    -1.11872351e+00,  3.17614938e-02, -1.11687854e-01, -3.08959109e-02,
    #    -3.16065208e-01,  2.44773268e-01,  3.50447005e-01,  8.86457067e-02,
    #     1.83953416e+00, -8.73191456e-01,  9.69922919e-01,  2.73776070e-03,
    #     4.46671650e-02, -9.16588448e-02,  7.51263302e-02,  2.33997781e-01,
    #     2.26520181e-02, -1.73708288e-02, -7.91001370e-03,  4.65841533e-01,
    #     5.42649379e-01, -4.04111395e-02, -2.03337020e-02,  1.01744864e-03,
    #    -3.93635191e-03,  2.99703456e-01]
    
    # previous_best_values = [
    #     -4.91916303e-01,  8.21191482e-01,  2.51136612e-01,  3.89854737e-03,
    #     9.85478496e-03,  9.77317446e-01,  2.66997039e-01,  1.84503474e+00,
    #    -4.17032982e-02, -2.65510012e-01, -2.07504236e+00,  2.38107513e+00,
    #    -1.74325902e+00, -1.08841124e+00, -2.26276465e-01,  5.04960332e-02,
    #    -2.59261655e+00, -1.11872351e+00,  3.17614938e-02, -1.11687854e-01,
    #    -3.08959109e-02, -3.16065208e-01,  2.44773268e-01,  3.50447005e-01,
    #     8.86457067e-02,  1.83953416e+00, -8.73191456e-01,  9.69922919e-01,
    #     2.73776070e-03,  4.46671650e-02, -9.16588448e-02,  7.51263302e-02,
    #     2.33997781e-01,  2.26520181e-02, -1.73708288e-02, -7.91001370e-03,
    #     4.65841533e-01,  5.42649379e-01, -4.04111395e-02, -2.03337020e-02,
    #     1.01744864e-03, -3.93635191e-03,  2.99703456e-01]
    
    # previous_best_values = [-4.65599879e-01,  1.09626589e+00,  2.22809021e-01,  2.49183098e-03,
    #     3.05748654e-02,  8.46440624e-01,  1.33884866e-01,  1.74667034e+00,
    #    -4.18136520e-02, -2.60726431e-01, -2.06676524e+00,  2.35920560e+00,
    #    -1.69172887e+00, -1.09101560e+00, -2.29169671e-01,  5.21814256e-02,
    #    -2.59257206e+00, -1.14376999e+00,  2.98465059e-02, -1.12562215e-01,
    #    -3.00964523e-02, -3.07537626e-01,  2.43310898e-01,  3.48520882e-01,
    #     8.91033840e-02,  1.80458172e+00, -9.23239671e-01,  9.35685413e-01,
    #     1.06493740e-03,  4.01018298e-02, -1.11016074e-01,  1.23799150e-01,
    #     3.21359879e-01,  2.12791291e-02, -1.23564660e-02, -6.22889890e-03,
    #     4.65164692e-01,  6.15427419e-01, -3.98506106e-02, -1.94804303e-02,
    #     1.08380551e-03, -5.18309596e-03,  1.83224320e-01]
    
    # previous_best_values = [
    #     -6.41147017e-01,  1.05317692e+00,  6.22077149e-02,  4.66610387e-03,
    #     4.25994982e-02,  6.56875809e-01,  2.27281263e-01,  1.49239284e+00,
    #    -4.01105997e-02, -2.46674985e-01, -2.08128070e+00,  2.33380115e+00,
    #    -1.75113727e+00, -1.09547189e+00, -2.29482130e-01,  9.32683740e-02,
    #    -2.64714473e+00, -1.12244203e+00,  3.07893884e-02, -1.10701277e-01,
    #    -5.48478211e-02, -2.60838908e-01,  2.46251415e-01,  3.50987137e-01,
    #     9.14777387e-02,  1.80673525e+00, -8.82070930e-01,  9.03482423e-01,
    #    -1.46225057e-03,  4.29119718e-02, -1.05235322e-01,  1.17429097e-01,
    #     3.72348304e-01,  2.02420546e-02, -1.62303160e-02,  8.11828272e-03,
    #     4.79643530e-01,  7.30229014e-01, -4.68518924e-02, -2.16909209e-02,
    #     8.22392527e-04, -1.66850155e-02,  9.94765080e-02
    # ]
    
    # previous_best_values =  [
    #     -0.627024806777948, 1.0480377284716385, 0.06430438315044351, 0.0032263540563926447, 0.042253450558402905, 
    #     0.640230854107747, 0.2132003005846988, 1.5146415228333874, -0.040076126527660606, -0.24669396370833502, 
    #     -2.0826929721203724, 2.333213261948929, -1.7413976103528936, -1.0952382810917332, -0.2294388778543284, 
    #     0.09064397638445214, -2.6374338460196647, -1.1292470489109092, 0.031011264776644672, -0.11041562200398725, 
    #     -0.05188148994545726, -0.2680211590512026, 0.24775370278097503, 0.3511686777671562, 0.09123334540947106, 
    #     1.8063181041922207, -0.8717452751457218, 0.88937444668243, -0.0023117256333004483, 0.04356195297841235, 
    #     -0.1014986016889961, 0.13629914380457014, 0.3631426872734995, 0.019826151533230352, -0.016042262461581535, 
    #     0.00843613400165829, 0.48155093672629573, 0.7391647178724947, -0.046620946710130014, -0.021475800956105327, 
    #     0.026681210502736457, -0.03237844167634448, 0.08373798132319596
    # ]
    
    # previous_best_values = [-0.47595144837934417, 1.0524650373861704, 0.11899362035094888, 0.0038734140503491905, 0.04019117732279853, 0.6547273557595066, 0.14979464751529864, 1.6318703840768194, -0.040226257224637095, -0.24984801010669, -2.085717670983983, 2.3230660056054386, -1.7476407679636696, -1.094254062606245, -0.22847955805236903, 0.08531543253999062, -2.625673290913198, -1.129719153446827, 0.030842526278989223, -0.10953205760973363, -0.05154813549984682, -0.28130682544338614, 0.2494463200963285, 0.35070233052562194, 0.09103381002153813, 1.7743827475189797, -0.7994344336349177, 0.8818754483585529, -0.009631227698231597, 0.04808973021184635, -0.05941691779947954, 0.24361184988357193, 0.3368023960598062, 0.01932312684638582, -0.018061589250921405, -0.02880173757253086, 0.4987486143327268, 0.8423185930550972, -0.04862164878804154, -0.021335033058846757, 0.163755525532994, -0.049597526625344446, 0.0016411107815335777]
    # previous_best_values = [-0.18062995286126304, 0.686601568470758, -0.00014207349494701186, -0.026650386554554033, 0.07104882938691356, 1.3314655417385, -0.5945473870293823, 2.7145254094136684, -0.0463325026681872, -0.3106208910544658, -2.093758262431699, 2.329345378038191, -1.7008865045327433, -1.0954447602583839, -0.22958706024030417, 0.19422372347096378, -2.6848931951925854, -1.2002066979099975, 0.03622460363430238, -0.10819181381150772, -0.03871950750357038, -0.1976558387636006, 0.25907391546313974, 0.36767275075911754, 0.1035281377412448, 1.846552214300687, -0.6426052337133896, 0.5076744852606687, -0.0067080582810151845, 0.0588227096061351, 0.07482810043888241, 0.2276812212194777, 0.4098119827573099, 0.015101999811002485, -0.01629749900110391, -0.007018030149747199, 0.3729819053957242, 0.7934572807229194, -0.057786813223948186, -0.021503788969318136, 0.23377780864081887, -0.22778509993517387, -0.20105998303366746]
    
    # previous_best_values = [0.12308380809937691, 0.6167734366901824, -6.905234447658942e-08, -0.020520710568598784, 0.11317656599087786, 2.203574030983476, -1.2165615756314654, 1.5865782411475107, -0.043688968977686524, -0.2995743263556788, -2.078385587264667, 2.3356929508496975, -1.6926246368805333, -1.094556829870168, -0.22989374239288535, 0.1947870346992607, -2.701978909675484, -1.1759241949342476, 0.03930721891294668, -0.10750615767231642, -0.040050929357376384, -0.20057963137905874, 0.2231770639831383, 0.37413984382731547, 0.1112855477394121, 1.708046063939085, -0.4632992307621673, 0.1257150957306162, 
    # -0.013135580142023442, 0.07322171079543743, 0.2853078560106997, 0.3468853380585464, 0.517710247262364, 0.012060097822820217, -0.03756110345047371, 0.2294672561505801, -0.04665547115876428, 0.5901985941516392, -0.04473967411033044, 0.012075942959207164, 0.20678231829876248, -0.19092155718132772, -0.371885886330388]
    
    # previous_best_values = [0.12295085237945075, 0.6177386729951149, -6.046183567565546e-08, -0.02055514334837799, 0.11344018368397785, 2.205047417468453, -1.2175769978069342, 1.583002794505349, -0.043676489382316414, 
        # -0.2995782041325914, -2.0782800523149945, 2.3357491026220014, -1.693304382070842, -1.094527167395321, -0.2298945791651671, 0.19461850337228093, -2.7022663307099055, -1.1754118407171408, 0.03930335537437521, -0.10751241076037665, -0.04012649952060393, -0.20059943965216664, 0.22305282914409175, 0.37429903579008483, 0.1113750857420232, 1.7084743721345035, -0.45814118430797235, 0.12021398204350386, -0.013176069825133169, 0.07316386325300506, 0.28899415155784647, 0.3447638291119759, 0.5251986513278077, 0.01222713459656792, -0.03708843028732621, 0.23725833705844723, -0.05495822295453015, 0.5839414454690997, -0.044903554475658775, 0.012271614026706748, 0.19806485320982892, -0.18747308723994116, -0.38357461279684113]
    
    # previous_best_values = [0.12323862822219994, 0.6183957249611867, -1.0165705742142345e-07, -0.02053378563041057, 0.1134885760352155, 2.2063184637632265, -1.2176993620340708, 1.581510552917055, -0.04368232302559942, 
    #     -0.2995771342128086, -2.078251685103651, 2.3357218100968105, -1.693370885695495, -1.094530222114177, -0.22991592079495887, 0.19458565700134434, -2.7022122279835763, -1.1754316964883986, 0.03930662498182427, -0.10751764456215751, -0.0399440723377351, -0.20068318144572594, 0.22289552985902322, 0.3743526040761691, 0.11138786358535796, 1.7083721852489306, -0.4541474115505293, 0.1160298920050179, -0.01320059768117341, 0.07305394210779279, 0.2919964803548851, 0.34185647226471017, 0.5299098416846537, 0.012334272750180918, -0.03678861063388446, 0.24186863038556694, -0.06018242363818872, 0.5800694910999753, -0.045004969045353505, 0.012395824029279932, 0.19218916830360588, -0.18519043725667417, -0.39014428728685546]
    
    previous_best_values = [0.1230243800839618, 0.6191720299851127, -2.3280344384240303e-07, -0.020563759284078914, 0.1135349628174986, 2.20705196071948, -1.2183340894470418, 1.5799032500057237, -0.0436873903142408, -0.2995503422788831, -2.078190942922463, 2.335714330675413, -1.6935087152480237, -1.094542478123508, -0.22991275654402593, 0.19452686725055338, -2.70221081838102, -1.1752795377491556, 0.03932773593530256, -0.10750609661547705, -0.03991859479109913, -0.20072364056375158, 0.22285496406775507, 0.3743729432388033, 0.11137590080067977, 1.709423376749869, -0.45020803849068647, 0.11283133096297475, -0.013141785459664383, 0.07282695266780875, 0.2946167310023212, 0.3338135564683993, 0.5344263960722166, 0.012467076665257853, -0.03637397049464164, 0.2473014913597948, -0.06283368220768366, 0.5773412763402044, -0.04521543808835432, 0.012247470785197952, 0.18238622202996205, -0.1823704254987203, -0.3945560457085364]
    
    if theta is None:
        theta = previous_best_values
    
    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation,  p2_pulse, rotation
    ]
    num_operation_params : int = sum([op.num_params for op in operations])
    assert num_operation_params==len(theta)
    params_value = lists.add(theta, _rand(num_operation_params, sigma=sigma))
    
    params_bound = []
    params_lock  = []
    for op in operations:
        n = op.num_params
        if op is rotation:
            params_bound += _rot_bounds(n)
            params_lock  += _rot_lock(n)
        elif op is stark_shift:
            params_bound += _stark_bounds(n)
            params_lock  += _stark_lock(n)
        elif op is p2_pulse:
            params_bound += _p2_bounds(n)
            params_lock  += _p2_lock(n)
        else:
            raise ValueError("Not an option")
    
    assert len(params_value)==len(params_bound)==num_operation_params==len(params_lock)  
    param_config : List[BaseParamType] = []
    for i, (initial_value, bounds, is_locked) in enumerate(zip(params_value, params_bound, params_lock)):        
        if is_locked:
            this_config = FixedParam(index=i, value=initial_value)
        else:
            this_config = FreeParam(index=i, initial_guess=initial_value, bounds=bounds, affiliation=None)   # type: ignore       
        param_config.append(this_config)
        
    #Lock first operations:
    if num_free_params is None:
        num_fixed_params = 0
    else:
        num_fixed_params = num_operation_params-num_free_params
    param_config = fix_random_params(param_config, num_fixed_params)
    assert num_fixed_params == sum([1 if param.lock==ParamLock.FIXED else 0 for param in param_config ])
    
    return param_config, operations          
    
    
def optimized_Sx2_pulses(num_attempts:int=1, num_runs_per_attempt:int=int(1e5), num_moments:int=40, num_transition_frames:int=0) -> LearnedResults:
    # Similar to previous method:
    _, cost_function, _, _ = _common_4_legged_search_inputs(num_moments, num_transition_frames=0)
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)

    best_result : LearnedResults = None
    best_score = np.inf
    
    for attempt_ind in range(num_attempts):
        print(strings.num_out_of_num(attempt_ind+1, num_attempts))
        
        param_config, operations = _sx_sequence_params(standard_operations, sigma=0.000)
        
        try:            
            results = learn_custom_operation(
                num_moments=num_moments, 
                initial_state=initial_state, 
                cost_function=cost_function, 
                operations=operations, 
                max_iter=num_runs_per_attempt, 
                parameters_config=param_config
            )
        
        except Exception as e:
            errors.print_traceback(e)
        
        else:
            if results.score < best_score:
                print("Score: ",results.score)
                print("Theta: ",results.theta)
                best_result = results
                best_score = results.score
        
    return best_result
    
    
def optimized_Sx2_pulses_by_partial_repetition(
    num_moments:int=40, 
    num_attempts:int=1000, 
    num_runs_per_attempt:int=int(100*1e3), 
    num_free_params:int|None=None
) -> LearnedResults:
    
    # Constants:
    num_transition_frames:int=0
    
    # Similar to previous method:
    _, cost_function, _, _ = _common_4_legged_search_inputs(num_moments, num_transition_frames=num_transition_frames)
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    
    # Params and operations:
    
    param_config, operations = _sx_sequence_params(standard_operations)
    base_theta = [param.get_value() for param in param_config]
    results = None
    score:float=np.inf
   
    for attempt_ind in range(num_attempts):
    
        if results is None:
            _final_state = coherent_control.custom_sequence(initial_state, theta=base_theta, operations=operations)
            score = cost_function(_final_state)
            theta = base_theta

        elif results.score < score:
            score = results.score
            theta = results.operation_params
            print(f"score: {results.score}")
            print(f"theta: {list(theta)}")
        
        else: 
            pass
            # score stays the best score
            # theta stays the best theta
            
        param_config, operations = _sx_sequence_params(standard_operations, sigma=0.002, theta=theta, num_free_params=num_free_params)            
        print(strings.num_out_of_num(attempt_ind+1, num_attempts))
        
        try:            
            results = learn_custom_operation(
                num_moments=num_moments, 
                initial_state=initial_state, 
                cost_function=cost_function, 
                operations=operations, 
                max_iter=num_runs_per_attempt, 
                parameters_config=param_config
            )
        
        except Exception as e:
            errors.print_traceback(e)
        

    assert results is not None
    return results

if __name__ == "__main__":
    # _study()
    # results = disasseociate_affiliation()
    results = optimized_Sx2_pulses_by_partial_repetition()
    print("Done.")