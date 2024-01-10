# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.config_verifier
===========================

Verification of configuration parameters.
"""

import os
from cohere_core.utilities.config_errors_dict import *

__author__ = "Barbara Frosik, Dave Cyl"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['verify']
           

def ver_list_int(param_name, param_value):
    """
    This function verifies if all elements in a given list are int.

    Parameters
    ----------
    param_name : str
        the parameter being evaluated

    param_value : list
        the list to evaluate for int values

    Returns
    -------
    eval : boolean
        True if all elements are int, False otherwise
    """
    if not issubclass(type(param_value), list):
        print (param_name + ' is not a list')
        return False
    for e in param_value:
        if type(e) != int:
            print (param_name + ' should be list of integer values')
            return False
    return True


def ver_list_float(param_name, param_value):
    """
    This function verifies if all elements in a given list are float.

    Parameters
    ----------
    param_name : str
        the parameter being evaluated

    param_value : list
        the list to evaluate for float values

    Returns
    -------
    eval : boolean
        True if all elements are float, False otherwise
    """
    if not issubclass(type(param_value), list):
        print (param_name + ' is not a list')
        return False
    for e in param_value:
        if type(e) != float:
            print (param_name + ' should be list of float values')
            return False
    return True


def get_config_error_message(config_file_name, map_file, config_parameter, config_error_no):
    """
    This function returns an error message string for this config file from the error map file using
    the parameter and error number as references for the error.

    :param config_file_name: The config file being verified
    :param map_file: The dictionary of error dictionary files
    :param config_parameter: The particular config file parameter being tested
    :param config_error_no: The error sequence in the test
    :return: An error string describing the error and where it was found
    """
    config_map_dic = config_map_names.get(map_file)

    error_string_message = config_map_dic.get(config_parameter)[config_error_no]
    # presented_message = "File=" + config_file_name, "Parameter=" + config_parameter, "Error=" + error_string_message

    return(error_string_message)


def ver_config(config_map):
    """
    This function verifies experiment main config file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_error_map_file'
    fname = 'config'

    config_parameter = 'Workingdir'
    if 'working_dir' in config_map:
        working_dir = config_map['working_dir']
        if type(working_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
    else:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print(error_message)
        return error_message

    config_parameter = 'ExperimentID'
    if 'experiment_id' in config_map:
        experiment_id = config_map['experiment_id']
        if type(experiment_id) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
    else:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print(error_message)
        return error_message

    config_parameter = 'Scan'
    if 'scan' in config_map:
        scan = config_map['scan']
        if type(scan) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    return ("")


def ver_config_rec(config_map):
    """
    This function verifies experiment config_rec file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    import string

    config_map_file = 'config_rec_error_map_file'
    fname = 'config_rec'

    config_parameter = 'Datadir'
    if 'data_dir' in config_map:
        data_dir = config_map['data_dir']
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        if not os.path.isdir(data_dir):
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        if not os.path.isfile(data_dir + '/data.tif') and not os.path.isfile(data_dir + '/data.npy'):
            config_error = 2
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Savedir'
    if 'save_dir' in config_map:
        save_dir = config_map['save_dir']
        if type(save_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Initguess'
    if 'init_guess' in config_map:
        init_guess = config_map['init_guess']
        init_guess_options = ['random', 'continue', 'AI_guess']
        if init_guess not in init_guess_options:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        elif init_guess == 'continue':
            config_parameter = 'Continuedir'
            if 'continue_dir' in config_map:
                continue_dir = config_map['continue_dir']
                if type(continue_dir) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return error_message
            else:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return error_message
        elif init_guess == 'AI_guess':
            config_parameter = 'Aitrainedmodel'
            if 'AI_trained_model' in config_map:
                AI_trained_model = config_map['AI_trained_model']
                if type(AI_trained_model) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(config_error)
                    return error_message
            else:
                config_error = 1
                print(fname, config_map_file, config_parameter, config_error)
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return error_message

    config_parameter = 'Reconstruction'
    if 'reconstructions' in config_map:
        reconstructions = config_map['reconstructions']
        if type(reconstructions) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Device'
    if 'device' in config_map:
        device = config_map['device']
        if not ver_list_int('device', device):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Algorithmsequence'
    if 'algorithm_sequence' in config_map:
        algorithm_sequence = config_map['algorithm_sequence']
        if type(algorithm_sequence) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print (config_error)
            return (error_message)
        # check for supported characters
        alg_seq_chars = list(string.ascii_lowercase) + list(string.ascii_uppercase) + list(string.digits) + ['*', '+', '(', ')', ' ']
        if 0 in [c in alg_seq_chars for c in algorithm_sequence]:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(config_error)
            return (error_message)
        # check brackets, nested are not allowed
        br_count = 0
        for c in algorithm_sequence:
            if c == '(':
                br_count += 1
                if br_count > 1:
                    break
            elif c == ')':
                br_count -= 1
                if br_count < 0:
                    break
        if br_count != 0:
            config_error = 2
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(config_error)
            return (error_message)
    else:
        config_error = 3
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print (error_message)
        return (error_message)

    config_parameter = 'Hiobeta'
    if 'hio_beta' in config_map:
        hio_beta = config_map['hio_beta']
        if type(hio_beta) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Initialsupportarea'
    if 'initial_support_area' in config_map:
        initial_support_area = config_map['initial_support_area']
        if not issubclass(type(initial_support_area), list):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        for e in initial_support_area:
            if type(e) != int and type(e) != float:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Generations'
    if 'ga_generations' in config_map:
        generations = config_map['ga_generations']
        if type(generations) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        if 'reconstructions' in config_map:
            reconstructions = config_map['reconstructions']
        else:
            reconstructions = 1
        if reconstructions < 2:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Gametrics'
        if 'ga_metrics' in config_map:
            ga_metrics = config_map['ga_metrics']
            if not issubclass(type(ga_metrics), list):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print (error_message)
                return (error_message)
            metrics_options = ['chi', 'sharpness', 'summed_phase', 'area']
            for metric in ga_metrics:
                if metric not in metrics_options:
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print (error_message)
                    return (error_message)

        config_parameter = 'Gabreedmodes'
        if 'ga_breed_modes' in config_map:
            ga_breed_modes = config_map['ga_breed_modes']
            if not issubclass(type(ga_breed_modes), list):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print (error_message)
                return (error_message)
            breed_options = ['none', 'sqrt_ab', 'dsqrt', 'pixel_switch', 'b_pa', '2ab_a_b', '2a_b_pa', 'sqrt_ab_pa',\
'sqrt_ab_pa_recip', 'sqrt_ab_recip', 'max_ab', 'max_ab_pa', 'min_ab_pa', 'avg_ab', 'avg_ab_pa']
            for breed in ga_breed_modes:
                if breed not in breed_options:
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print (error_message)
                    return (error_message)

        config_parameter = 'Gacullings'
        if 'ga_cullings' in config_map:
            ga_cullings = config_map['ga_cullings']
            if not ver_list_int('ga_cullings', ga_cullings):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
            if sum(ga_cullings) >= reconstructions:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Gashrinkwrapthresholds'
        if 'ga_shrink_wrap_thresholds' in config_map:
            ga_shrink_wrap_thresholds = config_map['ga_shrink_wrap_thresholds']
            if not ver_list_float('ga_shrink_wrap_thresholds', ga_shrink_wrap_thresholds):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Gashrinkwrapgausssigmas'
        if 'ga_shrink_wrap_gauss_sigmas' in config_map:
            ga_shrink_wrap_gauss_sigmas = config_map['ga_shrink_wrap_gauss_sigmas']
            if not ver_list_float('ga_shrink_wrap_gauss_sigmas', ga_shrink_wrap_gauss_sigmas):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Galowpassfiltersigmas'
        if 'ga_lowpass_filter_sigmas' in config_map:
            ga_lowpass_filter_sigmas = config_map['ga_lowpass_filter_sigmas']
            if not ver_list_float('ga_lowpass_filter_sigmas', ga_lowpass_filter_sigmas):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                return (error_message)

        config_parameter = 'Gagenpcstart'
        if 'ga_gen_pc_start' in config_map:
            ga_gen_pc_start = config_map['ga_gen_pc_start']
            if type(ga_gen_pc_start) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Twintrigger'
    if 'twin_trigger' in config_map:
        twin_trigger = config_map['twin_trigger']
        if not ver_list_int('twin_trigger', twin_trigger):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Twinhalves'
        if 'twin_halves' in config_map:
            twin_halves = config_map['twin_halves']
            if not ver_list_int('twin_halves', twin_halves):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Shrinkwraptrigger'
    if 'shrink_wrap_trigger' in config_map:
        if not ver_list_int('shrink_wrap_trigger', config_map['shrink_wrap_trigger']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Shrinkwraptype'
        if 'shrink_wrap_type' in config_map:
            shrink_wrap_type = config_map['shrink_wrap_type']
            if type(shrink_wrap_type) != str:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print (error_message)
                return (error_message)
            if shrink_wrap_type != "GAUSS":
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print (error_message)
                return (error_message)

        config_parameter = 'Shrinkwrapthreshold'
        if 'shrink_wrap_threshold' in config_map:
            shrink_wrap_threshold = config_map['shrink_wrap_threshold']
            if type(shrink_wrap_threshold) != float:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Shrinkwrapgausssigma'
        if 'shrink_wrap_gauss_sigma' in config_map:
            shrink_wrap_gauss_sigma = config_map['shrink_wrap_gauss_sigma']
            if type(shrink_wrap_gauss_sigma) != float and type(shrink_wrap_gauss_sigma) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Phasesupporttrigger'
    if 'phase_support_trigger' in config_map:
        if not ver_list_int('phase_support_trigger', config_map['phase_support_trigger']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Phmphasemin'
        if 'phm_phase_min' in config_map:
            phm_phase_min = config_map['phm_phase_min']
            if type(phm_phase_min) != float:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Phmphasemax'
        if 'phm_phase_max' in config_map:
            phm_phase_max = config_map['phm_phase_max']
            if type(phm_phase_max) != float:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Pcinterval'
    if 'pc_interval' in config_map:
        pc_interval = config_map['pc_interval']
        if type(pc_interval) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Pctype'
        if 'pc_type' in config_map:
            pc_type = config_map['pc_type']
            if type(pc_type) != str:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
            if pc_type != "LUCY":
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Pclucyiterations'
        if 'pc_LUCY_iterations' in config_map:
            pc_LUCY_iterations = config_map['pc_LUCY_iterations']
            if type(pc_LUCY_iterations) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Pcnormalize'
        if 'pc_normalize' in config_map:
            pc_normalize = config_map['pc_normalize']
            if type(pc_normalize) != bool:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Pclucykernel'
        if 'pc_LUCY_kernel' in config_map:
            pc_LUCY_kernel = config_map['pc_LUCY_kernel']
            if not ver_list_int('pc_LUCY_kernel', pc_LUCY_kernel):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
        else:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Resolutiontrigger'
    if 'resolution_trigger' in config_map:
        if not ver_list_int('resolution_trigger', config_map['resolution_trigger']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Lowpassfilterswsigmarange'
        if 'lowpass_filter_sw_sigma_range' in config_map:
            lowpass_filter_sw_sigma_range = config_map['lowpass_filter_sw_sigma_range']
            if not ver_list_float('lowpass_filter_sw_sigma_range', lowpass_filter_sw_sigma_range):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Lowpassfilterrange'
        if 'lowpass_filter_range' in config_map:
            lowpass_filter_range = config_map['lowpass_filter_range']
            if not ver_list_float('lowpass_filter_range', lowpass_filter_range):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Averagetrigger'
    if 'average_trigger' in config_map:
        if not ver_list_int('average_trigger', config_map['average_trigger']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Progresstrigger'
    if 'progress_trigger' in config_map:
        if not ver_list_int('progress_trigger', config_map['progress_trigger']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    # return empty string if verified
    return ("")


def ver_config_data(config_map):
    """
    This function verifies experiment config_data file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_data_error_map_file'
    fname = 'config_data'

    config_parameter = 'Datadir'
    if 'data_dir' in config_map:
        data_dir = config_map['data_dir']
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Adjustdimensions'
    if 'adjust_dimensions' in config_map:
        if not ver_list_int('pad_crop', config_map['adjust_dimensions']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Centershift'
    if 'center_shift' in config_map:
        if not ver_list_int('center_shift', config_map['center_shift']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Binning'
    if 'binning' in config_map:
        if not ver_list_int('binning', config_map['binning']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Intensitythreshold'
    if 'intensity_threshold' in config_map:
        intensity_threshold = config_map['intensity_threshold']
        if type(intensity_threshold) != float and type(intensity_threshold) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
    else:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print(error_message)
        return (error_message)

    config_parameter = 'Alienalg'
    if 'alien_alg' in config_map:
        alien_alg = config_map['alien_alg']
        alien_alg_options = ['block_aliens', 'alien_file', 'AutoAlien1', 'none']
        if alien_alg not in alien_alg_options:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        elif alien_alg == 'block_aliens':
            config_parameter = 'Aliens'
            if 'aliens' in config_map:
                aliens = config_map['aliens']
                if issubclass(type(aliens), list):
                    for a in aliens:
                        if not issubclass(type(a), list):
                            config_error = 0
                            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                            print (error_message)
                            return (error_message)
                        if not ver_list_int('aliens', a):
                            config_error = 1
                            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                            print (error_message)
                            return (error_message)
                        if (len(a) < 6):
                            config_error = 2
                            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                            print(error_message)
                            return (error_message)
            else:
                config_error = 3
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
        elif alien_alg == 'alien_file':
            config_parameter = 'AlienFile'
            if 'alien_file' in config_map:
                alien_file = config_map['alien_file']
                if type(alien_file) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)
            else:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
        elif alien_alg == 'AutoAlien1':
            config_parameter = 'Aa1sizethreshold'
            if 'AA1_size_threshold' in config_map:
                AA1_size_threshold = config_map['AA1_size_threshold']
                if type(AA1_size_threshold) != float and type(AA1_size_threshold) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Aa1asymthreshold'
            if 'AA1_asym_threshold' in config_map:
                AA1_asym_threshold = config_map['AA1_asym_threshold']
                if type(AA1_asym_threshold) != float and type(AA1_asym_threshold) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Aa1minpts'
            if 'AA1_min_pts' in config_map:
                AA1_min_pts = config_map['AA1_min_pts']
                if type(AA1_min_pts) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Aa1eps'
            if 'AA1_eps' in config_map:
                AA1_eps = config_map['AA1_eps']
                if type(AA1_eps) != float and type(AA1_eps) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(AA1_size_threshold)
                    return (error_message)

            config_parameter = 'Aa1ampthreshold'
            if 'AA1_amp_threshold' in config_map:
                AA1_amp_threshold = config_map['AA1_amp_threshold']
                if type(AA1_amp_threshold) != float and type(AA1_amp_threshold) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(AA1_size_threshold)
                    return (error_message)

            config_parameter = 'Aa1savearrs'
            if 'AA1_save_arrs' in config_map:
                AA1_save_arrs = config_map['AA1_save_arrs']
                if type(AA1_save_arrs) != bool:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(AA1_size_threshold)
                    return (error_message)

            config_parameter = 'Aa1expandcleanedsigma'
            if 'AA1_expandcleanedsigma' in config_map:
                AA1_expandcleanedsigma = config_map['AA1_expandcleanedsigma']
                if type(AA1_expandcleanedsigma) != float and type(AA1_expandcleanedsigma) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(AA1_size_threshold)
                    return (error_message)

    # return empty string if verified
    return ("")


def ver_config_prep(config_map):
    """
    This function verifies experiment config_prep file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_prep_error_map_file'
    fname = 'config_prep'

    config_parameter = 'Roi'
    if 'roi' in config_map:
        if not ver_list_int('roi', config_map['roi']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print (error_message)
            return (error_message)

    config_parameter = 'Datadir'
    if 'data_dir' in config_map:
        data_dir = config_map['data_dir']
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return error_message
    else:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print (error_message)
        return (error_message)

    config_parameter = 'Darkfield'
    if 'darkfield_filename' in config_map:
        darkfield_filename = config_map['darkfield_filename']
        if type(darkfield_filename) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Whitefield'
    if 'whitefield_filename' in config_map:
        whitefield_filename = config_map['whitefield_filename']
        if type(whitefield_filename) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Excludescans'
    if 'exclude_scans' in config_map:
        if not ver_list_int('exclude_scans', config_map['exclude_scans']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'MinFiles'
    if 'min_files' in config_map:
        min_files = config_map['min_files']
        if type(min_files) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Excludescans'
    if 'exclude_scans' in config_map:
        if not ver_list_int('exclude_scans', config_map['exclude_scans']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print (error_message)
            return (error_message)

    config_parameter = 'Separatescans'
    if 'separate_scans' in config_map:
        separate_scans = config_map['separate_scans']
        if type(separate_scans) != bool:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Separatescanranges'
    if 'separate_scan_ranges' in config_map:
        separate_scan_ranges = config_map['separate_scan_ranges']
        if type(separate_scan_ranges) != bool:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    return ("")


def ver_config_disp(config_map):
    """
    This function verifies experiment config_disp file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_disp_error_map_file'
    fname = 'config_disp'

    config_parameter = 'Resultsdir'
    if 'results_dir' in config_map:
        results_dir = config_map['results_dir']
        if type(results_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('results_dir parameter should be string')
            return (error_message)

    config_parameter = 'Diffractometer'
    if 'diffractometer' in config_map:
        diffractometer = config_map['diffractometer']
        if type(diffractometer) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('diffractometer parameter should be string')
            return (error_message)

    config_parameter = 'Detector'
    if 'detector' in config_map:
        detector = config_map['detector']
        if type(detector) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('detector parameter should be string')
            return (error_message)

    config_parameter = 'Crop'
    if 'crop' in config_map:
        crop = config_map['crop']
        if not issubclass(type(crop), list):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('crop should be list')
            return (error_message)
        for e in crop:
            if type(e) != int and type(e) != float:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('crop should be a list of int or float')
                return (error_message)

    config_parameter = 'Rampups'
    if 'rampups' in config_map:
        rampups = config_map['rampups']
        if type(rampups) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('rampups should be float')
            return (error_message)

    config_parameter = 'Energy'
    if 'energy' in config_map:
        energy = config_map['energy']
        if type(energy) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('energy should be float')
            return (error_message)

    config_parameter = 'Delta'
    if 'delta' in config_map:
        delta = config_map['delta']
        if type(delta) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('delta should be float')
            return (error_message)

    config_parameter = 'Gamma'
    if 'gamma' in config_map:
        gamma = config_map['gamma']
        if type(gamma) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('gamma should be float')
            return (error_message)

    config_parameter = 'Detdist'
    if 'detdist' in config_map:
        detdist = config_map['detdist']
        if type(detdist) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('detdist should be float')
            return (error_message)

    config_parameter = 'Dth'
    if 'dth' in config_map:
        dth = config_map['dth']
        if type(dth) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('dth should be float')
            return (error_message)

    return ("")


def verify(file_name, conf_map):
    """
    Verifies parameters.

    Parameters
    ----------
    file_name : str
        name of file the parameters are related to. Supported: config_prep, config_data, config_rec, config_disp

    conf_map : dict
        parameters

    Returns
    -------
    str
        a message with description of error or empty string if no error
    """
    if file_name == 'config':
        return ver_config(conf_map)
    elif file_name == 'config_prep':
        return ver_config_prep(conf_map)
    elif file_name == 'config_data':
        return ver_config_data(conf_map)
    elif file_name == 'config_rec':
        return ver_config_rec(conf_map)
    elif file_name == 'config_disp':
        return ver_config_disp(conf_map)
    else:
        return ('verifier has no fumction to check config file named', file_name)
