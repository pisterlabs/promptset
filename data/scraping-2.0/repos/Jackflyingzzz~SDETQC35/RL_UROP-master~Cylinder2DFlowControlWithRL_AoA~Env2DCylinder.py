# coding=utf-8
"""
Environment seen by the RL agent. It is the main class of the repo.
"""

#from printind.printind_decorators import printi_all_method_calls as printidc
#from printind.printind_function import printi, printiv
from tensorforce.environments import Environment
import tensorforce
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from threading import Thread
from tensorforce import TensorforceError

# a bit hacky, but meeehh... FIXME!!
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../Simulation/")

from dolfin import Expression, File, plot
from probes import PenetratedDragProbeANN, PenetratedLiftProbeANN, PressureProbeANN, VelocityProbeANN, RecirculationAreaProbe
from generate_msh import generate_mesh
from flow_solver import FlowSolver
from msh_convert import convert
from dolfin import *

import numpy as np
import os

import pickle

import time
import math
import csv

import shutil


# TODO: check that right types etc from tensorfoce examples
# typically:

# from tensorforce.contrib.openai_gym import OpenAIGym
# environment = OpenAIGym('MountainCarContinuous-v0', visualize=False)

# printiv(environment.states)
# environment.states = {'shape': (2,), 'type': 'float'}
# printiv(environment.actions)
# environment.actions = {'max_value': 1.0, 'shape': (1,), 'min_value': -1.0, 'type': 'float'}

class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')  # Initialise ring array 'data' as length-array of floats
        self.index = 0  # Initialise InPointer as 0 (where new data begins to be written)

    def extend(self, x):
        "adds array x to ring buffer"
        x_indices = (self.index + np.arange(x.size)) % self.data.size  # Find indices that x will occupy in 'data' array
        self.data[x_indices] = x  # Input the new array into ring buffer ('data')
        self.index = x_indices[-1] + 1  # Find new index for next new data

    def get(self):
        "Returns the first-in-first-out data in the ring buffer (returns data in order of introduction)"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]


# @printidc()
class Env2DCylinder(Environment):
    """Environment for 2D flow simulation around a cylinder."""

    def __init__(self, path_root, geometry_params, flow_params, solver_params, output_params,
                 optimization_params, inspection_params, n_iter_make_ready=None, verbose=0, size_history=2000,
                 reward_function='plain_drag', size_time_state=50, number_steps_execution=1, simu_name="Simu"):
        """

        """

        # TODO: should actually save the dicts in to double check when loading that using compatible simulations together

        #printi("--- call init ---")

        self.observation = None
        self.thread = None

        self.path_root = path_root
        self.flow_params = flow_params
        self.geometry_params = geometry_params
        self.solver_params = solver_params
        self.output_params = output_params
        self.optimization_params = optimization_params
        self.inspection_params = inspection_params
        self.size_time_state = size_time_state
        self.verbose = verbose
        self.n_iter_make_ready = n_iter_make_ready
        self.size_history = size_history
        self.reward_function = reward_function
        self.number_steps_execution = number_steps_execution

        self.simu_name = simu_name


        # If previous output.csv (epidosde avgs) exists, obtain its last row to get last simulated episode number
        name="output.csv"
        last_row = None
        if(os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, 'r') as f:
                for row in reversed(list(csv.reader(f, delimiter=";", lineterminator="\n"))):
                    last_row = row
                    break
        if(not last_row is None):
            self.episode_number = int(last_row[0])  # Current episode. Updates once its avgs have been recorded
            self.last_episode_number = int(last_row[0])  # Current episode. Updates when a new episode starts (upon reset)
        else:
            self.last_episode_number = 0
            self.episode_number = 0

        # Initialise arrays that wil, store episode data
        self.episode_drags = np.array([])
        self.episode_areas = np.array([])
        self.episode_lifts = np.array([])

        self.initialized_visualization = False

        self.start_class()

        print("--- done buffers initialisation ---")

    def start_class(self):
        '''
        Initialise attributes needed by the Environment object:
        Initialise history buffers, remesh if necessary, load initialization quantities if not remesh, create flow solver
        object, initialise probes, make converged flow if necessary, simulate to random position if random_start,
        fill probes buffer if no. probes changed wrt last remesh
        :return:
        '''
        self.solver_step = 0  # Numerical step
        self.accumulated_drag = 0  # Sum of step drags so far
        self.accumulated_lift = 0

        self.initialized_vtu= False

        self.resetted_number_probes = False

        self.area_probe = None

       # Create ring buffers to hold recent history of jet values, probe values, lift, drag, area
        self.history_parameters = {}

        for crrt_jet in range(2):
            self.history_parameters["jet_{}".format(crrt_jet)] = RingBuffer(self.size_history)

        self.history_parameters["number_of_jets"] = 2

        for crrt_probe in range(len(self.output_params["locations"])):
            if self.output_params["probe_type"] == 'pressure':
                self.history_parameters["probe_{}".format(crrt_probe)] = RingBuffer(self.size_history)
            elif self.output_params["probe_type"] == 'velocity':
                self.history_parameters["probe_{}_u".format(crrt_probe)] = RingBuffer(self.size_history)
                self.history_parameters["probe_{}_v".format(crrt_probe)] = RingBuffer(self.size_history)

        self.history_parameters["number_of_probes"] = len(self.output_params["locations"])
        print("Number of probes: {}".format(self.history_parameters["number_of_probes"]))

        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)
        self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

        # ------------------------------------------------------------------------
        # Remesh if necessary
        h5_file = '.'.join([self.path_root, 'h5'])  # mesh/turek_2d.h5
        msh_file = '.'.join([self.path_root, 'msh'])  # mesh/turek_2d.msh
        self.geometry_params['mesh'] = h5_file

        # Regenerate mesh?
        if self.geometry_params['remesh']:

            if self.verbose > 0:
                print("Remesh")

            generate_mesh(self.geometry_params, template=self.geometry_params['template'])

            if self.verbose > 0:
                print("Generate .msh done")
            print(msh_file)
            assert os.path.exists(msh_file)

            convert(msh_file, h5_file)
            if self.verbose > 0:
                print("Convert to .h5 done")
            print(h5_file)
            assert os.path.exists(h5_file)

        # ------------------------------------------------------------------------
        # If no remesh, load initialization fields and buffers

        if self.n_iter_make_ready is None:
            if self.verbose > 0:
                print("Load initial flow state")

            # Load initial fields
            self.flow_params['u_init'] = 'mesh/u_init.xdmf'
            self.flow_params['p_init'] = 'mesh/p_init.xdmf'

            if self.verbose > 0:
                print("Load buffer history")

            # Load ring buffers
            with open('mesh/dict_history_parameters.pkl', 'rb') as f:
                self.history_parameters = pickle.load(f)

            # Check everything is good to go

            if not "number_of_probes" in self.history_parameters:
                self.history_parameters["number_of_probes"] = 0
                print("Warning!! The number of probes was not set in the loaded hdf5 file")

            if not "number_of_jets" in self.history_parameters:
                self.history_parameters["number_of_jets"] = 2
                print("Warning!! The number of jets was not set in the loaded hdf5 file")

            if not "lift" in self.history_parameters:
                self.history_parameters["lift"] = RingBuffer(self.size_history)
                print("Warning!! No lift history found in the loaded hdf5 file")

            if not "drag" in self.history_parameters:
                self.history_parameters["drag"] = RingBuffer(self.size_history)
                print("Warning!! No lift history found in the loaded hdf5 file")

            if not "recirc_area" in self.history_parameters:
                self.history_parameters["recirc_area"] = RingBuffer(self.size_history)
                print("Warning!! No recirculation area history found in the loaded hdf5 file")

            # if not the same number of probes, reset
            if not self.history_parameters["number_of_probes"] == len(self.output_params["locations"]):
                for crrt_probe in range(len(self.output_params["locations"])):
                    if self.output_params["probe_type"] == 'pressure':
                        self.history_parameters["probe_{}".format(crrt_probe)] = RingBuffer(self.size_history)
                    elif self.output_params["probe_type"] == 'velocity':
                        self.history_parameters["probe_{}_u".format(crrt_probe)] = RingBuffer(self.size_history)
                        self.history_parameters["probe_{}_v".format(crrt_probe)] = RingBuffer(self.size_history)

                self.history_parameters["number_of_probes"] = len(self.output_params["locations"])

                print("Warning!! Number of probes was changed! Probes buffer content reset")

                self.resetted_number_probes = True

        # ------------------------------------------------------------------------
        # create the flow simulation object
        self.flow = FlowSolver(self.flow_params, self.geometry_params, self.solver_params)

        # ------------------------------------------------------------------------
        # Setup probes
        if self.output_params["probe_type"] == 'pressure':
            self.ann_probes = PressureProbeANN(self.flow, self.output_params['locations'])

        elif self.output_params["probe_type"] == 'velocity':
            self.ann_probes = VelocityProbeANN(self.flow, self.output_params['locations'])
        else:
            raise RuntimeError("Unknown probe type")

        # Setup drag and lift measurement
        self.drag_probe = PenetratedDragProbeANN(self.flow)
        self.lift_probe = PenetratedLiftProbeANN(self.flow)

        # ------------------------------------------------------------------------
        # No flux from jets for starting
        self.Qs = np.zeros(2)
        self.action = np.zeros(2)

        # ------------------------------------------------------------------------
        # prepare the arrays for plotting positions
        self.compute_positions_for_plotting()


        # ------------------------------------------------------------------------
        # if necessary, make converge
        if self.n_iter_make_ready is not None:
            self.u_, self.p_ = self.flow.evolve(self.Qs)
            path=''
            if "dump_vtu" in self.inspection_params:
                path = 'results/area_out.pvd'
            self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)
            if self.verbose > 0:
                print("Compute initial flow for {} steps".format(self.n_iter_make_ready))

            for _ in range(self.n_iter_make_ready):
                self.u_, self.p_ = self.flow.evolve(self.Qs)

                self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
                self.drag = self.drag_probe.sample(self.u_, self.p_)
                self.lift = self.lift_probe.sample(self.u_, self.p_)
                self.recirc_area = self.area_probe.sample(self.u_, self.p_)

                self.write_history_parameters()  # Add new step data to history buffers
                self.visual_inspection()  # Create dynamic plots, show step data in command line and save it to saved_models/debug.csv
                self.output_data()  # Extend arrays of episode qtys, generate vtu files for area, u and p

                self.solver_step += 1

            encoding = XDMFFile.Encoding.HDF5
            mesh = convert(msh_file, h5_file)
            comm = mesh.mpi_comm()

            # save field data
            XDMFFile(comm, 'mesh/u_init.xdmf').write_checkpoint(self.u_, 'u0', 0, encoding)
            XDMFFile(comm, 'mesh/p_init.xdmf').write_checkpoint(self.p_, 'p0', 0, encoding)

            # save buffer dict
            with open('mesh/dict_history_parameters.pkl', 'wb') as f:
                pickle.dump(self.history_parameters, f, pickle.HIGHEST_PROTOCOL)

        # ----------------------------------------------------------------------
        # if reading from disk (no remesh), show to check everything ok
        if self.n_iter_make_ready is None:
            # If random_start == True, let's start in a random position of the vortex shedding
            if self.optimization_params["random_start"]:
                rd_advancement = np.random.randint(1712)  # 1712 is the number of steps of a shedding period. Needs to be hardcoded
                for j in range(rd_advancement):
                    self.flow.evolve(self.Qs)
                print("Simulated {} iterations before starting the control".format(rd_advancement))

            self.u_, self.p_ = self.flow.evolve(self.Qs)  # Initial step with Qs = 0
            path=''
            if "dump_vtu" in self.inspection_params:
                path = 'results/area_out.pvd'
            self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)

            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
            self.drag = self.drag_probe.sample(self.u_, self.p_)
            self.lift = self.lift_probe.sample(self.u_, self.p_)
            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            self.write_history_parameters()

        # ----------------------------------------------------------------------
        # if necessary, fill the probes buffer
        if self.resetted_number_probes:
            print("Need to fill again the buffer; modified number of probes")

            # TODO: Execute runs for 1 action step, so we would be running for (T/Nb)*size_history. While it should be just for size_history.  In practice, remesh if probes change
            for _ in range(self.size_history):
                self.execute()

        # ----------------------------------------------------------------------
        # ready now

        self.ready_to_use = True

    def write_history_parameters(self):
        '''
        Add data of last step to history buffers
        :return:
        '''
        for crrt_jet in range(2):
            self.history_parameters["jet_{}".format(crrt_jet)].extend(self.Qs[crrt_jet])

        if self.output_params["probe_type"] == 'pressure':
            for crrt_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}".format(crrt_probe)].extend(self.probes_values[crrt_probe])
        elif self.output_params["probe_type"] == 'velocity':
            for crrt_probe in range(len(self.output_params["locations"])):
                self.history_parameters["probe_{}_u".format(crrt_probe)].extend(self.probes_values[2 * crrt_probe])  # probes values are ordered in u,v pairs
                self.history_parameters["probe_{}_v".format(crrt_probe)].extend(self.probes_values[2 * crrt_probe + 1])

        self.history_parameters["drag"].extend(np.array(self.drag))
        self.history_parameters["lift"].extend(np.array(self.lift))
        self.history_parameters["recirc_area"].extend(np.array(self.recirc_area))

    def compute_positions_for_plotting(self):
        '''
        Obtain the coordinates of the probes and the jets for plotting.
        '''

        ## where the pressure probes are
        self.list_positions_probes_x = []
        self.list_positions_probes_y = []

        # get the coordinates of probe positions
        for crrt_probe in self.output_params['locations']:
            if self.verbose > 2:
                print(crrt_probe)
            self.list_positions_probes_x.append(crrt_probe[0])
            self.list_positions_probes_y.append(crrt_probe[1])

        ## where the jets are
        self.list_positions_jets_x = []
        self.list_positions_jets_y = []

        height_cylinder = self.geometry_params['height_cylinder']
        length_cylinder = height_cylinder * self.geometry_params['ar']
        jet_width = self.geometry_params['jet_width']

        # get the coordinates of jet positions
        crrt_x = length_cylinder / 2 - jet_width / 2
        for jet in range(2):
            crrt_y = height_cylinder / 2 - jet * height_cylinder
            self.list_positions_jets_x.append(crrt_x)
            self.list_positions_jets_y.append(1.1 * crrt_y)

    def show_flow(self):

        height_cylinder = self.geometry_params['height_cylinder']
        length_cylinder = height_cylinder * self.geometry_params['ar']

        plt.figure()
        plot(self.u_)
        plt.scatter(self.list_positions_probes_x, self.list_positions_probes_y, c='k', marker='o')
        plt.scatter(self.list_positions_jets_x, self.list_positions_jets_y, c='r', marker='o')
        plt.xlim([-length_cylinder / 2 - self.geometry_params['x_upstream'],
                  length_cylinder / 2 + self.geometry_params['x_downstream']])
        plt.ylim([-self.geometry_params['height_domain'] / 2 - self.geometry_params['cylinder_y_shift'],
                  self.geometry_params['height_domain'] / 2 + self.geometry_params['cylinder_y_shift']])
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.show()

        plt.figure()
        p = plot(self.p_)
        cb = plt.colorbar(p, fraction=0.1, shrink=0.3)
        plt.scatter(self.list_positions_probes_x, self.list_positions_probes_y, c='k', marker='o')
        plt.scatter(self.list_positions_jets_x, self.list_positions_jets_y, c='r', marker='o')
        plt.xlim([-length_cylinder / 2 - self.geometry_params['x_upstream'],
                  length_cylinder / 2 + self.geometry_params['x_downstream']])
        plt.ylim([-self.geometry_params['height_domain'] / 2 - self.geometry_params['cylinder_y_shift'],
                  self.geometry_params['height_domain'] / 2 + self.geometry_params['cylinder_y_shift']])
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.tight_layout()
        cb.set_label("P")
        plt.show()

    def show_control(self):
        plt.figure()

        linestyles = ['-', '--', ':', '-.']

        for crrt_jet in range(2):
            crrt_jet_data = self.history_parameters["jet_{}".format(crrt_jet)].get()
            plt.plot(crrt_jet_data, label="jet {}".format(crrt_jet), linestyle=linestyles[crrt_jet], linewidth=1.5)
        plt.legend(loc=2)
        plt.ylabel("Control Q")
        plt.xlabel("Actuation step")
        plt.tight_layout()
        plt.pause(1.0)
        plt.savefig("saved_figures/control_episode_{}.pdf".format(self.episode_number))
        plt.show()
        plt.pause(2.0)

    def show_drag(self):
        plt.figure()
        crrt_drag = self.history_parameters["drag"].get()
        plt.plot(crrt_drag, label="episode drag", linewidth=1.2)
        plt.plot([0, self.size_history - 1], [self.inspection_params['line_drag'], self.inspection_params['line_drag']],
                 label="mean drag no control", linewidth=2.5, linestyle="--")
        plt.ylabel("measured drag D")
        plt.xlabel("actuation step")
        range_drag_plot = self.inspection_params["range_drag_plot"]
        plt.legend(loc=2)
        plt.ylim(range_drag_plot)
        plt.tight_layout()
        plt.pause(1.0)
        plt.savefig("saved_figures/drag_episode_{}.pdf".format(self.episode_number))
        plt.show()
        plt.pause(2.0)

    def visual_inspection(self):
        '''
        Create dynamic plots, show step data in command line and save it to saved_models/debug.csv (or to
        saved_models/test_strategy.csv if single run)
        '''
        total_number_subplots = 5
        crrt_subplot = 1

        # If plot =! False, initialise dynamic plotting if not done before
        if(not self.initialized_visualization and self.inspection_params["plot"] != False):
            plt.ion()  # Turn the interactive plot mode on
            plt.subplots(total_number_subplots, 1)
            # ax.set_xlim([0, self.nbr_points_animate_plot])
            # ax.set_ylim([0, 1024])
            print("Dynamic plotting turned on")

            self.initialized_visualization = True

        if("plot" in self.inspection_params and self.inspection_params["plot"] != False):
            modulo_base = self.inspection_params["plot"]

            if self.solver_step % modulo_base == 0:

                # Plot velocity field
                height_cylinder = self.geometry_params['height_cylinder']
                length_cylinder = height_cylinder * self.geometry_params['ar']

                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plot(self.u_)
                plt.scatter(self.list_positions_probes_x, self.list_positions_probes_y, c='k', marker='o')
                plt.scatter(self.list_positions_jets_x, self.list_positions_jets_y, c='r', marker='o')
                plt.xlim([-length_cylinder / 2 - self.geometry_params['x_upstream'],
                          length_cylinder / 2 + self.geometry_params['x_downstream']])
                plt.ylim([-self.geometry_params['height_domain'] / 2 - self.geometry_params['cylinder_y_shift'],
                          self.geometry_params['height_domain'] / 2 + self.geometry_params['cylinder_y_shift']])
                plt.ylabel("V")
                crrt_subplot += 1

                # Plot pressure field
                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plot(self.p_)
                plt.scatter(self.list_positions_probes_x, self.list_positions_probes_y, c='k', marker='o')
                plt.scatter(self.list_positions_jets_x, self.list_positions_jets_y, c='r', marker='o')
                plt.xlim([-length_cylinder / 2 - self.geometry_params['x_upstream'],
                          length_cylinder / 2 + self.geometry_params['x_downstream']])
                plt.ylim([-self.geometry_params['height_domain'] / 2 - self.geometry_params['cylinder_y_shift'],
                          self.geometry_params['height_domain'] / 2 + self.geometry_params['cylinder_y_shift']])
                plt.ylabel("P")
                crrt_subplot += 1

                # Plot jets MFR
                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plt.cla()
                for crrt_jet in range(2):
                    crrt_jet_data = self.history_parameters["jet_{}".format(crrt_jet)].get()
                    plt.plot(crrt_jet_data, label="jet {}".format(crrt_jet))
                plt.legend(loc=6)
                plt.ylabel("M.F.R.")
                crrt_subplot += 1

                # plt.subplot(total_number_subplots, 1, crrt_subplot)
                # plt.cla()
                # for crrt_probe in range(len(self.output_params["locations"])):
                #     if self.output_params["probe_type"] == 'pressure':
                #         crrt_probe_data = self.history_parameters["probe_{}".format(crrt_probe)].get()
                #         plt.plot(crrt_probe_data, label="probe {}".format(crrt_probe))
                #     elif self.output_params["probe_type"] == 'velocity':
                #         crrt_probe_data = self.history_parameters["probe_{}_u".format(crrt_probe)].get()
                #         plt.plot(crrt_probe_data, label="probe {}".format(crrt_probe))
                #         crrt_probe_data = self.history_parameters["probe_{}_v".format(crrt_probe)].get()
                #         plt.plot(crrt_probe_data, label="probe {}".format(crrt_probe))
                # # plt.legend(loc=6)
                # if self.output_params["probe_type"] == "pressure":
                #     plt.ylabel("pressure")
                # elif self.output_params["probe_type"] == "velocity":
                #     plt.ylabel("velocity")
                # if "range_pressure_plot" in self.inspection_params:
                #     range_pressure_plot = self.inspection_params["range_pressure_plot"]
                #     plt.ylim(range_pressure_plot)
                # crrt_subplot += 1


                # Plot lift and drag recent history
                plt.subplot(total_number_subplots, 1, crrt_subplot)
                ax1 = plt.gca()
                plt.cla()

                crrt_drag = self.history_parameters["drag"].get()

                ax1.plot(crrt_drag, color='r', linestyle='-')
                if 'line_drag' in self.inspection_params:
                    ax1.plot([0, self.size_history - 1],
                             [self.inspection_params['line_drag'], self.inspection_params['line_drag']],
                             color='r',
                             linestyle='--')

                ax1.set_ylabel("drag")
                if "range_drag_plot" in self.inspection_params:
                    range_drag_plot = self.inspection_params["range_drag_plot"]
                    ax1.set_ylim(range_drag_plot)

                ax2 = ax1.twinx()

                crrt_lift = self.history_parameters["lift"].get()

                ax2.plot(crrt_lift, color='b', linestyle='-', label="lift")
                if 'line_lift' in self.inspection_params:
                    ax2.plot([0, self.size_history - 1],
                             [self.inspection_params['line_lift'], self.inspection_params['line_lift']],
                             color='b',
                             linestyle='--')

                ax2.set_ylabel("lift")
                if "range_lift_plot" in self.inspection_params:
                    range_lift_plot = self.inspection_params["range_lift_plot"]
                    ax2.set_ylim(range_lift_plot)

                plt.xlabel("buffer steps")

                crrt_subplot += 1

                # Plot recirculation area recent history
                plt.subplot(total_number_subplots, 1, crrt_subplot)
                plt.cla()
                crrt_area = self.history_parameters["recirc_area"].get()
                plt.plot(crrt_area)
                plt.ylabel("RecArea")
                plt.xlabel("buffer steps")
                #if "range_drag_plot" in self.inspection_params:
                #    range_drag_plot = self.inspection_params["range_drag_plot"]
                plt.ylim([0, 0.03])
                crrt_subplot += 1

                # plt.tight_layout()
                plt.tight_layout(pad=0, w_pad=0, h_pad=-0.5)
                plt.draw()
                plt.pause(0.5)

        if (self.inspection_params["dump_CL"] != False and self.solver_step % self.inspection_params["dump_CL"] == 0 and self.inspection_params["dump_CL"] < 10000):
            # Display information in command line
            print("%s | Ep N: %4d, step: %4d, Rec Area: %.4f, drag: %.4f, lift: %.4f, jet_0: %.4f, jet_1: %.4f"%(self.simu_name,
            self.episode_number,
            self.solver_step,
            self.history_parameters["recirc_area"].get()[-1],
            self.history_parameters["drag"].get()[-1],
            self.history_parameters["lift"].get()[-1],
            self.history_parameters["jet_0"].get()[-1],
            self.history_parameters["jet_1"].get()[-1]))

        if (self.inspection_params["dump_debug"] != False and self.solver_step % self.inspection_params["dump_debug"] == 0 and self.inspection_params["dump_debug"] < 10000):
            # Save everything that happens in a debug.csv file!
            name = "debug.csv"
            if(not os.path.exists("saved_models")):
                os.mkdir("saved_models")
            if(not os.path.exists("saved_models/"+name)):
                with open("saved_models/"+name, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Name", "Episode", "Step", "RecircArea", "Drag", "Lift", "Jet0", "Jet1"])
                    spam_writer.writerow([self.simu_name,
                                          self.episode_number,
                                          self.solver_step,
                                          self.history_parameters["recirc_area"].get()[-1],
                                          self.history_parameters["drag"].get()[-1],
                                          self.history_parameters["lift"].get()[-1],
                                          self.history_parameters["jet_0"].get()[-1],
                                          self.history_parameters["jet_1"].get()[-1]])
            else:
                with open("saved_models/"+name, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.simu_name,
                                          self.episode_number,
                                          self.solver_step,
                                          self.history_parameters["recirc_area"].get()[-1],
                                          self.history_parameters["drag"].get()[-1],
                                          self.history_parameters["lift"].get()[-1],
                                          self.history_parameters["jet_0"].get()[-1],
                                          self.history_parameters["jet_1"].get()[-1]])

        if("single_run" in self.inspection_params and self.inspection_params["single_run"] == True):
            # if ("dump" in self.inspection_params and self.inspection_params["dump"] > 10000):
                self.sing_run_output()

    def sing_run_output(self):
        '''
        Perform output for single runs (testing of strategies or baseline flow)
        '''
        name = "test_strategy.csv"
        if(not os.path.exists("saved_models")):
            os.mkdir("saved_models")
        if(not os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, "w") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Name", "Step", "Drag", "Lift", "RecircArea"] + ["Jet" + str(v) for v in range(len(self.Qs))])
                spam_writer.writerow([self.simu_name, self.solver_step, self.history_parameters["drag"].get()[-1], self.history_parameters["lift"].get()[-1], self.history_parameters["recirc_area"].get()[-1]] + [str(v) for v in self.Qs.tolist()])
        else:
            with open("saved_models/"+name, "a") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.simu_name, self.solver_step, self.history_parameters["drag"].get()[-1], self.history_parameters["lift"].get()[-1], self.history_parameters["recirc_area"].get()[-1]] + [str(v) for v in self.Qs.tolist()])
        return

    def output_data(self):
        '''
        Extend arrays of episode drag,lift and recirculation
        If episode just ended, record avgs into saved_models/output.csv and empty episode lists
        Update best_model if improved
        Generate vtu files for area, u and p
        '''

        # Extend drag, lift and area histories
        self.episode_drags = np.append(self.episode_drags, [self.history_parameters["drag"].get()[-1]])
        self.episode_areas = np.append(self.episode_areas, [self.history_parameters["recirc_area"].get()[-1]])
        self.episode_lifts = np.append(self.episode_lifts, [self.history_parameters["lift"].get()[-1]])

        # If new episode (not single run), record avg qtys for last ep
        if(self.last_episode_number != self.episode_number and "single_run" in self.inspection_params and self.inspection_params["single_run"] == False):
            self.last_episode_number = self.episode_number  # Update last_episode_number, as no we will record the avgs for the last episode
            # Find avgs for the 2nd half of each ep
            avg_drag = np.average(self.episode_drags[len(self.episode_drags)//2:])
            avg_area = np.average(self.episode_areas[len(self.episode_areas)//2:])
            avg_lift = np.average(self.episode_lifts[len(self.episode_lifts)//2:])

            name = "output.csv"
            if(not os.path.exists("saved_models")):
                os.mkdir("saved_models")
            if(not os.path.exists("saved_models/"+name)):
                with open("saved_models/"+name, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Episode", "AvgDrag", "AvgLift", "AvgRecircArea"])
                    spam_writer.writerow([self.last_episode_number, avg_drag, avg_lift, avg_area])
            else:
                with open("saved_models/"+name, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.last_episode_number, avg_drag, avg_lift, avg_area])

            # Also write in Cylinder2DFlowControlWithRL folder (useful to have data of all episodes together in parallel runs)
            if(not os.path.exists("../episode_averages")):
                os.mkdir("../episode_averages")
            if(not os.path.exists("../episode_averages/"+name)):
                with open("../episode_averages/"+name, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Episode", "AvgDrag", "AvgLift", "AvgRecircArea"])
                    spam_writer.writerow([self.last_episode_number, avg_drag, avg_lift, avg_area])
            else:
                with open("../episode_averages/"+name, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.last_episode_number, avg_drag, avg_lift, avg_area])

            # Empty the episode lists for the new episode
            self.episode_drags = np.array([])
            self.episode_areas = np.array([])
            self.episode_lifts = np.array([])

            if(os.path.exists("saved_models/output.csv")):
                if(not os.path.exists("best_model")):
                    shutil.copytree("saved_models", "best_model")

                else :
                    with open("saved_models/output.csv", 'r') as csvfile:
                        data = csv.reader(csvfile, delimiter = ';')
                        for row in data:
                            lastrow = row
                        last_iter_drag = lastrow[1]  # get avg drag of last episode

                    with open("best_model/output.csv", 'r') as csvfile:
                        data = csv.reader(csvfile, delimiter = ';')
                        for row in data:
                            lastrow = row
                        best_iter_drag = lastrow[1]  # get avg drag of best episode in best_model/

                    # If last episode model is better than current best model, we replace the best model
                    if float(best_iter_drag) < float(last_iter_drag):
                        print("best_model updated")
                        if(os.path.exists("best_model")):
                            shutil.rmtree("best_model")
                        shutil.copytree("saved_models", "best_model")

        if "dump_vtu" in self.inspection_params and self.inspection_params["dump_vtu"] < 10000 and self.solver_step % self.inspection_params["dump_vtu"] == 0:

            if not self.initialized_vtu:  # Initialize results .pvd output if not done already
                self.u_out = File('results/u_out.pvd')
                self.p_out = File('results/p_out.pvd')
                self.initialized_vtu = True

            # Generate vtu files for area, drag and lift
            if(not self.area_probe is None):
                self.area_probe.dump(self.area_probe)
            self.u_out << self.flow.u_
            self.p_out << self.flow.p_


    def __str__(self):
        # printi("Env2DCylinder ---")
        print('')

    def close(self):
        self.ready_to_use = False

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        if self.solver_step > 0:
            mean_accumulated_drag = self.accumulated_drag / self.solver_step
            mean_accumulated_lift = self.accumulated_lift / self.solver_step

            if self.verbose > -1:
                print("Mean accumulated drag on the whole episode: {}".format(mean_accumulated_drag))

        if self.inspection_params["show_all_at_reset"]:
            self.show_drag()
            self.show_control()

        self.start_class()

        next_state = np.transpose(np.array(self.probes_values))
        if self.verbose > 0:
            print(next_state)

        self.episode_number += 1

        return(next_state)

    def execute(self, actions=None):
        '''
        Run solver for one action step, until next RL env state (For the flow solver this means to run for number_steps_execution)
        In this interval, control is made continuous between actions and net MFR = 0 is imposed
        :param: actions
        :return: next state (probe values at end of action step)
                 terminal
                 reward (- CD (-) mean over action step + 0.2 * abs of CL mean over action step)
        '''
        action = actions

        if self.verbose > 1:
            print("--- Call execute ---")

        if action is None:
            if self.verbose > -1:
                print("Careful, no action given to execute(). By default, no jet (Q=0)!")

            nbr_jets = 2
            action = np.zeros((nbr_jets, ))

        if self.verbose > 2:
            print(action)
            
        self.previous_action = self.action  # Store previous action before overwritting action attribute
        self.action = action

        # Run for one action step (several -number_steps_execution- numerical timesteps keeping action=const but changing control)
        for crrt_control_nbr in range(self.number_steps_execution):

            # Try to enforce a continuous, smoother control
            if "smooth_control" in self.optimization_params:
                # Original approach used in the JFM paper
                # self.Qs += self.optimization_params["smooth_control"] * (np.array(action) - self.Qs)

                # Linear interpolation in the control
                self.Qs = np.array(self.previous_action) + (np.array(self.action) - np.array(self.previous_action)) * ((crrt_control_nbr + 1) / self.number_steps_execution)
            else:
                self.Qs = np.transpose(np.array(action))

            # Impose a zero net Qs
            if "zero_net_Qs" in self.optimization_params:
                if self.optimization_params["zero_net_Qs"]:
                    self.Qs = self.Qs - np.mean(self.Qs)

            # Evolve one numerical timestep forward
            self.u_, self.p_ = self.flow.evolve(self.Qs)

            # Output flow data when relevant
            self.visual_inspection()  # Create dynamic plots, show step data in command line and save it to saved_models/debug.csv
            self.output_data()  # Extend arrays of episode qtys, generate vtu files for area, u and p

            # We have done one solver step
            self.solver_step += 1

            # Sample probes and drag
            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
            self.drag = self.drag_probe.sample(self.u_, self.p_)
            self.lift = self.lift_probe.sample(self.u_, self.p_)
            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            # Write to the history buffers
            self.write_history_parameters()

            self.accumulated_drag += self.drag
            self.accumulated_lift += self.lift

        # TODO: the next_state may incorporate more information: maybe some time information?
        next_state = np.transpose(np.array(self.probes_values))

        if self.verbose > 2:
            print(next_state)

        terminal = False

        if self.verbose > 2:
            print(terminal)

        reward = self.compute_reward()

        if self.verbose > 2:
            print(reward)

        if self.verbose > 1:
            print("--- done execute ---")

        return(next_state, terminal, reward)

    def compute_reward(self):
        mean_drag_no_control = - self.inspection_params['line_drag']

        # NOTE: reward should be computed over the whole number of iterations in each execute loop
        if self.reward_function == 'plain_drag':  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.history_parameters["drag"].get()[-self.number_steps_execution:]
            return(np.mean(values_drag_in_last_execute) + mean_drag_no_control)
        elif(self.reward_function == 'recirculation_area'):
            return - self.area_probe.sample(self.u_, self.p_)
        elif(self.reward_function == 'max_recirculation_area'):
            return self.area_probe.sample(self.u_, self.p_)
        elif self.reward_function == 'drag':  # a bit dangerous, may be injecting some momentum
            return self.history_parameters["drag"].get()[-1] + mean_drag_no_control
        elif self.reward_function == 'drag_plain_lift':  # a bit dangerous, may be injecting some momentum
            avg_length = min(500, self.number_steps_execution)
            avg_drag = np.mean(self.history_parameters["drag"].get()[-avg_length:])
            avg_lift = np.mean(self.history_parameters["lift"].get()[-avg_length:])
            return avg_drag + mean_drag_no_control - 0.2 * abs(avg_lift)
        elif self.reward_function == 'max_plain_drag':  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.history_parameters["drag"].get()[-self.number_steps_execution:]
            return - (np.mean(values_drag_in_last_execute) + mean_drag_no_control)
        elif self.reward_function == 'drag_avg_abs_lift':  # a bit dangerous, may be injecting some momentum
            avg_length = min(500, self.number_steps_execution)
            avg_abs_lift = np.mean(np.absolute(self.history_parameters["lift"].get()[-avg_length:]))
            avg_drag = np.mean(self.history_parameters["drag"].get()[-avg_length:])
            return avg_drag + mean_drag_no_control - 0.2 * avg_abs_lift

        # TODO: implement some reward functions that take into account how much energy / momentum we inject into the flow

        else:
            raise RuntimeError("Reward function {} not yet implemented".format(self.reward_function))

    def states(self):
        '''
        Returns the state space specification.
        :return: dictionary of state descriptions with the following attributes:
        type ("bool" / "int" / "float") – state data type (required)
        shape (int > 0 / iter[int > 0]) – state shape (default: scalar)
        '''

        if self.output_params["probe_type"] == 'pressure':
            return dict(type='float',
                        shape=(len(self.output_params["locations"]) * self.optimization_params["num_steps_in_pressure_history"], )
                        )

        elif self.output_params["probe_type"] == 'velocity':
            return dict(type='float',
                        shape=(2 * len(self.output_params["locations"]) * self.optimization_params["num_steps_in_pressure_history"], )
                        )

    def actions(self):
        '''
        Returns the action space specification.
        :return: dictionary of action descriptions with the following attributes:
        type ("bool" / "int" / "float") – action data type (required)
        shape (int > 0 / iter[int > 0]) – action shape
        min_value/max_value (float) – minimum/maximum action value
        '''

        # NOTE: we could also have several levels of dict in dict, for example:
        # return { str(i): dict(continuous=True, min_value=0, max_value=1) for i in range(self.n + 1) }

        return dict(type='float',
                    shape=(2, ),
                    min_value=self.optimization_params["min_value_jet_MFR"],
                    max_value=self.optimization_params["max_value_jet_MFR"])

    def max_episode_timesteps(self):
        return None
