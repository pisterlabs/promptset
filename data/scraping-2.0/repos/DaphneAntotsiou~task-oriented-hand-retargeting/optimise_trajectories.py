__author__ = 'DafniAntotsiou'

import os
from pso import pso, particle2actuator
from functions import *
import mujoco_py as mp
from math import ceil
from mjviewerext import MjViewerExt
import glob
import argparse
from replay_trajectories import play


def argsparser():
    parser = argparse.ArgumentParser("Implementation of Task Oriented Hand Motion Retargeting")
    parser.add_argument('--model_path', help='path to model xml', type=str, default="model/MPL/MPL_Sphere_6.xml")
    parser.add_argument('--traj_path', help='path to the trajectory file or directory', default='trajectories')
    parser.add_argument('--out_dir', help='directory to save the output results', default='trajectories/result')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    boolean_flag(parser, 'play', default=False, help='playback the original and optimised trajectories')
    boolean_flag(parser, 'rot_scene', default=True, help='set if scene was rotated during HPE acquisition')

    return parser.parse_args()


def boolean_flag(parser, name, default=False, help=None):
    """ This function is from OpenAI's baselines.
    Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def optimise_actions(model_path, traj_path, rot_scene=False, fps=60, render=False, name=None, replay=False):
    per_hpe = False  # apply pso only on new hpe frame regardless of simulation fps
    data = read_npz(traj_path)

    iterations = [100]
    swarms = [100]
    c_tasks = [0.8]
    c_angles = [0.5]
    for it in iterations:
        for swarmsize in swarms:
            for c_task in c_tasks:
                for c_a in c_angles:
                    trajectory = {'obs': [], 'acs': [], 'hpe': []}

                    assert 'hpe' in data and 'obs' in data and 'acs' in data

                    if 'hpe' in data and 'obs' in data and 'acs' in data:
                        model = mp.load_model_from_path(model_path)
                        nsubstep = int(ceil(1/(fps * model.opt.timestep)))
                        sim = mp.MjSim(model, nsubsteps=nsubstep)
                        sim.reset()
                        if render:
                            viewer = MjViewerExt(sim)

                        # initialise environment
                        idvA, default_q = get_model_info(model)
                        default_mat = array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

                        default_q2 = rotmat2quat(default_mat)
                        default_q = default_q2 * default_q

                        init_pos = array([0, -0.8, 0.1])

                        m_in = np.zeros(shape=(5, 3))
                        for i in range(5):
                            for j in range(3):
                                m_in[i, j] = idvA[3 + i * 4].get_pos()[j]

                        for i in range(len(data['acs'])):

                            # set actions
                            mocap_pos = data['acs'][i][0:3]
                            mocap_rot = data['acs'][i][3:7]
                            ctrl = data['acs'][i][7:]
                            sim.data.ctrl[:] = ctrl[:]
                            sim.data.mocap_pos[:] = mocap_pos[:]
                            sim.data.mocap_quat[:] = mocap_rot[:]

                            if i == 0:
                                # first frame
                                # set state
                                qpos = data['obs'][i][:len(sim.data.qpos)]
                                qvel = data['obs'][i][len(sim.data.qpos):]
                                sim.data.qpos[:] = qpos[:]
                                sim.data.qvel[:] = qvel[:]

                                # copy state to actuators
                                for j in range(len(idvA)):
                                    idvA[j].get_val_from_sim(sim)

                            if i == 0 or not np.array_equal(data['hpe'][i], data['hpe'][i-1]) or not per_hpe:
                                # first or new frame - retarget and apply pso

                                idvA = obs2actions(data['hpe'][i], idvA=idvA, init_pos=init_pos,
                                                   default_q=default_q, default_mat=default_mat, m_in=m_in, ad_hoc=False,
                                                   rot_scene=rot_scene)

                                curr_state = (sim.data.qpos, sim.data.qvel)

                                obj_name = "Object"
                                obj_state = get_joint_state(obj_name, sim.data)
                                if obj_state is not None:
                                    objects = {obj_name: obj_state}
                                else:
                                    objects = None

                                pair_dist = get_active_contacts_dist(data=sim.data,
                                                                     contact_pairs=get_pair_contacts(model=model))
                                if pair_dist and len(pair_dist) > 2:
                                    # there are at least 2 fingers close to the object - enable contact
                                    pso_params = {'contact': True, 'swarmsize': swarmsize, 'maxiter': it,
                                                  'minfunc': 1e-4, 'minstep': 1e-4, 'hybrid_prc': 10}
                                else:
                                    # no object close enough to grab
                                    pso_params = {'contact': False, 'swarmsize': 2, 'maxiter': 3,
                                                  'minfunc': 1e-1, 'minstep': 1e-1, 'hybrid_prc': 0.5}

                                if pso_params['contact']:   # apply pso only for contact
                                    for j in range(len(idvA) - 1):
                                        idvA[j].set_value(idvA[j].get_value(), safe=True)

                                    sub_params = idvA[0:23]

                                    actions, error = pso(params=sub_params, obs=data['hpe'][i], model=model, norm=True, fps=10,
                                                         visualise=False,
                                                         default_mat=default_mat, hybrid_prc=pso_params['hybrid_prc'],
                                                         contact=pso_params['contact'], swarmsize=pso_params['swarmsize'],
                                                         initial_act=idvA, omega=0.1, phip=0.3, phig=0.7,
                                                         minstep=pso_params['minstep'], maxiter=pso_params['maxiter'],
                                                         minfunc=pso_params['minfunc'], hybrid_space=True, objects=objects
                                                         , initial_state=curr_state, rot_scene=rot_scene,
                                                         c_task=c_task, c_angle=c_a)

                                    sub_params = particle2actuator(actions, sub_params)

                                for j in range(len(idvA)):
                                    idvA[j].assign(sim)

                            # record frame
                            trajectory['obs'].append(np.concatenate((np.asarray(sim.data.qpos),
                                                                           np.asarray(sim.data.qvel)), axis=0))
                            mocap = np.concatenate((sim.data.mocap_pos.flatten(), sim.data.mocap_quat.flatten()), axis=0)
                            trajectory['acs'].append(np.concatenate((np.asarray(mocap), np.asarray(sim.data.ctrl)), axis=0))

                            trajectory['hpe'].append(np.array(data['hpe'][i]))

                            sim.step()
                            if render:
                                viewer.render()

                        if name is None:
                            name = 'pso_optimise'

                        np.savez(name, **trajectory)

                        if replay:
                            play(model_path, data=trajectory, fps=fps, loop=False, second_data=data)


def main(args):
    if not os.path.isfile(args.model_path):
        print("model path does not exist. Terminating...")
        exit(1)
    args.model_path = os.path.abspath(args.model_path)
    files = None
    if os.path.isdir(args.traj_path):
        filesExp = os.path.join(args.traj_path, "*.npz")
        files = glob.glob(filesExp)
        files.sort()
    elif os.path.isfile(args.traj_path):
        files = [args.traj_path]
    else:
        print("trajectory path does not exist. Terminating...")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    for f in files:
        np.random.seed(args.seed)
        out_name = os.path.join(args.out_dir, os.path.basename(f)[:-4] + '_pso')
        print("now working on " + f)
        optimise_actions(args.model_path, f, rot_scene=args.rot_scene, fps=60,
                         render=False, name=out_name, replay=args.play)


if __name__ == "__main__":
    args = argsparser()
    main(args)
