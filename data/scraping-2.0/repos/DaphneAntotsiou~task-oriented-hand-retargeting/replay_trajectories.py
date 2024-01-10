__author__ = 'DafniAntotsiou'

import os
from functions import read_npz
import mujoco_py as mp
from math import ceil
from mjviewerext import MjViewerExt
import glob
import argparse


def argsparser():
    parser = argparse.ArgumentParser("Replay trajectories from Task Oriented Hand Motion Retargeting")
    parser.add_argument('--model_path', help='path to model xml', type=str, default="model/MPL/MPL_Sphere_6.xml")
    parser.add_argument('--traj_path', help='path of the trajectory file or directory', type=str, default='trajectories')
    parser.add_argument('--opt_dir', help='path of the optimised trajectory directory', type=none_or_str, default='trajectories/result')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    boolean_flag(parser, 'play', default=True, help='playback the original and optimised trajectories')
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


def none_or_str(value):
    if value == 'None':
        return None
    return value


def replay_frame(sim, frame, data):
    if 'obs' in data and 'acs' in data:
        assert frame < len(data['obs'])
        assert len(data['acs']) == len(data['obs'])

        if frame < len(data['obs']):
            # set actions
            mocap_pos = data['acs'][frame][0:3]
            mocap_rot = data['acs'][frame][3:7]
            ctrl = data['acs'][frame][7:]
            sim.data.ctrl[:] = ctrl[:]
            sim.data.mocap_pos[:] = mocap_pos[:]
            sim.data.mocap_quat[:] = mocap_rot[:]

            # set state
            qpos = data['obs'][frame][:len(sim.data.qpos)]
            qvel = data['obs'][frame][len(sim.data.qpos):]
            sim.data.qpos[:] = qpos[:]
            sim.data.qvel[:] = qvel[:]
    return sim


def play(model_path, data, fps=60, loop=False, second_data=None):
    model = mp.load_model_from_path(model_path)
    nsubstep = int(ceil(1 / (fps * model.opt.timestep)))
    sim = mp.MjSim(model, nsubsteps=nsubstep)
    sim.reset()
    viewer = MjViewerExt(sim)
    if second_data is not None:
        sim2 = mp.MjSim(model, nsubsteps=nsubstep)
        sim2.reset()
        viewer2 = MjViewerExt(sim2)
    if not isinstance(data, (list,)):
        data = [data]

    if second_data is not None and not isinstance(second_data, (list,)):
        second_data = [second_data]

    while True:
        for traj in range(len(data)):
            print("trajectory", traj)
            if 'obs' in data[traj] and 'acs' in data[traj]:
                    for i in range(len(data[traj]['acs'])):
                        replay_frame(sim, i, data[traj])
                        sim.step()
                        viewer.render()

                        if second_data is not None and traj < len(second_data) and i < len(second_data[traj]['acs']):
                            replay_frame(sim2, i, second_data[traj])
                            sim2.step()
                            viewer2.render()
        if not loop:
            break


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

    #replay
    data = []
    second_data = []
    for f in files:
        opt_name = os.path.join(args.opt_dir, os.path.basename(f)[:-4] + '_pso.npz')
        data.append(read_npz(f))
        if os.path.isfile(opt_name):
            # add second environment with the pso result to show side by side
            second_data.append(read_npz(opt_name))
        else:
            second_data = None

    play(model_path=args.model_path, data=data, second_data=second_data, loop=True, fps=60)


if __name__ == "__main__":
    args = argsparser()
    main(args)
