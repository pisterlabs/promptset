import openai
import pybullet as pb
import numpy as np
import optas
import sys
from time import sleep,perf_counter, time
import threading
import multiprocessing as mp
from scipy.spatial.transform import Rotation as Rot
import math

import re
from optas.spatialmath import *


class VisualBox:
    def __init__(
        self,
        base_position,
        half_extents,
        rgba_color=[0, 1, 0, 1.0],
        base_orientation=[0, 0, 0, 1],
    ):
        visid = pb.createVisualShape(
            pb.GEOM_BOX, rgbaColor=rgba_color, halfExtents=half_extents
        )
        self._id = pb.createMultiBody(
            baseMass=0.0,
            basePosition=base_position,
            baseOrientation=base_orientation,
            baseVisualShapeIndex=visid,
        )

    def reset(self, base_position, base_orientation=[0, 0, 0, 1]):
        pb.resetBasePositionAndOrientation(
            self._id,
            base_position,
            base_orientation,
        )

class Controller:

    def __init__(self, dt):
        self.response_gain = 0.4
        link_ee = 'storz_tilt_endoscope_link_cm_optical'
        T = 2
        kuka = optas.RobotModel(urdf_filename='resources/lbr_with_tilt_endoscope.urdf', time_derivs=[1])
        kuka_name = kuka.get_name()
        kuka.add_base_frame('pybullet_world', [1, 0, 0])
        self.kuka_name =kuka_name
        
        
        # box_pos = builder.add_parameter('box_pos', 3)
        # insertion_depth = builder.add_parameter('depth', 1)
        # axis_align = builder.add_parameter('axis_align', 3)

        T = 1
        builder = optas.OptimizationBuilder(T, robots=[kuka], derivs_align=True)
        # Setup parameters
        qc = builder.add_parameter("qc", kuka.ndof)  # current robot joint configuration
        pg = builder.add_parameter("pg", 7)

        # Get joint velocity
        dq = builder.get_model_state(kuka_name, t=0, time_deriv=1)

        # Get next joint state
        q = qc + dt * dq

        # Get jacobian
        J = kuka.get_global_link_geometric_jacobian(link_ee, qc)

        # Get end-effector velocity
        dp = J @ dq

        # Get current end-effector position
        pc = kuka.get_global_link_position(link_ee, qc)
        Rc = kuka.get_global_link_rotation(link_ee, qc)

        print("dp = {0}".format(dp.size()))
        Om = skew(dp[3:])

        # Get next end-effector position (Global)
        p = pc + dt * dp[:3]
        R = (Om * dt + I3()) @ Rc

        # Get next end-effector position (Current end-effector position)

        p = Rc.T @ dt @ dp[:3]
        R = Rc.T @ (Om * dt + I3())

        # Cost: match end-effector position
        Rotq = Quaternion(pg[3], pg[4], pg[5], pg[6])
        Rg = Rotq.getrotm()

        pg_ee = -Rc.T @ pc + Rc.T @ pg[:3]
        Rg_ee = Rc.T @ Rg

        diffp = p - pg_ee[:3]
        diffR = Rg_ee.T @ R

        W_p = optas.diag([1e3, 1e3, 1e3])
        builder.add_cost_term("match_p", diffp.T @ W_p @ diffp)

        w_dq = 0.01
        builder.add_cost_term("min_dq", w_dq * optas.sumsqr(dq))

        w_ori = 1e1
        builder.add_cost_term("match_r", w_ori * optas.sumsqr(diffR - I3()))

        builder.add_leq_inequality_constraint("eff_x", diffp[0] * diffp[0], 1e-6)
        builder.add_leq_inequality_constraint("eff_y", diffp[1] * diffp[1], 1e-8)
        builder.add_leq_inequality_constraint("eff_z", diffp[2] * diffp[2], 1e-8)
        # opts = {'qpsol': {'printLevel': 'none'}}
        opts = {"print_status": False,"print_out": False,"print_in": False,"print_header": False,"print_time": False,"print_iteration": False,"print":False}
        optimization = builder.build()
        # optimization.set_options(opts)
        # optas.set_print_level(0)
        # optas.print_options()
        # self.solver = optas.CasADiSolver(optimization).setup("sqpmethod", opts)
        self.solver = optas.ScipyMinimizeSolver(builder.build()).setup('SLSQP')


    # def finish_criteria(self, q, bp, z):
    #     tol = 0.005**2
    #     f = self._finish_criteria(q, bp, z)
    #     return f < tol

    def __call__(self, qc, pg):

        # resp = self.response_gain*f
        # print(resp)
        # output_file = "qpoases_output.txt"
        self.solver.reset_parameters({"qc": optas.DM(qc), "pg": optas.DM(pg)})
        # with open(output_file, "w") as f:
            # sys.stdout = f
        solution = self.solver.solve()

        return solution[f"{self.kuka_name}/dq"].toarray().flatten()



class Robot:
    def __init__(self,time_step):
        self.time_step = time_step
        self.id = pb.loadURDF(
            'resources/lbr_with_tilt_endoscope.urdf',
            basePosition=[1, 0, 0],
        )
        self._dq = np.zeros(7)
        self._robot = optas.RobotModel(urdf_filename='resources/lbr_with_tilt_endoscope.urdf', time_derivs=[0, 1])
        self._robot.add_base_frame('pybullet_world', [1, 0, 0])
        # self._J = self._robot.get_global_geometric_jacobian_function('lbr_link_ee')
        self._J = self._robot.get_global_geometric_jacobian_function('storz_tilt_endoscope_link_cm_optical')
        self._p = self._robot.get_global_link_position_function('storz_tilt_endoscope_link_cm_optical')
        self._Tf = self._robot.get_global_link_transform_function('storz_tilt_endoscope_link_cm_optical')

    def Tf(self):
        return self._Tf(self.q()).toarray()

    def p(self):
        return self._p(self.q()).toarray().flatten()

    def J(self, q):
        return self._J(q).toarray()

    @property
    def num_joints(self):
        return pb.getNumJoints(self.id)

    @property
    def joint_indices(self):
        return list(range(self.num_joints))

    @property
    def joint_info(self):
        joint_info = []
        for joint_index in self.joint_indices:
            joint_info.append(pb.getJointInfo(self.id, joint_index))
        return joint_info

    @property
    def joint_types(self):
        return [info[2] for info in self.joint_info]

    @property
    def revolute_joint_indices(self):
        return [
            joint_index
            for joint_index, joint_type in zip(self.joint_indices, self.joint_types)
            if joint_type == pb.JOINT_REVOLUTE
        ]

    @property
    def ndof(self):
        return len(self.revolute_joint_indices)

    def update(self):
        self._dq = self.dq()

    def q(self):
        return np.array([state[0] for state in pb.getJointStates(self.id, self.revolute_joint_indices)])

    def dq(self):
        return np.array([state[1] for state in pb.getJointStates(self.id, self.revolute_joint_indices)])

    def ddq(self):
        return (self.dq() - self._dq)/self.time_step

    def tau(self):
        return np.array([state[3] for state in pb.getJointStates(self.id, self.revolute_joint_indices)])

    def tau_ext(self):
        tau = self.tau()
        q = self.q().tolist()
        dq = self.dq().tolist()
        ddq = self.ddq().tolist()
        tau_ext = np.array(pb.calculateInverseDynamics(self.id, q, dq, ddq)) - tau
        # lim = 1.1*np.array([-7.898,  3.47 , -1.781, -5.007,  1.453,  1.878,  2.562])
        lim = 2*np.ones(7)

        for i in range(7):
            if -lim[i] <= tau_ext[i] < lim[i]:
                tau_ext[i] = 0.

        return tau_ext

    def f_ext(self):
        J = self.J(self.q())
        tau_ext = self.tau_ext()
        J_inv = np.linalg.pinv(J, rcond=0.05)
        f_ext = J_inv.T @ tau_ext
        f_ext *= np.array([0.01, 0.01, 0.01, 1, 1, 1])

        return f_ext

    def reset(self, q, deg=False):
        for joint_index, joint_position in zip(self.revolute_joint_indices, q):
            pb.resetJointState(self.id, joint_index, joint_position)

    def cmd(self, q):
        pb.setJointMotorControlArray(
            self.id,
            self.revolute_joint_indices,
            pb.POSITION_CONTROL,
            q,
        )

    def cmd_vel(self, dq):
        pb.setJointMotorControlArray(
            self.id,
            self.revolute_joint_indices,
            pb.VELOCITY_CONTROL,
            dq,
        )


class HRI:
    def __init__(self,key_path):
        # self.thread_rob = threading.Thread(target=self.robot_loop)

        # self.thread_hri = threading.Thread(target=self.hri_loop)
        self.manager = mp.Manager()
        self.shared_dict = self.manager.dict()

        self.depth = 0.1
        self.lock = threading.Lock()
        with open(key_path, 'r') as file:
            self.key = file.read()
        
    
    @ staticmethod
    def get_box_pose(box, noise=0.005):
        p, r = pb.getBasePositionAndOrientation(box._id)
        noisep = np.random.uniform(-noise, noise, size=(len(p),))
        noiser = np.random.uniform(-noise, noise, size=(len(r),))
        return np.array(p) + noisep, np.array(r) + noiser
    
    def run(self):

        input_thread = threading.Thread(target=self.hri_loop)
        input_thread.daemon = True  
        input_thread.start()

        self.robot_loop()
        

    def robot_loop(self):
        #robotic simulation setup
        if 'gui' in sys.argv:
            connect_args = [pb.GUI]
        else:
            connect_args = [pb.DIRECT]

        pb.connect(
            *connect_args
        )

        pb.resetSimulation()
        gravz = -9.81
        pb.setGravity(0, 0, gravz)

        sampling_freq = 240
        time_step = 1./float(sampling_freq)
        pb.setTimeStep(time_step)

        box_base_position = np.array([0.4, 0.4, 0.2])

        pb.resetDebugVisualizerCamera(
            cameraDistance=0.2,
            cameraYaw=0,
            cameraPitch=20,
            cameraTargetPosition=box_base_position,
        )
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
        
        robot = Robot(time_step)
        q0 = np.deg2rad([-40, -45, 0, 90, 0, -45, 0])
        robot.reset(q0)
        robot.cmd(q0)
        
        

        init_pos = np.array([0.2,0.2,0.3])
        controller = Controller(time_step)


        # environment setup
        box_half_extents = [0.01, 0.01, 0.01]
        # constdist = optas.np.array([0.0, 0.0, -0.04])
        box_id1 = VisualBox(
            base_position=init_pos ,
            half_extents=box_half_extents,
        )

        box_id2 = VisualBox(
            base_position=np.array([-0.3,-0.3,0.3]) ,
            half_extents=box_half_extents,
        )

        t = 0.
        rest_wait_time = 5.  # secs
        time_factor_for_rest = 0.0001

        # wait for everything is ready
        while pb.isConnected():
            pb.stepSimulation()
            sleep(time_step*time_factor_for_rest)
            t += time_step
            robot.update()
            if t > rest_wait_time:
                break
        # self.isLoopRun = True
        print("!!!System has been setup")
        # initial task
        
        depth = self.depth
        box_id = box_id1

        time_factor = 0.000001
        q = q0

        # loop for robotic simulation
        # while(pb.isConnected() and self.isLoopRun):
        start_time = time()
        print("Run into robot_loop")
        pginit = optas.np.array([0.4, 0.0, 0.06, 0.0, 1.00, 0.0, 0.0])
        # pginit = optas.np.array([float(init_pos[0]), float(init_pos[1]), float(init_pos[2]), 0.0, 1.00, 0.0, 0.0])
        while True:
            t = time() - start_time
            # if t > Tmax_start:
            #     break
            # box.reset(
            #     base_position=constdist)
            # base_orientation=yaw2quat(state[2]).toarray().flatten(),
            # )
            nv = 1.0
            pginit1 = pginit + optas.np.array(
                    [
                        0.03 * optas.np.sin(2 * math.pi * nv * t),
                        0.03 * optas.np.cos(2 * math.pi * nv * t),
                        0.00 * math.pi * t,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
            

            pb.resetBasePositionAndOrientation(
                box_id._id,
                 pginit1[:3],
                pginit1[3:],
            )
            # box_base_position, box_base_orientation = HRI.get_box_pose(box_id)
            # R_box = Rot.from_quat(box_base_orientation).as_matrix()

            # goal_z = -R_box[:3, 1]
            self.lock.acquire()
            try:
                pginit1[2] += self.depth
                dqgoal = controller(q, pginit1)
            finally:
                self.lock.release()

            q += time_step * dqgoal
            robot.cmd(q)

            pb.stepSimulation()
            sleep(time_step*time_factor)
            robot.update()

        pb.disconnect()

    def hri_loop(self):
        # Provide your OpenAI API key
        openai.api_key = self.key
        pattern = r"@{(-?\d+\.\d+)}"
        print("Here is chatgpt Thread")
        pre_command = 'assume you are a robotic controller, when I tell you move closer to my target. You directly give me a sentence like \"0.5\". No explains, only you can reply is a number from -1.0 to 1.0.\
        Further means number close to 1.0 or -1.0. Closer means number close to 1.0. if I say \"send it\", you need to output the number in such format: @{0.3}'
        conversation_history = ""
        conversation_history += f"User: {pre_command}\nAssistant: "

        response=chat_with_gpt(conversation_history)
        print("ChatGPT: " + response)
        # Example usage
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break
            # chat_history.append(user_input)
            conversation_history += f"User: {user_input}\nAssistant: "
            response = chat_with_gpt(conversation_history)
            # sys.stdout = sys.__stdout__
            # sys.stdout = original_stdout
            print("ChatGPT: " + response)

            match = re.search(pattern, response)
            if match:
                self.lock.acquire()
                self.depth = float(match.group(1))
                print("Robot: Command has been sent")
                self.lock.release()



def chat_with_gpt(prompt):
    messages = [
        {"role": "system", "content" : prompt}
        ]
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    if 'choices' in response and len(response.choices)>0:
        return response.choices[0].message['content']
    else:
        return None
    
def main():

    # setup pybullet
    
    key_path = sys.argv[1]
    instance = HRI(key_path)
    instance.run()



    # # Provide your OpenAI API key
    # openai.api_key = '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'

    # # chat_with_gpt("Target Position is")

    # # Example usage
    # while True:
    #     user_input = input("User: ")
    #     if user_input.lower() == 'exit':
    #         break
    #     response = chat_with_gpt(user_input)
    #     print("ChatGPT: " + response)


if __name__ == "__main__":
    sys.exit(main())