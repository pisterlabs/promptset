import sys
import os

import rospy
import rosgraph
import moveit_commander
from gazebo_msgs.srv import SpawnModel, GetModelState
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from openai import OpenAI

from api_config import api_key, base_url
from prompt_examples import debug_prompt, code_prompt, chat_prompt, gazebo_prompt, jibot3_prompt
from sdf.SDFCreator import SDFCreator

class AgentBase():
    def __init__(self, name='agent_base', model='gpt-3.5-turbo'):
        self.code_prompt = code_prompt
        self.debug_prompt = debug_prompt
        self.chat_prompt = chat_prompt
        self.allow_code_history_messages = False
        self.allow_debug_history_messages = True
        self.allow_chat_history_messages = False
        self.code_history_messages = []
        self.debug_history_messages = []
        self.chat_history_messages = []
        self.model = model
        self.name = name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.retry_max = 3
        self.retry_count = 0

    def init_node(self):
        rospy.init_node(self.name)

    def load_prompt(self, prompt):
        self.code_prompt = prompt

    def run(self):
        while not rospy.is_shutdown():
            query = input('>>> ')
            if query == 'exit':
                break
            response = self.get_code_response(query)
            self.execute(response.content)

    def callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "@recognized_speech: %s", data.data)
        response = self.get_code_response(data.data)
        self.execute(response.content)

    def listen(self):
        rospy.Subscriber('recognized_speech', String, self.callback)
        # 保持python程序运行，直到节点被停止
        rospy.loginfo('Listening for speech...')
        rospy.spin()

    def chat(self):
        while True:
            query = input('>>> ')
            if query == 'exit':
                break
            response = self.get_chat_response(query)
            rospy.loginfo(self.name + ': ' + response.content)

    def _process_query(self, query, prompt, history_messages, allow_history):
        query_formatted = [{'role': 'user', 'content': query}]
        messages = prompt + history_messages + query_formatted if allow_history else prompt + query_formatted
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "text"},
            messages=messages
        )
        return response.choices[0].message

    def get_code_response(self, query):
        return self._process_query(query, self.code_prompt, self.code_history_messages, self.allow_code_history_messages)

    def get_debug_response(self, query):
        return self._process_query(query, self.debug_prompt, self.debug_history_messages, self.allow_debug_history_messages)
    
    def get_chat_response(self, query):
        return self._process_query(query, self.chat_prompt, self.chat_history_messages, self.allow_chat_history_messages)

    def execute(self, code: str):
        try:
            rospy.loginfo(self.name + ': \n```python\n' + code + '\n```')
            exec(code)
            rospy.loginfo(self.name + ': Done!' + '\n')
            return 0, None
        except Exception as e:
            rospy.logwarn('Terminal: ' + str(e))
            if self.retry_count >= self.retry_max:
                rospy.logwarn('Terminal: Retry limit reached! Failed to execute code!')
                self.retry_count = 0
                return 1, str(e)
            corrected_code = self.get_debug_response(code + '\n---\n' + str(e)).content
            self.retry_count += 1
            ret, err_str = self.execute(corrected_code)
            if ret:
                return ret, err_str
            else:
                if self.retry_count != 0:
                    self.retry_count = 0
                    self.debug_history_messages.append({'role': 'user', 'content': code + '\n---\n' + str(e)})
                    self.debug_history_messages.append({'role': 'assistant', 'content': corrected_code})
                return 0, None



class GazeboAgent(AgentBase):
    def __init__(self, name='gazebo_agent', model='gpt-3.5-turbo'):
        super().__init__(name=name, model=model)
        self.code_prompt = gazebo_prompt
        self.spawn_sdf_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.spawn_urdf_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.sdf = SDFCreator()

    def init_node(self):
        super().init_node()
        rospy.loginfo('Waiting for service ...')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        rospy.loginfo('Service ready!')
        moveit_commander.roscpp_initialize(sys.argv)

    def save_sdf(self, sdf_str: str, model_name: str):
        # 如果sdf文件夹不存在则创建
        if not os.path.exists('sdf'):
            os.makedirs('sdf')
        with open(f'sdf/{model_name}.sdf', 'w') as f:
            f.write(sdf_str)

    def spawn_cube(self, position, model_name: str, model_size = 0.1, model_mass = 0.1):
        
        cube_sdf = self.sdf.create_cube(model_size, model_mass)

        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]

        self.spawn_sdf_model(model_name, cube_sdf, '', pose, 'world')

class Jibot3Agent(AgentBase):
    def __init__(self, name='jibot3_agent', model='gpt-3.5-turbo'):
        super().__init__(name=name, model=model)
        self.code_prompt = jibot3_prompt
        self.arm_group = moveit_commander.MoveGroupCommander('arm')
        self.gripper_group = moveit_commander.MoveGroupCommander('gripper')
    
    def move_joint(self, joint_goal_input, degree_rad: str, mode: str):
        # robot = moveit_commander.RobotCommander()
        # scene = moveit_commander.PlanningSceneInterface()
        if degree_rad not in ["degree", "rad"]:
            raise ValueError("The second argument 'degree_rad' should be 'degree' or 'rad'.")
        if mode not in ['INC', 'ABS']:
            raise ValueError("The second argument 'mode' should be 'INC' or 'ABS'.")
        if degree_rad == 'rad' and (max(joint_goal_input) > 3.14 or min(joint_goal_input) < -3.14):
            raise ValueError("Each element of the first argument 'joint_goal_rad' should be in range [-3.14, 3.14] if 'degree_rad' is 'rad'.")
        if degree_rad == 'degree' and (max(joint_goal_input) > 180 or min(joint_goal_input) < -180):
            raise ValueError("Each element of the first argument 'joint_goal_rad' should be in range [-180, 180] if 'degree_rad' is 'degree'.")
        
        if degree_rad == "degree":
            joint_goal_input = [i * 3.14 / 180 for i in joint_goal_input]

        joint_goal = self.get_joint_values()
        previous_joints = joint_goal
        rospy.loginfo("Joint values before moving: " + str(joint_goal))
        if mode == 'INC':
            joint_goal[0] += joint_goal_input[0]
            joint_goal[1] += joint_goal_input[1]
            joint_goal[2] -= joint_goal_input[2]
            joint_goal[3] += joint_goal_input[3]
            joint_goal[4] += joint_goal_input[4]
        elif mode == 'ABS':
            joint_goal[0] = joint_goal_input[0]
            joint_goal[1] = joint_goal_input[1]
            joint_goal[2] = joint_goal_input[2]
            joint_goal[3] = joint_goal_input[3]
            joint_goal[4] = joint_goal_input[4]

        self.arm_group.go(joint_goal, wait=True)
        # self.arm_group.stop()  # 调用stop()确保没有剩余的移动

        current_joints = self.get_joint_values()
        rospy.loginfo("Joint values after moving: " + str(current_joints))

        if previous_joints == current_joints:
            raise ValueError("The robot failed to move to the target joint values. Please check the joint values and try again.")

        # 清理moveit_commander
        # moveit_commander.roscpp_shutdown()

    def move_gripper(self, goal_state):
        if goal_state == "open":
            self.gripper_group.set_named_target("gripper_open")
        elif goal_state == "close":
            self.gripper_group.set_named_target("gripper_close")
        else:
            rospy.loginfo("Invalid command to gripper. Use 'open' or 'close'.")
            raise ValueError("The second argument 'goal_state' should be 'open' or 'close'.")

        self.gripper_group.go(wait=True)
        self.gripper_group.stop()

        # 清理moveit_commander
        # moveit_commander.roscpp_shutdown()

    def move_to(self, pose: str):
        if pose == "straight":
            self.arm_group.set_named_target("arm_default")
        elif pose == "default":
            self.arm_group.set_named_target("arm_zero")
        else:
            rospy.loginfo("Invalid command to arm. Use 'straight' or 'default'.")
            raise ValueError("The second argument 'pose' should be 'straight' or 'default'.")

        self.arm_group.go(wait=True)
        self.arm_group.stop()


    def move_to_cartesian_pose(self, cartesian_pose, mode):
        """
        移动机械臂的末端执行器到指定的笛卡尔坐标和姿态。

        Args:
            x (float): 目标X坐标。
            y (float): 目标Y坐标。
            z (float): 目标Z坐标。
            qx (float): 目标姿态的四元数X分量。
            qy (float): 目标姿态的四元数Y分量。
            qz (float): 目标姿态的四元数Z分量。
            qw (float): 目标姿态的四元数W分量。
        """

        if mode not in ['INC', 'ABS']:
            raise ValueError("The second argument 'mode' should be 'INC' or 'ABS'.")

        previous_pose = self.get_cartesian_pose()
        rospy.loginfo("Pose before moving: " + str(previous_pose))
        pose_goal = Pose()

        if mode == 'INC':
            pose_goal.orientation.x = previous_pose.orientation.x + \
                cartesian_pose[3]
            pose_goal.orientation.y = previous_pose.orientation.y + \
                cartesian_pose[4]
            pose_goal.orientation.z = previous_pose.orientation.z + \
                cartesian_pose[5]
            pose_goal.orientation.w = previous_pose.orientation.w + \
                cartesian_pose[6]
            pose_goal.position.x = previous_pose.position.x + cartesian_pose[0]
            pose_goal.position.y = previous_pose.position.y + cartesian_pose[1]
            pose_goal.position.z = previous_pose.position.z + cartesian_pose[2]
        elif mode == 'ABS':
            pose_goal.orientation.x = cartesian_pose[3]
            pose_goal.orientation.y = cartesian_pose[4]
            pose_goal.orientation.z = cartesian_pose[5]
            pose_goal.orientation.w = cartesian_pose[6]
            pose_goal.position.x = cartesian_pose[0]
            pose_goal.position.y = cartesian_pose[1]
            pose_goal.position.z = cartesian_pose[2]

        self.arm_group.set_pose_target(pose_goal)
        self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets() # 清除目标位姿

        current_pose = self.get_cartesian_pose()

        if previous_pose == current_pose:
            raise ValueError("The robot failed to move to the target pose. Please check the pose values and try again.")

        # 关闭moveit_commander
        # moveit_commander.roscpp_shutdown()

    def get_cartesian_pose(self):
        pose = self.arm_group.get_current_pose().pose
        rospy.loginfo("Pose: " + str(pose))
        return pose
    
    def get_joint_values(self):
        joint_values = self.arm_group.get_current_joint_values()
        rospy.loginfo("Joint values: " + str(joint_values))
        return joint_values


if __name__ == '__main__':
    agent = AgentBase()
    agent.init_node()
    agent.listen()