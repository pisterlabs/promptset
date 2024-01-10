"""
    Base class for the mujoco environments
    -- THIS SHOULD NOT BE RAN INDEPENDENTLY, instead you should call one of the other four classes:
        - reach
        - push
        - slide
        - pick and place

    This provides the main work-horse methods for the environments, providing a setup and 
    rendering pipeline which can be altered based on the constructor's parameters.  
    Goals are environment specific and will not be found here.
"""

from hashlib import new
import  os
import  random
import  gym
import  mujoco_py
import  math
import  cv2
import  numpy                               as np
from    os              import path
from    collections     import deque
from    gym.envs.mujoco import mujoco_env
from    mujoco_py       import GlfwContext
from    dm_control      import mujoco       as mjcontrol


if __name__ == '__main__' or __name__=='RobotArm__BaseEnv':
    from  utils.gelsight_driver     import SimulationApproach
    from  utils.xml_dm_formatter    import xml_dmControl_formatting
else:
    from .utils.gelsight_driver     import SimulationApproach
    from .utils.xml_dm_formatter    import xml_dmControl_formatting


# offscreen rendering for the gelsight and environment camera
GlfwContext(offscreen=True)


# This class is effectively RobotPick 
# -- the reward structure should encourage an agent to reach for an object and pick it up
# -- this is also used as the parent class for the other four environments
class RobotArm(mujoco_env.MujocoEnv):
    # Constructor method for the robot env
    # -- this takes in a width and height for the gelsight outputs
    # -- Note: Some of the below code was taken from mujoco_env.py
    def __init__(self, 
                 render             = True,
                 return_real        = True,  
                 return_tact        = False,   
                 return_kine        = True,     
                 return_raw         = False,
                 gel_width          = 160,     
                 gel_height         = 120,      
                 image_grayscale    = True, 
                 rescale            = 84,      
                 image_normalise    = True,
                 envcam_width       = 720, 
                 envcam_height      = 720,
                 frame_stack        = False, 
                 k_frames           = 0,
                 has_object         = False,  
                 random_object      = False, 
                 has_target         = False,  
                 random_target      = False, 
                 t_dist_threshold   = 0.01,
                 calc_table_lims    = False,
                 model_name         = None,
                 frame_skip         = 600,
                 object_is_freejoint= False,
                 seed               = 42
                 ):
        
        # the environment must have either a target object or position
        # a model file must also be provided
        assert has_object or has_target
        assert model_name!=None

        # random seed for the env
        self.seed(seed)

        # obs returns
        # -- real: env camera
        # -- tact: gelsight renders
        # -- kine: arm information
        self.return_real        = return_real
        self.return_tact        = return_tact
        self.return_kine        = return_kine     
        self.return_raw         = return_raw
        if self.return_real==False and self.return_tact==False and self.return_kine==False:
            self.return_raw=True

        # should the environment render be shown to the user
        self.render_            = render

        # does the environment contain an object or target (should contain at least 1)
        self.has_object         = has_object
        self.random_object      = random_object
        self.has_target         = has_target
        self.random_target      = random_target

        # gelsight and render information
        self.gel_width          = gel_width
        self.gel_height         = gel_height
        self.envcam_width       = envcam_width
        self.envcam_height      = envcam_height
        self.image_grayscale    = image_grayscale
        self.image_rescale      = rescale
        self.image_normalise    = image_normalise
        self.frame_stack        = frame_stack
        if self.frame_stack:
            self._k             = k_frames
            self._frames        = deque([], maxlen=self._k)         

        # the model(xml) name
        # frame skip amount (frames between steps)
        # if the table limits are needed - used in slide's env
        self.model_name         = model_name
        self.frame_skip         = frame_skip
        self.calc_table_lims    = calc_table_lims

        self.object_is_freejoint= object_is_freejoint


        # load the xml
        self._load_model(self.model_name)

        # create our lookup dictionaries
        self.create_xml_dictionary(self.model_name)

        # get the initial positions and velocities of all the joints
        # -- so that we can reset to this at the end of an episode
        self._setup_ini_pos_vel()

        if self.has_target:
            self._target_position_setup(t_dist_threshold)

        # get the current position(s) and load them into curr and prev attributes
        self._update_current_positions()
        self._update_previous_positions()

        # if we are using random spawns we need to get valid coordinates to place them within
        if self.random_object or self.random_target:
            self._calc_valid_spawn_coordinates()

        # do we need the table limits (used in slide env)
        if self.calc_table_lims:
            self._calc_table_limits()

        # setup the metrics for this model
        self._setup_metrics()

        # get an initial reading from the gelsight sensors and set the initial depths
        self._setup_ini_gelsight()

        # get all the ids of the gelsight geoms
        self._get_geom_gelsight_ids()

        # test the model runs
        self._setup_test_model()


    # set the observation space of the environment
    # -- this should really be done using spaces.box, however, it would always throw type-errors for me
    def _set_observation_space(self, obs):
        
        spaces = {}

        high_pixel_int = 255
        if self.image_normalise:
            high_pixel_int = 1

        # get any real dimensions
        if self.return_real:
            # if returning more than one item
            if self.return_tact or self.return_kine or self.return_raw:
                real    = obs[0]
            else:
                real    = obs

            height              = np.shape(real)[0]
            width               = np.shape(real)[1]
            channel             = np.shape(real)[2]
            spaces['real']      = gym.spaces.Box(low=0, high=high_pixel_int, shape=(height, width, channel))

        # get any tactile dimensions
        if self.return_tact:
            if self.return_real:
                tact     = obs[1]
            else:
                if len(obs)>1:
                    tact = obs[0]
                else:
                    tact = obs
            height              = np.shape(tact)[0]
            width               = np.shape(tact)[1]
            channel             = np.shape(tact)[2]        
            spaces['tact']      = gym.spaces.Box(low=0, high=high_pixel_int, shape=(height, width, channel))

        # get any kinematic or raw dimensions
        if self.return_kine or self.return_raw:
            if self.return_real==False and self.return_tact==False:
                arm_count       = len(obs)
            else:
                final_obs_index = len(obs)-1
                arm_count       = len(obs[final_obs_index])
            # 10 high is the max actuation force possible
            spaces['arm']       = gym.spaces.Box(low=0, high=10, shape=(arm_count,))

        dict_space              = gym.spaces.Dict(spaces)
        self.observation_space_ = dict_space 
        

    # return the observation space for the environment
    def get_observation_space(self):
        return self.observation_space_

    
    # get the return mode
    def get_return_mode(self):
        return [self.return_real, self.return_tact, self.return_kine, self.return_raw]


    # take a step in the environment, based on a given action
    def step(self, action):
        # iterate the step and adjusted step amounts
        # -- the latter iterates as an int 
        self.step_          = round(self.data.time / self.model.opt.timestep)
        self.skip_step      = round(self.step_ / self.frame_skip)

        # put an action into the system
        self.do_simulation(action, self.frame_skip)

        # get the current and previous whereabouts of the object and target position
        # -- frameskip can hide the movement of objects, hence, update previous every 2 steps
        # -- unsure if this is contributing to poor convergence
        if self.skip_step%2==0:
            self._update_previous_positions()
        self._update_current_positions()

        # get all the current contacts in the environment
        self.contacts       = self.data.contact

        # is our target reached?
        if self.has_object:
            self.target_reached = self.goal_distance(self.curr_targ_pos, self.curr_targ_obj)<=self.dist_threshold
        else:
            self.target_reached = self._gelsight_reached_pos()

        # get the environment renders from the envcam and gelsight
        envCam = self._envcam_render()
        if self.return_tact:
            self._gelsight_depths()
            gel0, gel1 = self._get_gelsight()
        else:
            gel0, gel1 = [0], [0]

        # produce the ob, reward, done, info attributes for the environment
        ob      = self._get_obs(envCam, gel0, gel1)
        done    = self._get_done()
        reward  = self._get_reward()
        info    = self._get_info(gel0, gel1)

        if self.frame_stack and self.return_real:
            self._frames.append(ob[0])
            ob[0] = np.concatenate(list(self._frames), axis=2)

        return ob, reward, done, info


    # get the most recent observation data
    # -- the kinematics section was taken and adapted from OpenAI's gym FetchEnv
    #       https://github.com/openai/gym/blob/150404104afc57ea618acca34d2d1f797d779869/gym/envs/robotics/fetch_env.py#L11
    def _get_obs(self, envCam, gel0, gel1):

        # gripper information
        grip_pos                    = self.curr_grip_pos
        dt                          = self.sim.nsubsteps*self.model.opt.timestep
        grip_velp                   = self.sim.data.get_site_xvelp("gripper:grip")*dt
        arm_qpos, arm_qvel          = self._get_robotArm_joints("robot")
        gripper_qpos, gripper_qvel  = self._get_robotArm_joints("gripper")
        arm_state                   = arm_qpos
        arm_vel                     = arm_qvel*dt
        gripper_state               = gripper_qpos
        gripper_vel                 = gripper_qvel*dt

        if self.has_object:
            # object's:
            # -- positon
            # -- velocity
            # -- relative position of the gripper from the object
            object_pos      = self.sim.data.get_site_xpos("goal:target_object")
            object_velp     = self.sim.data.get_site_xvelp("goal:target_object") * dt
            object_velr     = self.sim.data.get_site_xvelr("goal:target_object") * dt
            object_rel_pos  = object_pos - grip_pos
            object_velp    -= grip_velp
        else:
            object_pos = (
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)

        # all the joint information
        # -- exclude the first 7 & 6 are removed as they are target object freejoint
        # -- freejoint qpos is 6 3d dof + 4d value 
        # -- qvel is just 6 3d dof
        arm_info = np.concatenate([
            grip_pos.flatten(),
            arm_state.flatten(),
            arm_vel.flatten(),
            object_pos.flatten(),
            object_rel_pos.flatten(),
            gripper_state.flatten(),
            object_velp.flatten(),
            object_velr.flatten(),
            grip_velp.flatten(),
            gripper_vel.flatten(),
        ])


        # grayscale and resize the image
        eCam_out = self._image_preprocessing(envCam)
        if self.return_tact:
            gel0_out = self._image_preprocessing(gel0)
            gel1_out = self._image_preprocessing(gel1)

        # combine the items into a list        
        # -- real: camera array viewing the objec
        # -- tact: gelsight renders
        # -- kine: arm and gripper joint information
        # -- raw:  just the gripper position
        obs = []
        if self.return_real:
            obs.append(np.array(eCam_out))
        if self.return_tact:
            obs.append(np.array(gel0_out))
            obs.append(np.array(gel1_out))
        
        if self.return_kine and self.return_raw:
            obs.append(np.append(np.array(arm_info), np.array(grip_pos)))
        else:
            if self.return_kine:
                obs.append(np.array(arm_info))
            if self.return_raw:
                obs.append(np.array(grip_pos))
            
        if len(obs)==1:
            return obs[0]
        else:
            return obs


    # return the done variable for this step
    def _get_done(self):
        self.success = False

        # if target object is in contact with the target position
        if self.target_reached:
            self.success =  True
            return True

        if self.has_object:
            # slide environment, there is no z-axis so sthe object cannot fall
            if self.calc_table_lims:
                if not self._get_table_contact_with_leeway():
                    return True
            # environments where the target object has a z-axis-joint (so it can fall)
            else:
                # if the object is below the z-axis or hits  the ground
                if self._target_object_current_pos()[2] <= 0.001:
                    return True

        # if a sufficient amount of time has passed
        if self.skip_step >= 100:
            return True

        return False    


    # reach class rewards
    # -- reward for finding the target position
    # -- goal_distance taken from Open AI's gym FetchEnv
    #        https://github.com/openai/gym/blob/150404104afc57ea618acca34d2d1f797d779869/gym/envs/robotics/fetch_env.py#L11
    def _get_reward(self):
        # both target and object position in model
        if self.has_object and self.has_target:
            dist_gripper_to_object  = self.goal_distance(self.curr_grip_pos, self.curr_targ_obj)
            dist_object_to_target   = self.goal_distance(self.curr_targ_pos, self.curr_targ_obj)
            reward                  = dist_gripper_to_object + dist_object_to_target
        # only target position in model
        else:
            # Compute distance between goal and the achieved goal.
            reward = self.goal_distance(self.curr_grip_pos, self.curr_targ_pos)

        return -reward 


    # return additional information of the environment
    def _get_info(self, gel0, gel1):
        info = {
            'actuator_force': self.data.actuator_force,
            'gelsight0': gel0,
            'gelsight1': gel1,
            'contacts' : self.data.active_contacts_efc_pos,
            'step'     : self.step_,
            'step_adj' : self.skip_step,
            'success'  : self.success,
            'soft_grip': self.soft_grip_steps,
            'hard_grip': self.hard_grip_steps,
            'lifted'   : self.lifted_steps,
        }

        return info


    # reset to the inital positons
    def reset_model(self):
        # object positions and velocities
        qpos                = self.init_qpos
        qvel                = self.init_qvel

        # reset the step sizes
        self.step_          = 0
        self.skip_step      = 0
        self.success        = False

        # reset any environment specific metrics
        self._reset_metrics()

        # set the new state
        self.set_state(qpos, qvel)

        # if the TO and TP are randomised
        if self.random_target or self.random_object:
            touching = True

            # if the object has been placed in contact with the gripper fingers, choose another position
            while(touching):
                new_object_touching_gripper = True
                new_target_touching_gripper = True
                object_target_touching      = True

                if self.has_object:
                    new_object = self._set_body_placement("target object")
                    # counter acts a Mujoco bug where you cannot set an object's body position
                    # if it has a <freejoint/>
                    if self.object_is_freejoint:
                        qpos[0] = new_object[0]
                        qpos[1] = new_object[1]
                        qpos[2] = new_object[2]
                if self.has_target:
                    new_target = self._set_body_placement("target position")
                
                self.set_state(qpos, qvel)

                # load the new position(s) into the 'current' attributes
                self._update_current_positions()
    
                # one or the other
                if self.has_target or self.has_object:
                    # target touching gripper
                    if self.has_target:
                        new_object_touching_gripper = self.goal_distance(self.curr_grip_pos, self.curr_targ_pos)>self.dist_threshold

                    # object touching gripper
                    if self.has_object:
                        new_target_touching_gripper = self.goal_distance(self.curr_grip_pos, self.curr_targ_obj)>self.dist_threshold

                    # both target and object included
                    if self.has_target and self.has_object:
                        # are they touching one another
                        object_target_touching      = self.goal_distance(self.curr_targ_pos, self.curr_targ_obj)>self.dist_threshold                 

                # neither target object or position are in the model
                # -- this error should be unreachable (but just incase)
                else:
                    raise ValueError("Error, incorrect combination of objectives.  Must include a target object or position")

                # are any of the involved objects touching
                touching = not(new_object_touching_gripper and new_target_touching_gripper and object_target_touching)

            # update the previous positions to be the new target
            self._update_previous_positions()

        # get a blank gelsight image for the initial state
        depth_one   = np.ones((self.gel_height, self.gel_width))
        output      = self.simulation.generate(depth_one)
        gel0        = output
        gel1        = output       
        envCam      = self._envcam_render()

        # generate the observation state
        obs         = self._get_obs(envCam, gel0, gel1)

        # stack the real frames
        # -- do it k number of times to clear the old frames out
        if self.frame_stack and self.return_real:
            for k in range(self._k):
                self._frames.append(obs[0])
                
            obs[0] = np.concatenate(list(self._frames), axis=2)

        return obs, self._get_info(gel0, gel1)


    # set the initial target position and threhold of closeness
    def _target_position_setup(self, dist_threshold):
        # slide specific stuff
        self.dist_threshold = dist_threshold
        self.ini_targ_pos   = self._get_target_position()


    # get the target position
    def _get_target_position(self):
        # use the body id rather than the geom id
        target_body_id  = self._get_body_id("target position")
        target_pos      = self._get_body_pos(target_body_id)

        return target_pos


    # load the mujoco specific things 
    # -- most of this code is taken from the __init__ of mujocoEnv.py
    # -- it's taken instead of inherited as other parameters need to be injected in the middle
    def _load_model(self, model_name):
        # Mujoco env parameters
        dir_path    = os.path.dirname(os.path.realpath(__file__))
        model_path  = os.path.join(dir_path, "./models/" + model_name)

        # load the .xml
        fullpath = model_path

        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        # Mujoco env params
        self.model      = mujoco_py.load_model_from_path(fullpath)
        self.sim        = mujoco_py.MjSim(self.model)
        self.data       = self.sim.data
        self.viewer     = None
        self._viewers   = {}


    # initialise the gelsight sensors and get the initial depth readings
    def _setup_ini_gelsight(self):
        # get the sim approach (Daniel Gomes' code)
        # -- initialise the gelsight depth camers
        # -- pass the depth arrays into Gomes' gelsight simulation driver
        self._get_simapproach()
        self._initialise_gelsight_viewers()
        self._gelsight_depths()

        # initial depth readings (with nothing in the gel)
        # -- negative 0.001 to make it less sensitive to noise 
        self.ini_depth0, self.ini_depth1 = self.depth0-0.001, self.depth1-0.001
        

    # test the model runs
    # -- check we are not in the done state at initialisation
    def _setup_test_model(self):
        # set the action
        self._set_action_space()
        

        # inital object position (z-axis only)
        # -- everything spawns on the table, so this is effectively table height
        if self.has_object:
            self.ini_target_object_pos_z = self.curr_targ_obj[2]

        # test env with a blank action
        obs, _reward, done, _info  = self.step(0)

        # set the observation space 
        self._set_observation_space(obs)

        # check the environment is not in a done state
        assert not done


    # inital positions of the env
    def _setup_ini_pos_vel(self):
        # inital positions and velocities of the objects in the env
        self.init_qpos       = self.sim.data.qpos.ravel().copy()
        self.init_qvel       = self.sim.data.qvel.ravel().copy()


    # metrics to be recorded and used to grant rewards or recogneise the done state
    def _setup_metrics(self):
        # set number of steps to 0
        self.step_           = 0
        self.soft_grip_steps = 0
        self.hard_grip_steps = 0
        self.lifted_steps    = 0
    

    # these are the geom_ids of the gelsight sensor contact points
    def _get_geom_gelsight_ids(self):
        # I got these from obersvering contacts of geom1 and geom2 from self.data.contact
        # -- last values in each array are the friction placeholders, which are 
        #    invisible and are what hold the object up
        gelsight0_names = [ 'gelsight0:front', 'gelsight0:back', 'gelsight0:glass0', 'gelsight0:glass1'
                            'gelsight0:elastomer', 'gelsight0:elastCover', 'gelsight0:friction' ]

        gelsight1_names = [ 'gelsight1:front', 'gelsight1:back', 'gelsight1:glass0', 'gelsight1:glass1'
                            'gelsight1:elastomer', 'gelsight1:elastCover', 'gelsight1:friction' ]

        gel_0_geom_contacts = []
        gel_1_geom_contacts = []
        for i in range(len(self.geom_names)):

            # get all the geom ids that we do not want contacting one another
            # -- we do not want to return contact rewards for when the two gelsight elastomers
            #    come into contact with one another
            if self.geom_names[i] in gelsight0_names:
                gel_0_geom_contacts.append(i)
            if self.geom_names[i] in gelsight1_names:
                gel_1_geom_contacts.append(i)

        self.gel0 = gel_0_geom_contacts
        self.gel1 = gel_1_geom_contacts 

    
    # all the contacts in the environment
    # -- verbose prints all the objects to the terminal
    def _get_all_contacts(self, verbose=False):

        self.contacts = self.data.contact

        if verbose:
            print()
            for contact in self.contacts:
                if contact.geom1!=0 and contact.geom2!=0:
                    print("geom1", contact.geom1, "geom2", contact.geom2)

        return self.contacts


    # return a bool value as to whether the gelsight sights are colliding with one another
    def _gel_to_gel_contact(self):

        # get all the contacts in the environment
        contacts    = self._get_all_contacts()
        
        # for every contact
        for contact in contacts:
            
            # get the first geom in this collision
            geom1 = contact.geom1
            
            # if the first geom is within one of the ranges
            # -- then the gelsight elastomers are possibly in contact with one another
            # -- this checks all the gelsight geoms in case the environment is behaving in an irratic manner,
            #    such as joints clipping through one another
            in_gelsight_0_id_range = (geom1>=self.gel0[0] and geom1<=self.gel0[len(self.gel0)-1])
            in_gelsight_1_id_range = (geom1>=self.gel1[0] and geom1<=self.gel1[len(self.gel1)-1])

            if in_gelsight_0_id_range or in_gelsight_1_id_range:
                # is there a match between the two id ranges?
                geom2       = contact.geom2
                first_match = (geom1 in self.gel0) and (geom2 in self.gel1)
                secon_match = (geom1 in self.gel0) and (geom2 in self.gel1)
                if first_match or secon_match:
                    return True

        return False


    # update the current position of the objects and gripper
    def _update_current_positions(self):
        self.curr_grip_pos       = self.sim.data.get_site_xpos("gripper:grip")

        if self.has_object:
            self.curr_targ_obj   = self.sim.data.get_body_xpos("target object")
        if self.has_target:
            self.curr_targ_pos   = self.sim.data.get_body_xpos("target position")


    # update the prevus positons
    def _update_previous_positions(self):
        self.previous_grip_pos      = self.curr_grip_pos.copy()
        
        if self.has_object:
            self.previous_targ_obj  = self.curr_targ_obj.copy()
        if self.has_target:
            self.previous_targ_pos  = self.curr_targ_pos.copy()
    

    # has either gelsight sensor made contact with the target position?
    # -- target position is collision-less so it needs to be done like this
    #    instead of using the inbuilt contact attribute of Mujoco
    def _gelsight_reached_pos(self):
        
        within_goal_threshold = self.goal_distance(self.curr_grip_pos, self.curr_targ_pos)<=self.dist_threshold
        if within_goal_threshold:
            return True

        return False
    
    
    # distance of object a from object b 
    # -- taken from openAi's FetchEnv
    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)


    # preprocessing the gelsight state spaces
    def _image_preprocessing(self, img):
        # make a copy of the array, grayscale and resize
        output = img.copy()

        # grayscale
        if self.image_grayscale:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
            output = np.expand_dims(output, -1)
        # rescale
        if self.image_rescale is not None:
            output = cv2.resize(output, (self.image_rescale, self.image_rescale), interpolation=cv2.INTER_AREA)
        # normalise
        if self.image_normalise:
            output = output/255

        return output


    # initialise the offline viewer, provides:
    # -- depth calculations for the gelsight
    # -- viewable pixel arrays of the environment
    # -- NOT using an online viewer as it does not seem to work in conjunction with
    #    the offline viewer.  Specially I am unable to access both gelsight cameras.
    def _initialise_gelsight_viewers(self):
        self.offline    = mujoco_py.MjRenderContextOffscreen(self.sim,  device_id=-1)


    # get this step's new render information 
    def _gelsight_depths(self):
        # gelsight0 renders
        self.offline.render(self.gel_width, self.gel_height, camera_id=0)
        depth0      = self.offline.read_pixels(self.gel_width, self.gel_height, depth=True)[1].copy()

        # gelsight1 renders
        self.offline.render(self.gel_width, self.gel_height, camera_id=1)
        depth1      = self.offline.read_pixels(self.gel_width, self.gel_height, depth=True)[1].copy()
 
        # make available to the class
        self.depth0 = depth0
        self.depth1 = depth1


    # render the environment camera
    # -- render option, should the user be shown the environment
    def _envcam_render(self):
        self.offline.render(self.envcam_width, self.envcam_height,  camera_id=2)
        env_image   = self.offline.read_pixels(self.envcam_width, self.envcam_height, depth=True)[0].copy()

        # draw the environemnt
        if self.render_:
            env_display = cv2.flip(env_image, 0)   
            cv2.imshow('env_display', env_display)
            cv2.waitKey(1)

        return env_image


    # get the target object position
    def _target_object_current_pos(self):
        return self.sim.data.get_geom_xpos("target_object")


    # gelsight sensor simulation data
    # -- taken from Daniel Gomes' gelsight_driver.py main() function
    #       https://github.com/danfergo/gelsight_simulation
    def _get_simapproach(self, reduce_size=True):
        # sensor light sources
        light_sources_smartlab2014 = [
            {'position': [0, 1, 0.25],  'color': (255, 255, 255),   'kd': 0.6, 'ks': 0.5},  # white, top
            {'position': [-1, 0, 0.25], 'color': (255, 130, 115),   'kd': 0.5, 'ks': 0.3},  # blue, right
            {'position': [0, -1, 0.25], 'color': (108, 82, 255),    'kd': 0.6, 'ks': 0.4},  # red, bottom
            {'position': [1, 0, 0.25],  'color': (120, 255, 153),   'kd': 0.1, 'ks': 0.1},  # green, left
        ]

        # load the image taken from a real gelsight sensor
        dir_path        = os.path.dirname(os.path.realpath(__file__))
        model_path      = os.path.join(dir_path, './models/assets/background.png')
        background_img  = cv2.imread(model_path)


        # rescaling to fit the gelsight cameras
        if reduce_size:
            background_img = cv2.resize(background_img, (self.gel_width, self.gel_height), interpolation=cv2.INTER_AREA)

        # calculation attributes
        ka                  = 0.8
        px2m_ratio          = 5.4347826087e-05
        elastomer_thickness = 1  
        min_depth           = 0 
        texture_sigma       = 0.00001 

        # get the simlulation model
        self.simulation = SimulationApproach(
            light_sources       = light_sources_smartlab2014,
            background_img      = background_img,
            ka                  = ka,
            texture_sigma       = texture_sigma,
            px2m_ratio          = px2m_ratio,
            elastomer_thickness = elastomer_thickness,
            min_depth           = min_depth
        )

    
    # get the depth arrays from the gelsight cameras
    def _get_gelsight(self, render_cvs=False):
        # process the pixel arrays collected 
        gel1 = self.simulation.generate(self.depth1)
        gel0 = self.simulation.generate(self.depth0)
        
        # render the gelsight sensor images
        if render_cvs:
            # render in cv2
            cv2.imshow('Gelsight depth1',  gel1)
            cv2.waitKey(1)
            cv2.imshow('Gelsight depth0',  gel0)
            cv2.waitKey(1)

        return gel0, gel1
    

    # return the size of a particular geom
    def _get_geom_size(self, id):
        # all geoms
        geoms = self.model.geom_size

        return geoms[id]


    # return the size of a particule body
    def _get_body_pos(self, id):
        # all bodies
        bodies = self.model.body_pos

        return bodies[id]


    # default viewer 
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        # how far the viewer is from the robot
        self.viewer.cam.distance = 1


    # get all the robot arm joints
    # -- goal_distance taken from Open AI's gym FetchEnv
    #        https://github.com/openai/gym/blob/150404104afc57ea618acca34d2d1f797d779869/gym/envs/robotics/fetch_env.py#L11
    def _get_robotArm_joints(sim, name_):
        if sim.data.qpos is not None and sim.model.joint_names:
            names = [n for n in sim.model.joint_names if n.startswith(name_)]
            return (
                np.array([sim.data.get_joint_qpos(name) for name in names]),
                np.array([sim.data.get_joint_qvel(name) for name in names]),
            )
        return np.zeros(0), np.zeros(0)


    # split purely to make inheritance easier
    def _reset_metrics(self):
        # metric related attributes
        self.contact_time    = 0
        self.soft_grip_steps = 0
        self.hard_grip_steps = 0
        self.lifted_steps    = 0        

    
    # calculate the 2d positional vector for an object
    # -- this will return a rectangle regardless of the object's shape
    def _calc_object_limits_2d(self, pos, size):

        pos_x, pos_y    = pos[0],  pos[1]
        size_x, size_y  = size[0], size[1]

        bot_left_lim    = [(pos_x - size_x), (pos_y - size_y)] 
        top_right_lim   = [(pos_x + size_x), (pos_y + size_y)] 

        return [bot_left_lim, top_right_lim]


    # calculate the 3d positional vector for an object
    # -- this will return a cuboid regardless of the object's shape
    def _calc_object_limits_3d(self, pos, size):

        pos_x, pos_y, pos_z     = pos[0],  pos[1],  pos[2]
        size_x, size_y, size_z  = size[0], size[1], size[2]

        bot_left_lim            = [(pos_x - size_x), (pos_y - size_y), (pos_x - size_z)] 
        top_right_lim           = [(pos_x + size_x), (pos_y + size_y), (pos_z + size_z)] 

        return [bot_left_lim, top_right_lim]


    # take in the action choosen by the network and make it so our environment will understand
    # -- translate the value into an array int(16) -> arr[8]
    # -- assign an incremental step to the action
    # -- actions are discrete
    def action_handler(self, action, current_actions):
        action_ = current_actions.copy()

        # first two actions are both arm joints
        # -- their actuators are inverted, hence postive and negative for each choice
        if action<=3:
            offset  = 0.01
            if action%2==0:
                action_[0] += offset
                action_[1] -= offset
            else:
                action_[0] -= offset
                action_[1] += offset
        
        # all other actuators
        else:
            offset  = .1
            index   = math.floor(action/2)
            if action%2==0:
                action_[index] += offset
            else:
                action_[index] -= offset

        return action_


    # treat the play space as planar (2d flat surface)
    def _calc_table_limits(self):
        # table geom id
        table_id        = self._get_geom_id('surface')
        table_size      = self._get_geom_size(table_id)

        # table position in the env
        table_body_id   = self._get_body_id('table')
        table_pos       = self._get_body_pos(table_body_id)

        self.table_bot_left_lim, self.table_top_right_lim  = self._calc_object_limits_2d(table_pos, table_size) 

   
    # when spawning randomly, restrict the locations to the "spawn surface" geom
    def _calc_valid_spawn_coordinates(self):
        # spawn surface geom id
        spawn_surface_id        = self._get_geom_id('spawn surface')
        spawn_surface_size      = self._get_geom_size(spawn_surface_id)

        # surface id
        surface_id              = self._get_geom_id('surface')
        surface_size            = self._get_geom_size(surface_id)

        # table position in the env
        table_body_id           = self._get_body_id('table')
        table_pos               = self._get_body_pos(table_body_id).copy()

        # offset the table_pos by the geom size (assumes it fills the table)
        table_size              = spawn_surface_size
        table_pos[0]           -= (surface_size[0]-spawn_surface_size[0])/2
        table_pos[1]           -= (surface_size[1]-spawn_surface_size[1])/2

        self.bot_left_spawn_lim, self.top_right_spawn_lim = self._calc_object_limits_2d(table_pos, table_size) 


    # move a body in the environment to random location    
    def _set_body_placement(self, target_name):
        # table position in the env
        target_id    = self._get_body_id(target_name)
        target_pos   = self._get_body_pos(target_id)

        target_pos[0], target_pos[1], target_pos[2] = self._set_object_pos_randomly(target_name)

        return target_pos[0], target_pos[1], target_pos[2]


    # randomly set valid position for the target object or target position
    # -- validity: must be within the random of the table's surface
    def _set_object_pos_randomly(self, object_name):
        # get the target object's body id
        target   = self._get_body_id(object_name)

        # get the geom's x and y sizes to be used as an offset 
        # -- so that the object does not spawn off the table at the start
        # -- dividing by 4 as we need half the length size for each dimension
        #    and we are then apply the offset on both bot_left and top_right
        x_offset = self._get_geom_size(target)[0]/4
        y_offset = self._get_geom_size(target)[1]/4

        # randomly select numbers in the range of those limits
        # -- the z-axis does not change as we want the object to be on the table every time
        # -- rounded to reduce the state space complexity
        x = round(random.uniform(self.bot_left_spawn_lim[0] + x_offset, self.top_right_spawn_lim[0] - x_offset), 2)
        y = round(random.uniform(self.bot_left_spawn_lim[1] + y_offset, self.top_right_spawn_lim[1] - y_offset), 2)
        z = self.model.body_pos[target][2]

        return [x, y, z]
    

    # goals for this env - for use with hddqn
    # -- must be implemented 
    def get_goals(self):
        raise NotImplementedError()


    # intrinsic agent rewards for this env
    # -- must be implemented 
    def get_intrinsic_reward(self):
        raise NotImplementedError()


    # is the target object closer to the target on a particular axis?
    # used in goal calculations for an intrinsic agent
    # only cares if the gripper has move in the correct direction, compared to its previous position
    # -- Up, Down, Left, Right
    def _closer_by_UDLR(self, axis, threshold=0):
        # get the direction required (p_x = target position x coordinate)
        # -- if p_x - o_x < 0 then move left
        # -- if p_x - o_x > 0 then move right
        # -- if p_y - o_y < 0 then move down (y axis)
        # -- if p_y - o_y > 0 then move up   (y axis)
        # -- if p_z - o_z < 0 then move up   (z axis)
        # -- if p_z - o_z > 0 then move down (z axis)
        x = (self.curr_grip_pos[0] - self.previous_grip_pos[0])
        y = (self.curr_grip_pos[1] - self.previous_grip_pos[1])
        z = (self.curr_grip_pos[2] - self.previous_grip_pos[2])
        # if the object is on the rhs and the number decreases
        if axis==0 and x>+threshold:
            return True
        if axis==1 and x<-threshold:
            return True
        if axis==2 and y>+threshold:
            return True
        if axis==3 and y<-threshold:
            return True        
        if axis==4 and z>+threshold:
            return True
        if axis==5 and z<-threshold:
            return True

        return False


    """Slide env calculations for table limits"""
    # is our object still in contact with the table?
    # -- no z-axis joint so this checks if the object is above the positon coordinates of the table
    def _get_table_contact(self):
        # take the bot left and top right of the valid area (assumes a rectangle)
        # -- checks both x and y coordinates fall inbetween this range
        inside_x = (self.curr_targ_obj[0] > self.table_bot_left_lim[0] and 
                    self.curr_targ_obj[0] < self.table_top_right_lim[0])
        inside_y = (self.curr_targ_obj[1] > self.table_bot_left_lim[1] and 
                    self.curr_targ_obj[1] < self.table_top_right_lim[1])
        
        if inside_x and inside_y:
            return True
        
        return False

    
    # timing adjustment for contact
    # -- sometimes the model will throw out a false positional reading when the gripper
    #    comes into contact with the object, this is usually corrected within a step or two
    #    hence this method gives the calculations some leeway
    def _get_table_contact_with_leeway(self):
        
        if not self._get_table_contact():
            self.missing_surface_contact_time += 1
        else:
            self.missing_surface_contact_time  = 0
        
        if self.missing_surface_contact_time  >= 2:
            return False

        return True


    # create a tmp_xml.xml file - needed for dm_control as it does not know what "include" tags do
    # -- opens the target model
    # -- create a copy
    # -- scans for 'include', removes the tag and copies the body from that file
    # -- loops until no more includes are present
    def create_xml_dictionary(self, model_name):
        # create temp file and load it into deep mind's Mujoco wrapper
        format          = xml_dmControl_formatting(model_name)
        self.tmp_file   = format.get_output_name()
        mj_physics      = mjcontrol.Physics.from_xml_path(self.tmp_file)

        # all geoms and bodies
        geoms           = self.model.geom_size
        bodies          = self.model.body_pos

        # setup containers for all the geom and body names
        self.geom_names = []
        self.body_names = []

        # create a lookup array for id to name
        # -- using a standard array instead of a hashmap because most lookups
        #    we need are from id->name
        for i in range(np.shape(geoms)[0]):
            self.geom_names.append(mj_physics.model.id2name(i, 'geom'))

        for i in range(np.shape(bodies)[0]):
            self.body_names.append(mj_physics.model.id2name(i, 'body'))


    # find the geom id of the table's surface
    def _get_geom_id(self, geom):
        # search the geom_names array for the table surface geom
        for i in range(len(self.geom_names)):
            if self.geom_names[i]==geom:
                # found
                return i
        # didn't find
        return -1


    # find the main body id of the table
    def _get_body_id(self, body):
        # search the body_names array for the table body (main file)
        for i in range(len(self.body_names)):
            if self.body_names[i]==body:
                # found
                return i
        # didn't find
        return -1


    """ Deprecated - previously used in reward and done method(s) """
    # does the gelsight have full contact with the object
    # -- is it touching the friction placeholder?
    def _gel_in_contact_with_target(self):
        
        # initially set both to not be in contact 
        gel0, gel1 = False, False

        # get the gelsight geom ids and their associated friction values
        gel0_friction, gel1_friction = self.gel0, self.gel1
        gel0_friction, gel1_friction = gel0_friction[len(gel0_friction)-1], gel1_friction[len(gel1_friction)-1]

        # for all the contacts in the environment, 
        # check if either or both gelsight are in contact with the target object
        for contact in self.contacts:

            geom1 = contact.geom1
            
            # are any in contact with the first geom
            if (geom1==1) or (geom1==gel0_friction) or (geom1==gel1_friction):

                geom2 = contact.geom2

                # if in contact with the first one
                if (geom1==1 or geom2==1) and (geom1==gel0_friction or geom2==gel0_friction):
                    gel0 = True

                # if in contact with the second one
                if (geom1==1 or geom2==1) and (geom1==gel1_friction or geom2==gel1_friction):
                    gel1 = True

        return gel0, gel1


    # has the object been picked up
    def _calc_pickup_time(self):
        # get the friction placeholder contacts against the target object (bool)
        contact_gel0, contact_gel1 = self._gel_in_contact_with_target()

        # if the object has been picked up for a certain amount of time
        # -- is the target above where it started
        # -- is something in contact with the gelsight sensors
        target_above_inital_pos = self.curr_targ_obj[2] > self.ini_target_object_pos_z+0.00001
        contact_on_one_gelsight = (self.ini_depth0 > self.depth0).any() or (self.ini_depth1 > self.depth1).any()
        contact_with_fric_ph    = contact_gel0 and contact_gel1

        if target_above_inital_pos and contact_on_one_gelsight and contact_with_fric_ph:
            # add one to the contact time and lifted steps - former is the end state latter is for general information
            # -- if it gets to 5 sequential readings then end with successful outcome
            self.contact_time += 1
            self.lifted_steps += 1
            
        # else reset the contact_time counter
        else:
            self.contact_time = 0