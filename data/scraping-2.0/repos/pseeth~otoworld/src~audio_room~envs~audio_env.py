import gym
from pyroomacoustics import MicrophoneArray, ShoeBox, Room, linear_2D_array, Constants
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from copy import deepcopy
import nussl
import sys
sys.path.append("../../")

import constants
from utils import choose_random_files

# Suppress 'c' argument errors caused by room.plot()
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


class AudioEnv(gym.Env):
    def __init__(
        self,
        room_config,
        source_folders_dict,
        resample_rate=8000,
        num_channels=2,
        bytes_per_sample=2,
        corners=False,
        absorption=0.0,
        max_order=2,
        step_size=1,
        acceptable_radius=.5,
        num_sources=2,
        degrees=np.deg2rad(30),
        reset_sources=True,
        same_config=False
    ):
        """
        This class inherits from OpenAI Gym Env and is used to simulate the agent moving in PyRoom.

        Args:
            room_config (List or np.array): dimensions of the room. For Shoebox, in the form of [10,10]. Otherwise,
                in the form of [[1,1], [1, 4], [4, 4], [4, 1]] specifying the corners of the room
            resample_rate (int): sample rate in Hz
            num_channels (int): number of channels (used in playing what the mic hears)
            bytes_per_sample (int): used in playing what the mic hears
            corners (bool): False if using Shoebox config, otherwise True
            absorption (float): Absorption param of the room (how walls absorb sound)
            max_order (int): another room parameter
            step_size (float): specified step size else we programmatically assign it
            acceptable_radius (float): source is considered found/turned off if agent is within this distance of src
            num_sources (int): the number of audio sources the agent will listen to
            degrees (float): value of degrees to rotate in radians (.2618 radians = 15 degrees)
            reset_sources (bool): True if you want to choose different sources when resetting env
            source_folders_dict (Dict[str, int]): specify how many source files to choose from each folder
                e.g.
                    {
                        'car_horn_source_folder': 1,
                        'phone_ringing_source_folder': 1
                    }
            same_config (bool): If set to true, difficulty of env becomes easier - agent initial loc, source placement,
                                and audio files don't change over episodes
        """
        self.resample_rate = resample_rate
        self.absorption = absorption
        self.max_order = max_order
        self.audio = []
        self.num_channels = num_channels
        self.bytes_per_sample = bytes_per_sample
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)
        self.action_to_string = {
            0: "Forward",
            1: "Backward",
            2: "Rotate Right",
            3: "Rotate Left",
        }
        self.corners = corners
        self.room_config = room_config
        self.acceptable_radius = acceptable_radius
        self.step_size = step_size
        self.num_sources = num_sources
        self.source_locs = None
        self.min_size_audio = np.inf
        self.degrees = degrees
        self.cur_angle = 0
        self.reset_sources = reset_sources
        self.source_folders_dict = source_folders_dict
        self.same_config = same_config

        # choose audio files as sources
        self.direct_sources = choose_random_files(self.source_folders_dict)
        self.direct_sources_copy = deepcopy(self.direct_sources)

        # create the room and add sources
        self._create_room()
        self._add_sources()
        self._move_agent(new_agent_loc=None, initial_placing=True)

        # If same_config = True, keep track of the generated locations
        self.fixed_source_locs = self.source_locs.copy()
        self.fixed_agent_locs = self.agent_loc.copy()

        # The step size must be smaller than radius in order to make sure we don't
        # overstep a audio source
        if self.acceptable_radius < self.step_size / 2:
            raise ValueError(
                """The threshold radius (acceptable_radius) must be at least step_size / 2. Else, the agent may overstep 
                an audio source."""
            )

    def _create_room(self):
        """
        This function creates the Pyroomacoustics room with our environment class variables.
        """
        # non-Shoebox config (corners of room are given)
        if self.corners:
            self.room = Room.from_corners(
                self.room_config, fs=self.resample_rate,
                absorption=self.absorption, max_order=self.max_order
            )

            # The x_max and y_max in this case would be used to generate agent's location randomly
            self.x_min = min(self.room_config[0])
            self.y_min = min(self.room_config[1])
            self.x_max = max(self.room_config[0])
            self.y_max = max(self.room_config[1])

        # ShoeBox config
        else:
            self.room = ShoeBox(
                self.room_config, fs=self.resample_rate,
                absorption=self.absorption, max_order=self.max_order
            )
            self.x_max = self.room_config[0]
            self.y_max = self.room_config[1]
            self.x_min, self.y_min = 0, 0

    def _move_agent(self, new_agent_loc, initial_placing=False):
        """
        This function moves the agent to a new location (given by new_agent_loc). It effectively removes the
        agent (mic array) from the room and then adds it back in the new location.

        If initial_placing == True, the agent is placed in the room for the first time.

        Args:
            new_agent_loc (List[int] or np.array or None): [x,y] coordinates of the agent's new location.
            initial_placing (bool): True if initially placing the agent in the room at the beginning of the episode
        """
        # Placing agent in room for the first time (likely at the beginning of a new episode, after a reset)
        if initial_placing:
            if new_agent_loc is None:
                loc = self._sample_points(1, sources=False, agent=True)
                print("Placing agent at {}".format(loc))
                self.agent_loc = loc
                self.cur_angle = 0  # Reset the orientation of agent back to zero at start of an ep 
            else:
                self.agent_loc = new_agent_loc.copy()
                print("Placing agent at {}".format(self.agent_loc))
                self.cur_angle = 0
        else:
            # Set the new agent location (where to move)
            self.agent_loc = new_agent_loc
        
        # Setup microphone in agent location, delete the array at previous time step
        self.room.mic_array = None

        if self.num_channels == 2:
            # Create the array at current time step (2 mics, angle IN RADIANS, 0.2m apart)
            mic = MicrophoneArray(
                linear_2D_array(self.agent_loc, 2, self.cur_angle,
                                constants.DIST_BTWN_EARS), self.room.fs
            )
            self.room.add_microphone_array(mic)
        else:
            mic = MicrophoneArray(self.agent_loc.reshape(-1, 1), self.room.fs)
            self.room.add_microphone_array(mic)

    def _sample_points(self, num_points, sources=True, agent=False):
        """
        This function generates randomly sampled points for the sources (or agent) to be placed

        Args:
            num_points (int): Number of [x, y] random points to generate
            sources (bool): True if generating points for sources (agent must be False)
            agent(bool): True if generating points for agent (sources must be False)

        Returns:
            sample_points (List[List[int]]): A list of [x,y] points for source location
            or
            random_point (List[int]): An [x, y] point for agent location
        """
        assert(sources != agent)
        sampled_points = []

        if sources:
            angles = np.arange(0, 2 * np.pi, self.degrees).tolist()
            while len(sampled_points) < num_points:
                chosen_angles = np.random.choice(angles, num_points)
                for angle in chosen_angles:
                    direction = np.random.choice([-1, 1])
                    distance = np.random.uniform(2 * self.step_size, 5 * self.step_size)
                    x = (self.x_min + self.x_max) / 2
                    y = (self.y_min + self.y_max) / 2
                    x = x + direction * np.cos(angle) * distance
                    y = y + direction + np.sin(angle) * distance
                    point = [x, y]
                    if self.room.is_inside(point, include_borders=False):
                        accepted = True
                        if len(sampled_points) > 0:
                            dist_to_existing = euclidean_distances(
                                np.array(point).reshape(1, -1), sampled_points)
                            accepted = dist_to_existing.min() > 2 * self.step_size
                        if accepted and len(sampled_points) < num_points:
                            sampled_points.append(point)
            return sampled_points
        elif agent:
            accepted = False
            while not accepted:
                accepted = True
                point = [
                    np.random.uniform(self.x_min, self.x_max),
                    np.random.uniform(self.y_min, self.y_max),
                ]

                # ensure agent doesn't spawn too close to sources
                for source_loc in self.source_locs:
                    if(euclidean(point, source_loc) < 2.0 * self.acceptable_radius):
                        accepted = False 
            
            return point

    def _add_sources(self, new_source_locs=None, reset_env=False, removing_source=None):
        """
        This function adds the sources to the environment.

        Args:
            new_source_locs (List[List[int]]): A list consisting of [x, y] coordinates if the programmer wants
                to manually set the new source locations
            reset_env (bool): Bool indicating whether we reset_env the agents position to be the mean
                of all the sources
            removing_source (None or int): Value that will tell us if we are removing a source
                from sources
        """
        # Can reset with NEW, randomly sampled sources (typically at the start of a new episode)
        if self.reset_sources:
            self.direct_sources = choose_random_files(self.source_folders_dict)
        else:
            self.direct_sources = deepcopy(self.direct_sources_copy)

        if new_source_locs is None:
            self.source_locs = self._sample_points(num_points=self.num_sources)
        else:
            self.source_locs = new_source_locs.copy()

        print("Source locs {}".format(self.source_locs))

        self.audio = []
        self.min_size_audio = np.inf
        for idx, audio_file in enumerate(self.direct_sources):
            # Audio will be automatically re-sampled to the given rate (default sr=8000).
            a = nussl.AudioSignal(audio_file, sample_rate=self.resample_rate)
            a.to_mono()

            # normalize audio so both sources have similar volume at beginning before mixing
            loudness = a.loudness()

            # # mix to reference db
            ref_db = -40
            db_diff = ref_db - loudness
            gain = 10 ** (db_diff / 20)
            a = a * gain

            # Find min sized source to ensure something is playing at all times
            if len(a) < self.min_size_audio:
                self.min_size_audio = len(a)
            self.audio.append(a.audio_data.squeeze())

        # add sources using audio data
        for idx, audio in enumerate(self.audio):
            self.room.add_source(
                self.source_locs[idx], signal=audio[: self.min_size_audio])

    def _remove_source(self, index):
        """
        This function removes a source from the environment

        Args:
            index (int): index of the source to remove
        """
        if index < len(self.source_locs):
            src = self.source_locs.pop(index)
            src2 = self.direct_sources.pop(index)

            # actually remove source from the room
            room_src = self.room.sources.pop(index)

    def step(self, action, play_audio=False, show_room=False):
        """
        This function simulates the agent taking one step in the environment (and room) given an action:
            0 = Move forward
            1 = Move backward
            2 = Turn right x degrees
            3 = Turn left x degrees

        It calls _move_agent, checks to see if the agent has reached a source, and if not, computes the RIR.

        Args:
            action (int): direction agent is to move - 0 (L), 1 (R), 2 (U), 3 (D)
            play_audio (bool): whether to play the the mic audio (stored in "data")
            show_room (bool): Controls whether room is visually plotted at each step

        Returns:
            Tuple of the format List (empty if done, else [data]), reward, done
        """
        # return reward dictionary for each step
        reward = {
            'step_penalty': constants.STEP_PENALTY,
            'turn_off_reward': 0,
            'closest_reward': 0,
            'orient_penalty': 0
        }

        #print('Action:', self.action_to_string[action])

        # movement
        x, y = self.agent_loc[0], self.agent_loc[1]
        done = False

        if action in [0, 1]:
            if action == 0:
                sign = 1
            if action == 1:
                sign = -1
            x = x + sign * np.cos(self.cur_angle) * self.step_size
            y = y + sign * np.sin(self.cur_angle) * self.step_size
        elif action == 2:
            self.cur_angle = round((self.cur_angle + self.degrees) % (2 * np.pi), 4)
            reward['orient_penalty'] = constants.ORIENT_PENALTY
        elif action == 3:
            self.cur_angle = round((self.cur_angle - self.degrees) % (2 * np.pi), 4)
            reward['orient_penalty'] = constants.ORIENT_PENALTY
        # Check if the new points lie within the room
        try:
            if self.room.is_inside([x, y], include_borders=False):
                points = np.array([x, y])
            else:
                points = self.agent_loc
        except:
            # in case the is_inside func fails
            points = self.agent_loc

        # Move agent in the direction of action
        self._move_agent(new_agent_loc=points)

        # Check if goal state is reached
        for index, source in enumerate(self.source_locs):
            # Agent has found the source
            if euclidean(self.agent_loc, source) <= self.acceptable_radius:
                print(f'Agent has found source {self.direct_sources[index]}. \nAgent loc: {self.agent_loc}, Source loc: {source}')
                reward['turn_off_reward'] = constants.TURN_OFF_REWARD
                # If there is more than one source, then we want to remove this source
                if len(self.source_locs) > 1:
                    # remove the source (will take effect in the next step)
                    self._remove_source(index=index)

                    # Calculate the impulse response
                    self.room.compute_rir()
                    self.room.simulate()
                    data = self.room.mic_array.signals

                    # Convert the data back to Nussl Audio object
                    data = nussl.AudioSignal(
                        audio_data_array=data, sample_rate=self.resample_rate)

                    if play_audio or show_room:
                        self.render(data, play_audio, show_room)

                    done = False
                    return data, [self.agent_loc, self.cur_angle], reward, done

                # This was the last source hence we can assume we are done
                else:
                    done = True
                    self.reset()
                    return None, [self.agent_loc, self.cur_angle], reward, done

        if not done:
            # Calculate the impulse response
            self.room.compute_rir()
            self.room.simulate()
            data = self.room.mic_array.signals

            # Convert data to nussl audio signal
            data = nussl.AudioSignal(
                audio_data_array=data, sample_rate=self.resample_rate)

            if play_audio or show_room:
                self.render(data, play_audio, show_room)

            # be careful not to give too much reward here (i.e. 1/min_dist could be very large if min_dist is quite small)
            # related to acceptable radius size because this reward is only given when NOT turning off a source
            min_dist = euclidean_distances(
                np.array(self.agent_loc).reshape(1, -1), self.source_locs).min()
            reward['closest_reward'] = (1 / (min_dist + 1e-4))
            #print('agent_loc:', self.agent_loc, 'source_locs:', self.source_locs) 
            #print('cur angle:', self.cur_angle)
            #print('reward:', reward)

            # Return the room rir and convolved signals as the new state
            return data, [self.agent_loc, self.cur_angle], reward, done

    def reset(self, removing_source=None):
        """
        This function re-creates the room, then places sources and agent randomly (but separated) in the room.
        To be used after each episode.

        Args:
            removing_source (int): Integer that tells us the index of sources that we will be removing
        """
        # re-create room
        self._create_room()

        if not self.same_config:
            # randomly add sources to the room
            self._add_sources()
            # randomly place agent in room at beginning of next episode
            self._move_agent(new_agent_loc=None, initial_placing=True)
        else:
            # place sources and agent at same locations to start every episode
            self._add_sources(new_source_locs=self.fixed_source_locs)
            self._move_agent(new_agent_loc=self.fixed_agent_locs, initial_placing=True)

    def render(self, data, play_audio, show_room):
        """
        Play the convolved sound using SimpleAudio.

        Args:
            data (AudioSignal): if 2 mics, should be of shape (x, 2)
            play_audio (bool): If true, audio will play
            show_room (bool): If true, room will be displayed to user
        """
        if play_audio:
            data.embed_audio(display=True)

            # Show the room while the audio is playing
            if show_room:
                fig, ax = self.room.plot(img_order=0)
                plt.pause(1)

            plt.close()

        elif show_room:
            fig, ax = self.room.plot(img_order=0)
            plt.pause(1)
            plt.close()
