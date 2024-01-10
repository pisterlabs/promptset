import gym
import pygame
from grafter.wrapper import GrafterWrapper


KEYWORD_TO_KEY = {
    (ord("a"),): 1,
    (ord("w"),): 2,
    (ord("d"),): 3,
    (ord("s"),): 4,
    (ord('e'),): 5,
    (ord('q'),): 6,
    (ord('r'),): 7,
    (ord('f'),): 8,
    (ord('c'),): 9,
    (ord('t'),): 10,
    (ord('g'),): 11,
    (ord('v'),): 12,
    (ord('y'),): 13,
    (ord('h'),): 14,
    (ord('b'),): 15,
    (ord('u'),): 16,
}
from pygame.locals import VIDEORESIZE


class PlayWrapper(gym.Wrapper):
    """
    Modified from OpenAI gym's play functionality, so the renderer doesn't try to scale the pixel rgb output
    """

    def __init__(self, env, seed=100):
        assert isinstance(
            env, GrafterWrapper
        ), "This wrapper only works with the GrafterWrapper environment"
        super().__init__(env)

        self._seed = seed

    def _display_arr(self, screen, arr, video_size, transpose):
        pyg_img = pygame.surfarray.make_surface(
            arr.swapaxes(0, 1) if transpose else arr
        )
        pyg_img = pygame.transform.scale(pyg_img, video_size)
        screen.blit(pyg_img, (0, 0))

    def play(
        self, transpose=True, fps=5, zoom=None, callback=None, keys_to_action=KEYWORD_TO_KEY
    ):
        """Allows one to play the game using keyboard.

        To simply play the game use:

            play(gym.make("Pong-v4"))

        Above code works also if env is wrapped, so it's particularly useful in
        verifying that the frame-level preprocessing does not render the game
        unplayable.

        If you wish to plot real time statistics as you play, you can use
        gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
        for last 5 second of gameplay.

            def callback(obs_t, obs_tp1, action, rew, done, info):
                return [rew,]
            plotter = PlayPlot(callback, 30 * 5, ["reward"])

            env = gym.make("Pong-v4")
            play(env, callback=plotter.callback)


        Arguments
        ---------
        env: gym.Env
            Environment to use for playing.
        transpose: bool
            If True the output of observation is transposed.
            Defaults to true.
        fps: int
            Maximum number of steps of the environment to execute every second.
            Defaults to 30.
        zoom: float
            Make screen edge this many times bigger
        callback: lambda or None
            Callback if a callback is provided it will be executed after
            every step. It takes the following input:
                obs_t: observation before performing action
                obs_tp1: observation after performing action
                action: action that was executed
                rew: reward that was received
                done: whether the environment is done or not
                info: debug info
        keys_to_action: dict: tuple(int) -> int or None
            Mapping from keys pressed to action performed.
            For example if pressed 'w' and space at the same time is supposed
            to trigger action number 2 then key_to_action dict would look like this:

                {
                    # ...
                    sorted(ord('w'), ord(' ')) -> 2
                    # ...
                }
            If None, default key_to_action mapping for that env is used, if provided.
        """
        self.env.reset()
        self.env.seed(self._seed)
        rendered = self.env.render(mode="rgb_array")

        if keys_to_action is None:
            if hasattr(self.env, "get_keys_to_action"):
                keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, "get_keys_to_action"):
                keys_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                assert False, (
                    self.env.spec.id
                    + " does not have explicit key to action mapping, "
                    + "please specify one manually"
                )
        relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

        video_size = [rendered.shape[1], rendered.shape[0]]
        if zoom is not None:
            video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

        pressed_keys = []
        running = True
        env_done = True

        screen = pygame.display.set_mode(video_size)
        clock = pygame.time.Clock()

        while running:
            if env_done:
                env_done = False
                obs = self.env.reset()
            else:
                action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
                prev_obs = obs
                obs, rew, env_done, info = self.env.step(action)
                if callback is not None:
                    callback(prev_obs, obs, action, rew, env_done, info)
            if obs is not None:
                rendered = self.env.render(mode="rgb_array")
                self._display_arr(
                    screen, rendered, transpose=transpose, video_size=video_size
                )

            # process pygame events
            for event in pygame.event.get():
                # test events, set key states
                if event.type == pygame.KEYDOWN:
                    if event.key in relevant_keys:
                        pressed_keys.append(event.key)
                    elif event.key == 27:
                        running = False
                elif event.type == pygame.KEYUP:
                    if event.key in relevant_keys:
                        pressed_keys.remove(event.key)
                elif event.type == pygame.QUIT:
                    running = False
                elif event.type == VIDEORESIZE:
                    video_size = event.size
                    screen = pygame.display.set_mode(video_size)
                    print(video_size)

            pygame.display.flip()
            clock.tick(fps)
        pygame.quit()

    def _callback(self, prev_obs, obs, action, rew, env_done, info):
        if rew != 0:
            print(f"\nReward: {rew}")
        if env_done:
            print(f"Done!")
        if len(info) > 0:
            print(info)
