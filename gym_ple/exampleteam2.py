import gym
import random
from gym.envs.registration import registry, register, make, spec
import sys
sys.path.append('/content/PyGame-Learning-Environment')

from ple_env import PLEEnv
# Pygame
# ----------------------------------------
for game in ['Catcher', 'MonsterKong', 'FlappyBird', 'PixelCopter', 'PuckWorld', 'Snake', 'WaterWorld']:
    nondeterministic = False
    register(
        id='{}-v0'.format(game),
        entry_point='ple_env:PLEEnv',
        max_episode_steps=10000,
        kwargs={'game_name': game, 'display_screen':False},
        nondeterministic=nondeterministic,
    )

import logging
import os, sys

import ple_env
from gym.wrappers import RecordVideo
#from gym.wrappers import Monitor

import sys
sys.path.append('/content/PyGame-Learning-Environment')

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    video_folder = '/content/PyGame-Learning-Environment/output2'
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Dump result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
#    gym.upload(outdir)
    # Syntax for uploading has changed

