import random
import logging
import os, sys
import gym_ple
import ray
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.tune.registry import register_env
from gym.wrappers.record_video import RecordVideo
import gymnasium as gym

from gym.envs.registration import registry, register, make, spec
from gym_ple.ple_env import PLEEnv
# Pygame
# ----------------------------------------
for game in ['Catcher', 'MonsterKong', 'FlappyBird', 'PixelCopter', 'PuckWorld', 'RaycastMaze', 'Snake', 'WaterWorld']:
    nondeterministic = False
    register(
        id='{}-v0'.format(game),
        entry_point='gym_ple:PLEEnv',
        max_episode_steps=10000,
        kwargs={'game_name': game, 'display_screen':False},
        nondeterministic=nondeterministic,
    )

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 환경 인스턴스 생성

    env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])

    video_folder = '/content/video'
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    env.seed(0)

    # DQN 에이전트 설정
    config = DQNConfig().environment(env="FlappyBird-v0")
    config = config.training(replay_buffer_config={
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }).resources(num_gpus=0)

    # 에이전트 초기화
    agent = DQN(config=config)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        done = False
        while not done:
            action = agent.compute_action(ob)
            ob, reward, done, _ = env.step(action)

    env.close()
    ray.shutdown()

    logger.info("Successfully ran RandomAgent.")
