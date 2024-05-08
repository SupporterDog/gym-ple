import logging
import os
import sys

import gym
from gym.wrappers import RecordVideo
import gym_ple

import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  # 메모리 크기를 줄임
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.train_interval = 5  # 5 에피소드마다 한 번씩 학습

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

   def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def run_episodes(self, env, num_episodes, batch_size):
        for e in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(25):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, self.state_size])
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, num_episodes, time, self.epsilon))
                    break
            if e % self.train_interval == 0:
                self.replay(batch_size)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    # 환경 설정
    env_name = 'FlappyBird-v0' if len(sys.argv) < 2 else sys.argv[1]
    env = gym.make(env_name)
    env.seed(0)
    state_size = 288 * 512 * 3  # 상태 크기를 수정합니다.
    action_size = env.action_space.n

    # 에이전트 설정
    agent = DQNAgent(state_size, action_size)

    # 학습 파라미터 설정
    batch_size = 32
    num_episodes = 25

    # 에피소드 실행 및 학습
    agent.run_episodes(env, num_episodes, batch_size)

    # 환경 종료
    env.close()

