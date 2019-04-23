from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from keras.layers import Input, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np

import gym
import field

ENV_NAME = "Field-v0"
# del gym.envs.registry.env_specs[ENV_NAME]
# print(gym.envs.registry.env_specs)
env = gym.make(ENV_NAME)

### Modified from:
### https://hub.packtpub.com/build-reinforcement-learning-agent-in-keras-tutorial/

def build_model(state_size, num_actions):
    model = Sequential([
        Dense(16, input_shape = (state_size,), activation='sigmoid'),
        Dense(16, activation='sigmoid'),
        Dense(16, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    return model

def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks

    callbacks = build_callbacks(ENV_NAME)

num_actions = 100

memory = SequentialMemory(limit=500, window_length=1)
model = build_model(1, num_actions)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps = 0.05), attr='eps',
        value_max=1.0, value_min=0.01, value_test=0.05, nb_steps=10000)
callbacks = build_callbacks(ENV_NAME)

dqn = DQNAgent(model = model, nb_actions = num_actions,
        memory = memory, nb_steps_warmup = 1000,
        target_model_update = 1e-2, policy = policy)
dqn.compile(Adam(lr = 1e-3))
dqn.fit(env, nb_steps = 300000,
        visualize = False,
        verbose = 2,
        callbacks = callbacks)
dqn.test(env, nb_episodes = 5, visualize = False)
