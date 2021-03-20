import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Denses
import tensorflow_probability as tfp


class Actor(tf.keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.n_actions = n_actions

        self.fc1 = Dense(256, activation="relu")
        self.fc2 = Dense(256, activation="relu")

        self.mu = Dense(self.n_actions, activation="tanh")
        self.sig = Dense(self.n_actions, activation="softplus")

    def call(sefl, state):
        x = self.fc1(state)
        x = self.fc2(x)

        mu = self.mu(x)
        sig = self.sig(x)

        return mu, sig

class Critic(tf.keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.n_actions = n_actions

        self.fc1 = Dense(256, activation="relu")
        self.fc2 = Dense(256, activation="relu")

        self.val = Dense(1)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)

        val = self.val(x)
        return val


class PPO:
    def __init__(self, action_dim, action_bound, method='clip'):

        self.critic = Critic()

        self.actor = Actor(action_dim)
        self.actor_old = Actor(action_dim)

        self.actor_opt = tf.keras.optimizers.Adam(A_LR)
        self.critic_opt = tf.keras.optimizers.Adam(C_LR)

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []

    def train_actor(sefl):
        pass
