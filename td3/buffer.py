from math import gamma, tau
import numpy as np
import tensorflow as tf
import os


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones 

class CriticNetwork(tf.keras.Model):
    def __init__(self, beta, n_actions):
        super(CriticNetwork, self).__init__()
        self.n_actions = n_actions

        self.fc1 = tf.keras.layers.Dense(64, activation="relu")
        self.fc2 = tf.keras.layers.Dense(64, activation="relu")
        self.q = tf.keras.layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=beta)

    def call(self, state, action):
        q1_action_val = self.fc1(tf.concat([state, action], axis=1))
        q1_action_val = self.fc2(q1_action_val)

        q1 = self.q(q1_action_val)   

        return q1

class ActorNetwork(tf.keras.Model):
    def __init__(self, alpha, n_actions):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions

        self.fc1 = tf.keras.layers.Dense(64, activation="relu")
        self.fc2 = tf.keras.layers.Dense(64, activation="relu")
        
        self.mu = tf.keras.layers.Dense(self.n_actions, activation="tanh")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, 
            update_actor_interval=2, warmup=1000, 
            n_actions=2, max_size=1000000, batch_size=100,
            noise=0.1):
        
        self.gamma = gamma
        self.tau = tau
        self.max_actions = env.action_space.high
        self.min_actions = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, n_actions=n_actions)
        self.crititc1 = CriticNetwork(beta, n_actions=n_actions)
        self.crititc2 = CriticNetwork(beta, n_actions=n_actions)

        self.target_actor = ActorNetwork(alpha, n_actions=n_actions)
        self.target_critic_1 = ActorNetwork(beta, n_actions=n_actions)
        self.target_critic_2 = ActorNetwork(beta, n_actions=n_actions)

        self.noise = noise
        self