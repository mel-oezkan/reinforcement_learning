import os
from re import M
import time
from typing import no_type_check

from tensorflow.python.eager.backprop import GradientTape

import gym
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_math_ops import sigmoid_grad_eager_fallback

import tensorflow_probability as tfp

ENV_ID = "LunarLanderContinuous-v2"
RANDOM_SEED = 1
RENDER = False

TRAIN_EPS = 5000
TEST_EPS = 10
MAX_STEPS = 500
VAR = 1

class PolicyGradient(tf.keras.Model):

    def __init__(self, n_actions=2):
        super(PolicyGradient, self).__init__()
        self.n_actions = n_actions

        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(128, activation="relu")

        self.mu = layers.Dense(n_actions, activation="tanh")
        self.sig = layers.Dense(n_actions, activation="sigmoid")

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)

        mu = self.mu(x)
        sig = self.sig(x) + 1e-8

        return mu, sig


class Agent:
    def __init__(self, VAR, lr=0.01, gamma=0.9999):

        self.gamma = gamma

        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

        self.model = PolicyGradient()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def get_action(self, x):
        mu, sig = self.model(x)

        action_space = tfp.distributions.Normal(mu, sig)
        sample_action = action_space.sample()
        
        return sample_action

    def store_transition(self, s, a, r):
        self.state_buffer.append(np.array([s], np.float32))
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    def _discount_and_norm_reward(self):
        """ compute discount_and_norm_rewards """
        discount_reward_buffer = np.zeros((len(self.reward_buffer), 1))
        running_add = 0

        for t in reversed(range(0, len(self.reward_buffer))):
            # Gt = R + gamma * V'
            running_add = self.reward_buffer[t] +  self.gamma * running_add
            discount_reward_buffer[t] = running_add

        # normalize episode rewards
        discount_reward_buffer -= np.mean(discount_reward_buffer)
        discount_reward_buffer /= np.std(discount_reward_buffer)

        return discount_reward_buffer


    def learn(self):

        discount_reward_buffer_norm = self._discount_and_norm_reward()

        with tf.GradientTape() as tape:
            
            state = tf.cast(self.state_buffer, dtype=tf.float32)
            state = tf.squeeze(state, axis=1)

            mu, sig = self.model(state)
            
            action_space = tfp.distributions.Normal(mu, sig)
            sample_action = action_space.sample()
            
            log_prob = action_space.log_prob(sample_action)
            log_prob = tf.squeeze(log_prob, axis=1)
            
            discount_reward_buffer_norm = tf.convert_to_tensor(
                discount_reward_buffer_norm, dtype=tf.float32)


            loss =  tf.reduce_sum(-log_prob * discount_reward_buffer_norm)
            
        if not tf.math.is_nan(loss):
            grad = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        else:
            print(f"sample action: {(state.shape)}")
            
            print("something went wrong again")
            raise TypeError("gradient is nan")

        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def save(self):
        self.model.save('my_model')
        print("Succeed to save model weights !")

    def load(self):
        self.model = tf.keras.models.load_model('my_model')




if __name__ == '__main__':

    env = gym.make(ENV_ID).unwrapped

    agent = Agent(VAR)

    t0 = time.time()

    print(f"action spce: {env.action_space}")
    max_rew = 0
    min_rew = 0

    all_episode_reward = []
    for episode in range(TRAIN_EPS):

        state = env.reset()
        state = np.expand_dims(state, axis=0)

        episode_reward = 0

        for step in range(MAX_STEPS):  # in one episode
            if (episode+1) % 50 == 0:
                env.render()

            action = agent.get_action(state)
            action = tf.squeeze(action, axis=0) # shape = (2,)

            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward)

            state = next_state
            state = np.expand_dims(state, axis=0)

            episode_reward += reward
            if done: 
                break

        agent.learn()
        print(
            '| Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                episode + 1, TRAIN_EPS, episode_reward,
                time.time() - t0
            )
        )

        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

    agent.save()
    plt.plot(all_episode_reward)
    if not os.path.exists('image'):
        os.makedirs('image')
