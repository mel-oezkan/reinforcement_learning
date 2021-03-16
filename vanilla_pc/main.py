import os 
import time
from typing import no_type_check

from tensorflow.python.eager.backprop import GradientTape

import gym
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_probability as tfp

ENV_ID = "LunarLanderContinuous-v2"
RANDOM_SEED = 1
RENDER = False

TRAIN_EPS = 500
TEST_EPS = 10 
MAX_STEPS = 200
VAR = 2

class PolicyGradient(tf.keras.Model):

    def __init__(self, n_actions=2):
        super(PolicyGradient, self).__init__()
        
        self.fc1 = layers.Dense(64, activation="relu")
        self.fc2 = layers.Dense(64, activation="relu")
        
        self.mu = layers.Dense(n_actions, activation="tanh")
        self.sig = layers.Dense(n_actions)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)

        mu = self.mu(x)
        sig = self.sig(x)

        return mu, sig

    def reparam(self, mu, sig):
        eps = tf.random.normal(shape=mu.shape)
        z = eps * tf.exp(sig * .5) + mu
        
        return z  


class Agent:
    def __init__(self, action_range=2, lr=0.01, gamma=0.99):
        self.var = VAR
        self.gamma = gamma

        self.state_buffer = []
        self.action_buffer = [] 
        self.reward_buffer = []

        self.model = PolicyGradient(action_range)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def get_action(self, x):
        mu, sig = self.model(x)
        action_space = tfp.distributions.Normal(loc=mu, scale=sig)
        
        return action_space.sample()

    def store_transition(self, s, a, r):
        self.state_buffer.append(np.array([s], np.float32))
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    def _discount_and_norm_reward(self):
        """ compute discount_and_norm_rewards """
        discount_reward_buffer = np.zeros_like(self.reward_buffer)
        running_add = 0

        for t in reversed(range(0, len(self.reward_buffer))):
            # Gt = R + gamma * V'
            running_add = self.reward_buffer[t] +  self.gamma * running_add
            discount_reward_buffer[t] = running_add
        
        # normalize episode rewards
        discount_reward_buffer -= np.mean(discount_reward_buffer)
        discount_reward_buffer /= np.std(discount_reward_buffer)
        
        return discount_reward_buffer

    def log_normal_pdf(self, sample, mu, sig, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mu) ** 2. * tf.exp(-sig) + sig + log2pi),
            axis=[1,2])

    
    def learn(self):
        discount_reward_buffer_norm = self._discount_and_norm_reward()

        with tf.GradientTape() as tape:
            mu, sig = self.model(np.vstack(self.state_buffer))
            
            action_space = tfp.distributions.Normal(mu, sig)
            action = action_space.sample()

            log_prob = action_space.log_prob(action)
            log_prob = tf.reduce_sum(log_prob)

            loss = tf.reduce_sum(- log_prob * discount_reward_buffer_norm)
            
        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        
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
    
    agent = Agent()

    t0 = time.time()

    all_episode_reward = []
    for episode in range(TRAIN_EPS):
        print()

        state = env.reset()
        state = np.expand_dims(state, axis=0)

        episode_reward = 0

        for step in range(MAX_STEPS):  # in one episode
            if (episode+1) % 50 == 0:
                env.render()

            action = agent.get_action(state)
            action = tf.squeeze(action, axis=0)

            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward)
            
            state = next_state
            state = np.expand_dims(state, axis=0)

            episode_reward += reward
            if done: break

        agent.learn()
        print(
            'Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
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
   
