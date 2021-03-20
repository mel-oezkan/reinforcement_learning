import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

import numpy as np
from network import PolicyGradientNet

class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4):
        self.gamma = gamma 
        self.alpha = alpha
        
        self.n_actions = n_actions
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.policy = PolicyGradientNet(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.alpha))

    def choose_action(self, obs):
        state = tf.convert_to_tensor([obs], dtype=tf.float32)
        probs = self.policy(state)

        action_probs = tfp.distributions.Categorical(probs=probs)

        action = action_probs.sample()
    
        return action.numpy()[0]

    def store_transition(self, obs, act, rew):
        self.state_memory.append(obs)
        self.action_memory.append(act)
        self.reward_memory.append(rew)

    def learn(self):
        act = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rew = tf.convert_to_tensor(self.reward_memory, dtype=tf.float32)

        G = np.zeros_like(rew)
        for t in range(len(rew)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rew)):
                G_sum += rew[k]*discount
                discount *= self.gamma
            G[t] = G_sum

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)

                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(act[idx])

                loss += -g * tf.squeeze(log_prob)
        
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        






