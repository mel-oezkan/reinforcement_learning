import tensorflow as tf
from tensorflow.python.keras.backend import dtype
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense

import numpy as np
import gym
import time

class Policy_Net(tf.keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(Policy_Net, self).__init__()

        self.fc1 = Dense(fc1_dims, activation="relu")
        self.fc2 = Dense(fc2_dims, activation="relu")

        self.mu = Dense(n_actions, activation="tanh")
        self.sig = Dense(n_actions, activation="softplus")


    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)

        mu = self.mu(x)
        sig = self.sig(x)

        return mu, sig

class Q_Net(tf.keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(Q_Net, self).__init__()

        self.fc1 = Dense(fc1_dims, activation="relu")
        self.fc2 = Dense(fc2_dims, activation="relu")

        self.val = Dense(1)

    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(state)

        x = self.val(state)

        return x

class Agent:
    def __init__(self, env, gamma=0.9999):
        
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = Policy_Net(self.n_actions)
        self.act_opt = tf.keras.optimizers.Adam(learning_rate=0.0006)

        self.buffer_state = []
        self.buffer_act = []
        self.buffer_rew = []

        self.gamma = gamma

    def store(self, state, act, rew):
        self.buffer_state.append(state)
        self.buffer_act.append(act)
        self.buffer_rew.append(rew)

    #@tf.function
    def choose_act(self, obs, learn=False):
        state = tf.convert_to_tensor([obs], dtype=tf.float32)
        mu, sig =  self.actor(state)

        #print(f"mu.shape: {mu.shape}")
        #print(f"sig.shape: {sig.shape}")
        

        mu = tf.squeeze(mu, axis=-2)
        sig = tf.squeeze(sig, axis=-2)

        #print(f"mu.shape: {mu.shape}")
        #print(f"sig.shape: {sig.shape}")

        dist = tfp.distributions.Normal(mu, sig)
        act = dist.sample() 
        #print(f"act.shape: {act.shape}")

        # (1,steps,2)
        act = tf.squeeze(act, axis=0)
        #print(f"act.shape: {act.shape}")
        #print()

        act = tf.clip_by_value(act, self.min_action, self.max_action)

        if learn:
            log_prob = dist.log_prob(act) 
            log_prob = tf.squeeze(log_prob, axis=0)

            # print(log_prob)

            #if tf.math.reduce_any(tf.math.is_nan(log_prob)):
            #    if tf.math.reduce_any(tf.math.is_nan(act)):
            #        print(f"mu: {self.actor.trainable_variables}")
            #    else:
            #        print(act)
            
            return log_prob

        return act 

    def get_disc_rew(self):
        discount_reward_buffer = np.zeros((len(self.buffer_rew), 1))
        running_add = 0

        for t in reversed(range(0, len(self.buffer_rew))):
            # Gt <- R + gamma * G'
            running_add = self.buffer_rew[t] +  self.gamma * running_add
            discount_reward_buffer[t] = running_add

        # normalize episode rewards
        discount_reward_buffer -= np.mean(discount_reward_buffer)
        discount_reward_buffer /= np.std(discount_reward_buffer)

        return discount_reward_buffer

    def learn(self):
        loss = 0

        discounted_rew = self.get_disc_rew()

        with tf.GradientTape() as tape:
            states = tf.convert_to_tensor(self.buffer_state)

            # get action and its respective log_prob
            log_prob = self.choose_act(states, learn=True)
            # (batch, 2)
            
            # total_prob = tf.math.reduce_sum(log_prob, axis=0)
            #loss = tf.clip_by_value(loss, -1e15, 1)

            # compute loss from log_prob and discounted rewards
            loss = -tf.reduce_sum(log_prob * discounted_rew, axis=0)
            loss = tf.reduce_sum(loss)
            #loss -= loss -1000

        grad = tape.gradient(loss, self.actor.trainable_weights)

        self.act_opt.apply_gradients(zip(grad, self.actor.trainable_weights))

        self.buffer_state = []
        self.buffer_act = []
        self.buffer_rew = []

        return loss

    def save(self):
        self.actor.save('my_model')
        print("Succeed to save model weights !")

    def load(self):
        self.actor = tf.keras.models.load_model('my_model')


if __name__ == "__main__":

    tf.config.threading.set_inter_op_parallelism_threads(10)

    episodes = 3000
    timesteps = 200
    env = gym.make("LunarLanderContinuous-v2")
    
    agent = Agent(env)

    for ep in range(episodes):
        t0 = time.time()

        state = env.reset()
        state = np.expand_dims(state, axis=0)

        ep_rew = 0

        stepping = time.time()
        for step in range(timesteps):  # in one episode
            if (ep+1) % 10 == 0:
                env.render()

            action = agent.choose_act(state)
            # action = tf.squeeze(action, axis=0)
            
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward)

            state = next_state
            state = np.expand_dims(state, axis=0)

            ep_rew += reward

            if done: 
                break
        
        learning = time.time()
        _, total_prob = agent.learn()

        print(f"Episode: {ep}/{episodes}", end=" || ")
        print(f"Ep. Rew: {ep_rew:.0f}", end=" || ")
        print(f"Ep. Time: {(time.time() - t0):.5f}", end=" ||")
        print(f"Log_Prob: {(np.mean(total_prob)):.5f}", end=" || \n")
        # print(f"Step: {(learning - stepping):.5f}", end=" || ")
        # print(f"Learn: {(time.time() - learning):.5f}")
        