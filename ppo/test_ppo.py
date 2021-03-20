import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.eager.backprop import _aggregate_grads
import tensorflow_probability as tfp

#####################  hyper parameters  ####################

ENV_NAME = 'LunarLanderContinuous-v2'        # environment name
ALG_NAME = 'PPO'
RANDOMSEED = 1                  # random seed

EP_MAX = 1000                    # total number of episodes for training
EP_LEN = 100                    # total number of steps for each episode
GAMMA = 0.9                     # reward discount
A_LR = 0.01                   # learning rate for actor
C_LR = 0.02                   # learning rate for critic
BATCH = 32                      # update batchsize

A_UPDATE_STEPS = 10     # actor update steps
C_UPDATE_STEPS = 10     # critic update steps

EPS = 1e-8              # epsilon

# 注意：这里是PPO1和PPO2的相关的参数。
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty  PPO1
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better  PPO2
][1]                                                # choose the method for optimization

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters
EPSILON = 0.2

###############################  PPO  ####################################
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = tf.keras.layers.Dense(100, activation="relu")
        self.fc2 = tf.keras.layers.Dense(100, activation="relu")

        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        q_val = self.out(x)

        return q_val

class Actor(tf.keras.Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.fc1 = tf.keras.layers.Dense(100, activation="relu")
        self.fc2 = tf.keras.layers.Dense(100, activation="relu")

        self.a = tf.keras.layers.Dense(action_dim, activation="tanh")
       
        self.sigma = tf.keras.layers.Dense(action_dim, activation="softplus")

    def call(self, state, check=False):
        x1 = self.fc1(state)
        x2 = self.fc2(x1)
        
        if check:
            if tf.reduce_any(tf.math.is_nan(x2)):
                print(f"state: {state}")
                print(f"hidden1: {x1}")
                print(f"hidden2: {x2}")
                raise TypeError ("Hidden layer is nan")

        mu = self.a(x2)
        mu *= self.action_bound 

        sig = self.sigma(x2)

        return mu, sig

        

class PPO:
    def __init__(self, action_dim, action_bound, method='clip'):

        self.critic = Critic()

        self.actor = Actor(action_dim, action_bound)
        self.actor_old = Actor(action_dim, action_bound)
        
        self.actor_opt = tf.keras.optimizers.Adam(A_LR)
        self.critic_opt = tf.keras.optimizers.Adam(C_LR)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        
        self.action_bound = action_bound

    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        s = s[np.newaxis, :]
        mu, sigma = self.actor(s, check=True)

        if np.any(np.isnan(sigma)):
            print("faulty sigma")
            raise Exception("sigma value is smaller 0")

            
        pi = tfp.distributions.Normal(mu, sigma)    
        a = tf.squeeze(pi.sample(1), axis=0)[0]  

        if np.any(np.isnan(a)):
            print(f"state: {s}")
            print(f"mu: \n {mu}")
            print(f"sig: \n {sigma}")        
            raise Exception("Choosing is faulty")
        
        return np.clip(a, -self.action_bound, self.action_bound)

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    @tf.function
    def a_train(self, state, action, adv, test=False):
        '''
        policy network)
        '''

        state = tf.convert_to_tensor(state, dtype=tf.float32)         
        action = tf.convert_to_tensor(action, dtype=tf.float32)         
        
        # TD-Error
        adv = tf.convert_to_tensor(adv, dtype=tf.float32)     

        with tf.GradientTape() as tape:

            mu, sigma = self.actor(state)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(state)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            if test:
                print(mu_old)
                #print(surr)

            #ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(action) / (oldpi.prob(action) + EPS)
            surr = ratio * adv

            # PPO1
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(
                        ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv
                    ),
                    axis = [1]
                )
        a_grad = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grad, self.actor.trainable_weights))

        
            

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    
    def update_old_pi(self):
        '''
        更新actor_old参数。
        '''
        for pi, oldpi in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldpi.assign(pi)

    @tf.function
    def c_train(self, reward, state):
        ''' 更新Critic网络 '''
        # reward 是我们预估的 能获得的奖励
        reward = tf.convert_to_tensor(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)     # td-error
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def update(self, test=False):
        '''
        Update parameter with the constraint of KL divergent
        '''
        s = tf.convert_to_tensor(self.state_buffer, np.float32)
        a = tf.convert_to_tensor(self.action_buffer, np.float32)
        r = tf.convert_to_tensor(self.cumulative_reward_buffer, np.float32)

        self.update_old_pi()
        adv = (r - self.critic(s)).numpy()
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        # PPO2 clipping method, find this is better (OpenAI's paper)
        else:
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv, test)
        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()


    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done: v_s_ = 0
        else: v_s_ = self.critic(np.array([next_state], dtype=np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()



if __name__ == '__main__':

    env = gym.make(ENV_NAME).unwrapped

    # reproducible
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    ppo = PPO(
        action_dim = env.action_space.shape[0],
        action_bound = env.action_space.high,
    )

    
    all_ep_r = []
    test = False

    RENDER = 0
    for episode in range(EP_MAX):
        
        state = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        episode_reward = 0
        t0 = time.time()

        if episode == 2:
            test = True

        for t in range(EP_LEN):
            if RENDER:
                env.render()
    
            action = ppo.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ppo.store_transition(state, action, reward)
            state = state_
            
            episode_reward += reward
            
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                ppo.finish_path(state_, done)
                ppo.update(test)

        if episode == 0:
            all_ep_r.append(episode_reward)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + episode_reward * 0.1)
        print(
            'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode, EP_MAX, episode_reward,
                time.time() - t0
            )
        )

        RENDER = 0

    ppo.save_ckpt()
    plt.plot(all_ep_r)
    plt.show()

   