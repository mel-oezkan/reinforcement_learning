import gym
import numpy as np
from reinforce import Agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=4)
    env = gym.make("LunarLander-v2")

    score_hist = []

    n_eps = 2000
    RENDER = False

    for i in range(n_eps):
        done = False
        score = 0
        obs = env.reset()
        if i % 50 == 0:
            RENDER = True

        while not done:
            if RENDER: 
                env.render()

            act = agent.choose_action(obs)
            obs_, rew, done, info = env.step(act)

            agent.store_transition(obs, act, rew)
            obs = obs_
            
            score += rew
        score_hist.append(score)

        agent.learn()
        RENDER = False

        avg_score = np.mean(score_hist[-100:])
        print(f"eps: {i}   score: {score}   avg. score: {avg_score}")

    