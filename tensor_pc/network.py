import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class PolicyGradientNet(keras.Model):

    def __init__(self, n_actions):
        super(PolicyGradientNet, self).__init__()
        self.n_actions = n_actions

        self.fc1 = Dense(64, activation="relu")
        self.fc2 = Dense(64, activation="relu")
        self.pi = Dense(self.n_actions, activation="softmax")

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        pi = self.pi(value)
        
        return pi


