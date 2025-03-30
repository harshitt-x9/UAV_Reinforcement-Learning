import os
import keras
import tensorflow as tf
from keras.layers import Dense


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=400, fc2_dims=300, n_actions=2,
                  name='actor', chkpt_dir='src/models/static'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_ddpg.tf')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def __call__(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)