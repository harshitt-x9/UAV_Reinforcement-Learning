import os
import keras
import tensorflow as tf
from keras.layers import Dense



class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=400, fc2_dims=300, name='critic',
                chkpt_dir='src/models/static'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                    self.model_name+'_ddpg.tf')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    # have to define inputs as a tuple because the model.save() function
    # trips an error when trying to save a call function with two inputs.
    def __call__(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q