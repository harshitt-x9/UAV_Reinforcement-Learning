import keras
import tensorflow as tf
from keras.optimizers import Adam

from src.DDPG.ReplayBuffer import ReplayBuffer
from src.DDPG.CriticNetwork import CriticNetwork
from src.DDPG.ActorNetwork import ActorNetwork

class Agent():
    def __init__(self, input_dims, alpha=0.001, beta=0.002,
                 gamma=1, n_actions=2, max_size=60000, tau=0.01,
                 fc1=400, fc2=300, fc3=200, batch_size=32, noise=0.1,
                 chkpt_dir='src/models/static', number=1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = 1.0
        self.min_action = -1.0
        self.chkpt_dir = chkpt_dir

        self.actor = ActorNetwork(n_actions=n_actions, fc1_dims=fc1, fc2_dims=fc2,
                                  name='actor_' + str(number))
        self.critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2,
                                     name='critic_'+ str(number))
        self.target_actor = ActorNetwork(n_actions=n_actions, fc1_dims=fc1, fc2_dims=fc2,
                                          name='target_actor_'+ str(number))
        self.target_critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2,
                                            name='target_critic_'+ str(number))
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

        def store_transition(self, state, action, reward, new_state, done):
            self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(filepath=self.actor.checkpoint_file)
        self.target_actor.load_weights(filepath=self.target_actor.checkpoint_file)
        self.critic.load_weights(filepath=self.critic.checkpoint_file)
        self.target_critic.load_weights(filepath=self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic
                                (states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                    self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))
        

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))
        
        self.update_network_parameters()