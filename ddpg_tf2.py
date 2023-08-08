import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from  buffer import ReplayBuffer
from  networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_dims, alpha = 0.001, beta = 0.002, env= None,
    gamma = 0.99, n_actions = 2, max_size = 1_000_000, tau = 0.005,
     fc1=512, fc2 = 512, batch_size = 64, noise= 0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')

        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor.compile(optimizer = Adam(learning_rate = alpha))
        self.critic.compile(optimizer = Adam(learning_rate = beta))

        self.target_actor.compile(optimizer = Adam(learning_rate = alpha))
        self.target_critic.compile(optimizer = Adam(learning_rate = alpha))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self,tau=None):
        if tau is None:
            tau = self.tau
        weights = []
        target_actor_weights = self.target_actor_weights

        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + target_actor_weights[i]*(1-tau))
        self.target_actor.set_weights(weights)


        weights = []
        target_critic_weights = self.target_critic_weights

        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + target_critic_weights[i]*(1-tau))
        self.target_critic.set_weights(weights)
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('....saving models ....')

        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
   
    def load_models(self):
        print('....loading models ....')

        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)


    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation],dtype=tf.float32)
        actions = self.actor(state)

        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],mean=0, stddev=self.noise)

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]
    
    def learn(self):
        pass
        