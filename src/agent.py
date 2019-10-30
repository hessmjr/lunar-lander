"""
File containing all RL agent logic.
"""
import numpy as np
import random as rand
import collections as coll

from keras.models import *
from keras.layers import *
from keras.optimizers import *


class Learner(object):
    """
    Random action learner.  Basically does nothing but pass the random
    action the environment supplied and returns it.
    """
    def __init__(self, num_feat=4, num_acts=2, alpha=0, gamma=0, 
                 epsilon=0, epsilon_decay=0):
        self.train = False
    
    def querysetstate(self, state, rand_act):
        return rand_act

    def query(self, s_prime, reward, rand_act):
        return rand_act

    def terminate(self):
        return


class QLearner(Learner):
    """
    Q Learner based off ML4T learner, updated with Double Deep Q learning
    methodolgy to address continuous states.
    """

    def __init__(self, num_feat=4, num_acts=2,  alpha=0.2, gamma=0.4, 
                 epsilon=0.2, epsilon_decay=1.0):
        
        # set all user defined variables
        self.num_feat = num_feat
        self.num_acts = num_acts
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # set initial values
        self.train = True
        self.state = None
        self.action = None

        # create main prediction logic
        num_neurons = 64
        self.brain = Brain(num_feat, num_acts, alpha, num_neurons)

        # create memory for experience replay
        self.batch_size = 20
        self.memory = Memory(5000)

    def querysetstate(self, observation, rand_act):
        """
        Sets the initial state of the learner and retrieves the first
        action to take give that state.
        """
        # decay the randomness variable
        self.epsilon *= self.epsilon_decay

        # convert observation to numpy matrix
        state = np.reshape(observation, [1, self.num_feat])

        # retrieve action using explore/exploit logic
        action = self._get_action(state, rand_act)

        # save current state and action, return action to user
        self.state = state
        self.action = action
        return action

    def query(self, observation, reward, rand_act):
        """
        Trains the learner, if specified, with latest observations
        and predicts the next action to take.
        """
        # convert observation to numpy matrix
        s_prime = np.reshape(observation, [1, self.num_feat])

        # if still training add the newest observation and outcome to 
        # learner's memory, followed by experience replay
        if self.train:  
            sample = (self.state, self.action, reward, s_prime)
            self.memory.add(sample)
            self._exp_replay()

        # retrieve action using explore/exploit logic
        a_prime = self._get_action(s_prime, rand_act)

        # set the new state and action, return new action
        self.state = s_prime
        self.action = a_prime
        return a_prime

    def terminate(self, observation, reward):
        """
        Runs final training with given observations and updates the 
        logic target model with the current predictions
        """
        # convert observation to numpy matrix
        s_prime = np.reshape(observation, [1, self.num_feat])

        # if still training add the newest observation and outcome to 
        # learner's memory, followed by experience replay
        if self.train:  
            sample = (self.state, self.action, reward, s_prime)
            self.memory.add(sample)
            self._exp_replay()    

        # update the target model with current prediction weights
        self.brain.update_target()

    def _get_action(self, state, rand_act):
        """
        Chooses either a random action or utilizes the prediction 
        model to get the best predicted action
        """
        # check if still training and if random choice selected
        if self.train and rand.random() < self.epsilon:
            return rand_act

        # use model to select predictions and extract best action
        predictions = self.brain.predict(state)
        return predictions[0].argmax()

    def _exp_replay(self):
        """
        Updates Q model with experience replay by retrieving designated batches
        from memory and using Q update function to update the prediction model
        """
        # track all states visited and action prediction values
        states, updates = [], []

        # retrieve a sample to update from memory and if nothing returned
        # then stop processing
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return

        # iterate through batch samples and update
        for sample in batch:
            state, action, reward, s_prime = sample
            states.append(state)

            # calculate the target value with Q update function
            if reward != -100 and reward != 100:
                max_q = self.gamma * np.max(self.brain.predict(s_prime)[0])
                target = reward + max_q

            # if state is terminating then use only reward to update
            else:
                target = reward

            # update the prediction value for the given state
            predictions = self.brain.predict(state)
            predictions[0][action] = target
            updates.append(predictions)          

        # train the model with the update states sampled and new action prediction
        self.brain.train(np.squeeze(states), np.squeeze(updates))


class Brain:
    """
    Logic for the Q-learner.  Uses double deep NN with prediction model
    and target model.
    """

    def __init__(self, num_feat, num_acts, alpha, num_neurons):
        self.num_feat = num_feat
        self.num_acts = num_acts
        self.alpha = alpha

        # used with training prediction model
        self.epoch = 1
        self.verbose = False    
        
        # create double hidden layer neural network with specified number of 
        # neurons use rectified linear unit activation function for first two 
        # layers use linear activation function for final action layer
        model = Sequential([
            Dense(units=num_neurons, input_dim=self.num_feat, activation='relu'),
            Dense(units=num_neurons, activation='relu'),
            Dense(units=self.num_acts, activation='linear')
        ])

        # compile model utilizing Adam optimization with the specified learning rate
        # - machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
        # for the loss function use Mean Squared Error
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        # set the prediction model and create a clone for target model
        self.model = model
        self.model_target = clone_model(model)
        self.model_target.set_weights(model.get_weights())

    def train(self, x, y):
        """
        Trains the prediction model using gradient descent with given values
        """
        self.model.fit(x, y, epochs=self.epoch, verbose=self.verbose)

    def predict(self, s):
        """
        Retrieves predictions for a given state from prediction model
        """
        return self.model.predict(s)

    def target_predict(self, s):
        """
        Retrieves predictions for a given state from target model
        """
        return self.model.predict(s)

    def update_target(self):
        """
        Updates the target model with the prediction model's weights
        """
        self.model_target.set_weights(self.model.get_weights())


class Memory:
    """
    Memory for Q-learner.  Moved from list to deque for performance:
    Stack Overflow: 19441488 - Efficiency of len() and pop() in Python
    """

    def __init__(self, capacity):
        self.samples = coll.deque(maxlen=capacity)

    def add(self, sample):
        """
        Adds a new sample to memory
        """
        self.samples.append(sample)

    def sample(self, n):
        """
        Retrieves a sample from memory at n size. If not enough samples
        exist returns None.
        """
        if n > len(self.samples):
            return None
        return rand.sample(list(self.samples), n)
