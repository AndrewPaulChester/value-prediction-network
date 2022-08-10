""" wrapper for the taxi environment to provide compatibility with vpn framework"""

import numpy as np
import tensorflow as tf
import logging
import universe

from gym_craft.envs.craft_env import JsonCraftEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()


class CraftEnv(object):
    VALIDATION_MODE = 0

    def __init__(self, config="", verbose=1):
        self.env = JsonCraftEnv("screen","rooms","full",rewards={"base": 0, "failed-action": 0, "drop-off": 1})
        self.terminated = False
        self.env.seed(0)

        self.max_history = 1000
        self.reward_history = []
        self.length_history = []

        self.log_freq = 10
        self.log_t = 0
        self.verbose = verbose

        self.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space.image



    def observation(self):
        return self.env.observation_space.converter(self.obs)
        # obs=copy.deepcopy(self._map)
                
        # obs[self._pos_agent[0],self._pos_agent[1]]=0.5                
        # if(self._higher_dim_obs==True):
        #     "self._pos_agent"
        #     self._pos_agent
        #     obs=self.get_higher_dim_obs([self._pos_agent],[self._pos_goal])
            
        # return [obs]

    def reset(self, reset_episode=True, holes=None): 
        if reset_episode:
            self.t = 0
            self.episode_reward = 0
            self.last_step_reward = 0.0
            self.terminated = False

        self.obs = self.env.reset()
        return self.env.observation_space.converter(self.obs)

    # def remaining_time(self,normalized=True):
    #     return 0

    def last_reward(self):
        return self.last_step_reward

    def meta(self):
        return None

    def visualize(self):
        return self.env.render()
    
    def to_string(self):
        return self.obs

    def step(self, action):
        self.t +=1

        obs,reward,done,info = self.env.step(action)
        self.obs = obs
        self.terminated = done
        self.last_step_reward = reward
        self.episode_reward += reward


        to_log = None
        if self.terminated:
            if self.verbose > 0:
                logger.info('Episode terminating: episode_reward=%s episode_length=%s', 
                            self.episode_reward, self.t)
            self.log_episode(self.episode_reward, self.t)
            if self.log_t < self.log_freq:
                self.log_t += 1
            else:
                to_log = {}
                to_log["global/episode_reward"] = self.reward_mean(self.log_freq)
                to_log["global/episode_length"] = self.length_mean(self.log_freq)
                self.log_t = 0

        return self.env.observation_space.converter(self.obs), reward, self.terminated, to_log, 1

    def log_episode(self, reward, length):
        self.reward_history.insert(0, reward)
        self.length_history.insert(0, length)
        while len(self.reward_history) > self.max_history:
            self.reward_history.pop()
            self.length_history.pop()

    def reward_mean(self, num):
        return np.asarray(self.reward_history[:num]).mean()

    def length_mean(self, num):
        return np.asarray(self.length_history[:num]).mean()  

    def tf_visualize(self, x):
        return tf.zeros((1,1,1,1))
        #return tf.convert_to_tensor(self.env.observation_space.converter(self.obs))


    # def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
    #     pass

    # def inputDimensions(self):
    #     c,h,w= self.env.observation_space.image.shape
    #     return [(1,c,h,w)]
    #     # if(self._higher_dim_obs==True):
    #     #     return [(1,self._size_maze*6,self._size_maze*6)]
    #     # else:
    #     #     return [(1,self._size_maze,self._size_maze)]

    # def observationType(self, subject):
    #     return np.uint8

    # def nActions(self):
    #     return self.env.action_space.n

    
    # def get_higher_dim_obs(self,indices_agent,indices_reward): #called from observe
    #     pass

    # def inTerminalState(self):
    #     return self.done
