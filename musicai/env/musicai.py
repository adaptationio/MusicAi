
import numpy as np
from .state import *

class MusicAI():
    """Base class for Unity ML environments using unityagents (v0.4)."""

    def __init__(self, name='trader', seed=0):
        self.seed = seed
        print('SEED: {}'.format(self.seed))
        self.start = 0
        self.state = 0
        self.reward = 0
        self.done = False
        


    def reset(self):
        """Reset the environment."""
        #info = self.env.reset(train_mode=True)[self.brain_name]
        self.state
        return self.state

    def step(self, action):
        """Take a step in the environment."""
        #self.placement = self.env.placement
        self.state, self.reward, self.done
        #print(done)
        if self.done:
            self.start += 1

        return self.state, self.reward, self.done

    def render(self):
        """
        Render the environment to visualize the agent interacting.
        Does nothing because rendering is always on as is required by linux environments.
        """
        print("Render")

