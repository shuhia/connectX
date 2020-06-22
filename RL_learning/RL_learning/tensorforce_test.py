from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.environments import Environment
from kaggle_environments import make
from kaggle_environments.envs.connectx import connectx as ctx

import numpy as np



class CustomEnvironment(Environment):

    def __init__(self, enemy_agent):
        super().__init__()
        self.env = make("connectx", {'rows': 10, 'columns': 10, 'inarow': 5})
        self.trainer = self.env.train([None, enemy_agent])
        self.trainer.reset();


    def states(self):
        return dict(type='int', shape=(10*10), num_values=3)

    def actions(self):
        return dict(type='int', num_values=10)

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return 200

    # Optional
    def close(self):
        super().close()

    def reset(self):
        self.state = self.trainer.reset()['board'];

        return self.state

    def execute(self, actions):
        # print(actions)
        a = int(actions)
        raw = self.trainer.step(a)
        
        #print(raw)
        # print(self.env.render(mode='ansi'))
        next_state = raw[0]['board'];
        terminal = raw[2];
        reward = 0;
        
        if raw[1] != None: 
            reward = raw[1];
        return next_state, terminal, reward

from tensorforce.agents import Agent

env1 = CustomEnvironment('random');
env2 = CustomEnvironment('negamax');

agent = Agent.create(
    agent='tensorforce', environment=env1,update = 10,
    objective='policy_gradient', reward_estimation=dict(horizon=100), exploration = 0.5
)


runner1 = Runner(
    agent=agent,
    environment=env1,
    max_episode_timesteps=100
    
)

runner2 = Runner(
    agent=agent,
    environment=env2,
    max_episode_timesteps=100
    
)

runner1.run(num_episodes=2000, mean_horizon = 100)
runner2.run(num_episodes=200, mean_horizon = 10)
runner2.run(num_episodes=100, evaluation=True, mean_horizon = 100)
agent.save();
runner1.close()
runner2.close();



# Close agent and environment

print("done")