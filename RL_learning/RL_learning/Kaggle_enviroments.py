from kaggle_environments import make

from tensorforce import Agent, Environment
import kaggle_environments
from random import choice



env = make("connectx", {'rows': 10, 'columns': 10, 'inarow': 5})
EMPTY = 0;
def random_agent(obs, config):
    return choice([c for c in range(config.columns) if obs.board[c] == EMPTY])

# Training agent in first position (player 1) against the default random agent.
trainer = env.train([None, 'negamax'])

agent = Agent.load();

obs = trainer.reset()
config = env.configuration;
for _ in range(100):
    # Return a picture in ansi
    print(env.render(mode='ansi'))
    # Get action from agent 1

    action = agent.act(obs['board'], evaluation= True)

    action = int(action)
    obs, reward, done, info = trainer.step(action)
    
    print(env.render(mode='ansi'))

    if done:
        obs = trainer.reset()

