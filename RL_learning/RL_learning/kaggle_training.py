from kaggle_environments import make

from tensorforce import Agent, Environment
import kaggle_environments
from random import choice

from Agents import Agent_negamax
import strategy
agent1 = strategy.my_agent
agent2 =  Agent_negamax().negamaxAgent;

env = make("connectx", {'rows': 6, 'columns': 7, 'inarow': 4}, debug = True)
EMPTY = 0;
def random_agent(obs, config):
    return choice([c for c in range(config.columns) if obs.board[c] == EMPTY])

# Training agent in first position (player 1) against the default random agent.
trainer = env.train([None, 'negamax'])

obs = trainer.reset()
config = env.configuration;
agent_1_score = 0;
agent_2_score = 0;
tie = 0;
round = 1;
rounds = 100;
while(round < rounds):
    # Return a picture in ansi
    print(env.render(mode='ansi'))
    # Get action from agent 1

    action = agent2(obs,config)

    action = int(action)
    obs, reward, done, info = trainer.step(action)
    print(env.render(mode='ansi'), flush=True)
    print("round: {}".format(round))
    print("agent1: {}".format(agent_1_score))
    print("agent2: {}".format(agent_2_score))
    print("ties: {}".format(tie))
    if done:
        round += 1;
        if(reward > 0):
            agent_1_score += 1
        elif(reward <0):
            agent_2_score += 1
        else:
            tie += 1
        obs = trainer.reset()
    
