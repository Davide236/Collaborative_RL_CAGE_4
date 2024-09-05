from pprint import pprint
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent
from CybORG.Simulator.Actions import Sleep
from CybORG.Agents.Wrappers import BlueFlatWrapper
steps = 1000
sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, 
                                green_agent_class=EnterpriseGreenAgent, 
                                red_agent_class=SleepAgent,
                                steps=steps)
cyborg = CybORG(scenario_generator=sg, seed=1234)
agent = 'blue_agent_6'
reset = cyborg.reset(agent=agent)
initial_obs = reset.observation

print(f"\n{agent}: Initial Observation")
print("\nKeys Only: \n")
pprint(initial_obs.keys())

env = BlueFlatWrapper(env=cyborg)
env.reset()
action_space = env.action_labels(agent)
pprint(f'\nAction Space of Agent: {action_space}\n')