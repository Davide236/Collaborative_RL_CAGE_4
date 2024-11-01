from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, DiscoveryFSRed
from CybORG.Agents.Wrappers.VisualiseRedExpansion import VisualiseRedExpansion

steps = 1
sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, 
                                green_agent_class=EnterpriseGreenAgent, 
                                red_agent_class=DiscoveryFSRed,
                                steps=steps)
cyborg = CybORG(scenario_generator=sg, seed=7629)

cyborg.reset()
visualise = VisualiseRedExpansion(cyborg, steps)

for i in range(steps):
    # Whatever you want to do before each step
    cyborg.step()
    # Whatever you want to do after each step

    # Make a record of the environment state for the visualisation
    visualise.visualise_step()

# Visualise the episode
visualise.show_graph()