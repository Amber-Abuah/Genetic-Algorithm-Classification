import random
from GeneticNetwork import ActivationFunction
from GeneticAgent import Agent

# Three-way XOR classification task
inputs = [
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
    [0, 1, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 0, 1]
]
outputs = [1, 0, 1, 0, 0, 1, 0, 1]

agent_count_per_gen = 100 # Number of agents created for each generation
max_gen_count = 400 # Maximum number of generations
fittest_ratio = 0.1 # The top 10% of agents will be deemed the fittest and used as parents in the next generation
retain_fittest_ratio = 0.4 # Retain 40% of the fittest agents to be directly used in the next generation
retain_fittest_count = int(agent_count_per_gen * fittest_ratio * retain_fittest_ratio) # Integer number of fittest agents to directly use in next gen

# GenNet structure: num inputs -> num hidden neurons (layer 1) -> num hidden neurons (layer 2) -> num outputs
network_structure = [[3, 4, 4, 1], 
[ActivationFunction.NONE, ActivationFunction.RELU, ActivationFunction.RELU, ActivationFunction.SIGMOID]]

solution_found = False
gen_count = 0

while not solution_found and gen_count < max_gen_count:
    print("\nGeneration:", gen_count + 1)

    # Create agents for the new generation
    if gen_count == 0:
        agents = [Agent(network_structure) for _ in range(agent_count_per_gen)] # Create all agents from scratch
    else:
        agents = [fit_agents[i] for i in range(retain_fittest_count)] # Directly add top 40% of fittest agents from the last gen to current gen
        # Create new child agents from the parents of the fittest agents from last gen
        agents += [Agent(network_structure, fit_agents[random.randrange(0, len(fit_agents))], 
                         fit_agents[random.randrange(0, len(fit_agents))]) for _ in range(agent_count_per_gen - retain_fittest_count)]
        
    for a in agents:
        a.attempt_task(inputs, outputs)

    # Sort agents by fittest first
    agents.sort(reverse=True)
    fit_agents = agents[:retain_fittest_count]

    print("Fittest agents:")
    for f in fit_agents:
        print(f)
        if f.fitness == len(inputs):
            solution_found = True

    gen_count += 1

print("\nSolution found?", solution_found)
print("Best performing agent network structure:")
best_net = fit_agents[0].net
print(best_net)