# Genetic Algorithm Classification Task
Genetic algorithms are applied to agents to solve a classification task of solving the three-way XOR problem.

The genetic algorithms used are as follows:
- Each agents 'genes' are their weights and biases for their internal Genetic networks.
- Each agent has a fitness function determining how good they are at performing the classification task.
- Agents with the highest fitness scores are used as parents for next generations agents.
- Child agents either inherit their first parent, second parent or a random gene for each weight/ bias.
- A small number of the fittest agents directly partake in the next generation additionally.

**GeneticNetwork.py**: Contains _ActivationFunction_, _Neuron_, _Layer_ and _Network_ classes. These classes are built on top of each other to create a Neural Network from scratch.  
**GeneticAgent.py**: Contains the _Agent_ class. Agents contain their own genetic networks and record their own fitness after performing a task.  
**GeneticClassification.py**: Applies genetic algorithms to agents, aiming to solve the three way XOR classification problem.  