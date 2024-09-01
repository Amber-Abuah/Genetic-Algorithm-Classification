from GeneticNetwork import Network

class Agent:
    def __init__(self, network_structure, parent1 = None, parent2 = None):
        self.net = Network(network_structure) if parent1 is None else Network.create_child_network(parent1.net, parent2.net)
        self.fitness = 0

    def attempt_task(self, inputs, outputs):
        self.fitness = 0

        for i in range(len(inputs)):
            # Output classification is 1 if network output >= 0.5, otherwise 0
            guess = 1 if self.net.forward_pass(inputs[i])[0] >= 0.5 else 0 

            if guess == outputs[i]:
                self.fitness += 1

    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __str__(self) -> str:
        return "Agent with fitness: " + str(self.fitness)