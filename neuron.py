import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Neuron(object):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.value = None
    
    def input(self, values):
        # sigmoid(w1*v1 + w2*v2 + ... + wn*vn + bias)
        self.value = sigmoid(sum([w*v for w, v in zip(self.weights, values)])+self.bias)
        return self.value

    def __str__(self):
        return "{{weights:{},bias:{}}}".format(self.weights, self.bias)


class Brain(object):
    def __init__(self, layers):
        self.sizes = layers
        self.layers = []
    
    def init_layers(self):        
        for i in range(len(self.sizes)):
            self.layers.append([Neuron([np.random.random_sample() for k in range(self.sizes[i-1])], np.random.random_sample()) for j in range(self.sizes[i])])
        self.layers[0] = [Neuron([0], 0) for i in range(self.sizes[0])]
    
    def input(self, value_list):
        for i in range(self.sizes[0]):
            self.layers[0][i].value = value_list[i]
        for i in range(1,len(self.sizes)):
            for neuron in self.layers[i]:
                neuron.input(Brain.get_neurons_value(self.layers[i-1]))
                print(neuron.value)
        return Brain.get_neurons_value(self.layers[-1])

    @staticmethod
    def get_neurons_value(layer):
        return [neuron.value for neuron in layer]

    
    def __str__(self):
        s = "{\n"
        for l in self.layers:
            s += "\t"
            for n in l:
                s += str(n) + " "
            s += "\n"
        s += "}"
        return s

if __name__ == '__main__':
    b = Brain([3, 2, 1])
    b.init_layers()
    print(str(b))
    print(b.input([0.5,0.0,0.9]))

