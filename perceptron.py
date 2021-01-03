from exceptions import LengthError
import random

class NInput(object):
    def __init__(self, data: list):
        data.append(1)
        self.data: list = data
        self.label: int = self.__data_rules()
        
    def __data_rules(self):
        return 1 if (0.3 * self.data[0]) + 0.4 > self.data[1] else -1
                
class Perceptron(object):
            
    def __init__(self, input_size: int):

        self._input_size: int = input_size
        self.weights = self.__randomize_weights(input_size)

    def guess(self, _input: NInput) -> int:

        if not len(_input.data) == self._input_size:
            raise ValueError("Invalid input")

        _sum: int = 0
        for value in _input.data:
            idx = _input.data.index(value)
            _sum += value * self.weights[idx]
        
        return 1 if _sum >= 0 else -1 # activation function in this single line
    
    def train(self, n_input: NInput) -> tuple:

        guess: int = self.guess(n_input)
        err = n_input.label - guess

        for weight in self.weights: # Gradient decent
            w_idx = self.weights.index(weight)
            self.weights[w_idx] += err * n_input.data[w_idx] * 0.001
        
        return n_input.label, guess, self.weights

    @staticmethod
    def __randomize_weights(lenght: int) -> list:
        return [random.uniform(-1,1) for i_ in range(lenght)]

if __name__ == "__main__":
    
    Neuron = Perceptron(3)
    training_inputs: list = []
    testing_inputs: list = []
    fail = 0

    for _ in range(10000): # Training data
        data: list = [random.uniform(-1,1) for _ in range(2)]
        training_inputs.append(NInput(data))
    
    for _ in range(1000): # Testing data
        data: list = [random.uniform(-1,1) for _ in range(2)]
        testing_inputs.append(NInput(data))
    
    for _ in range(5):
        for train_input in training_inputs:
            label, guess, weights = Neuron.train(train_input)
            #fail += 1 if not real_guess == train_input.label else 0
            #print("{} | label {} | guess {} | weights {} // {}".format(
            #    train_input.data, 
            #    label, 
            #    guess, 
            #    weights,
            #    3*train_input.data[0]+2
            #    )
            #)
   
    for test_input in testing_inputs:
        if not test_input.label == Neuron.guess(test_input):
            fail += 1
    
    print(100 - (fail*100)/1000)
    