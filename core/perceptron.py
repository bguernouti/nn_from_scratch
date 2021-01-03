import random

class NInput(object):
    BIAS = 1 # Incredible change when adding it
    def __init__(self, data: list):
        self._values: list  = []
        for item in data:
            self._values.append(item)
        self.data: list = data
        self.data.append(self.BIAS)
        self.label: int = self.__data_rules()
        
    def __data_rules(self):
        return 1 if (0.3 * self.data[0]) + 0.4 > self.data[1] else -1
                
class Perceptron(object):
            
    def __init__(self, input_size: int, lr: float):

        self._input_size: int = input_size
        self.weights: list = self.__randomize_weights(input_size)
        self.lr: float = lr
        self.accuracy: float = 0

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
            self.weights[w_idx] += err * n_input.data[w_idx] * self.lr
        
        return n_input.label, guess, self.weights

    @staticmethod
    def __randomize_weights(lenght: int) -> list:
        return [random.uniform(-1,1) for i_ in range(lenght)]
