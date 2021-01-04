import random

class NInput(object):

    BIAS = 1 # Incredible change when adding it
    def __init__(self, values: list):

        if len(values) > 2: # Unfortunately this is the limit for this perceptron :)
            raise ValueError("A simple perceptron can accept only 2 inputs")

        self._values: list  = []
        for item in values:
            self._values.append(item)

        values.append(self.BIAS)
        self.data: list = values
        self.label: int = self.__label_rule()
        
    def __label_rule(self):
        """
        Classifier. 
        An equation based on the two inputs data[0], data[1] (x,y).
        Since x and y are generated randomly, and the perceptron is doing a "supervised learning",
        the inputs must be "labeled, or categorized", to create a "known dataset".
        While the input values are random float's numbers, so a mathematic equation becomes a good idea
        for making a rule to separate them.

        If this rule is edited, e.g (x > y, x <= 0, 2x < y, etc ...)
        train_data, and test_data must be regenerated
        """
        x = self.data[0]
        y = self.data[1]
        rule = 1 if (0.3 * x) + 0.4 > y else -1 # 0.3x + 0.4 > y
        return rule
                
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
        
        return 1 if _sum >= 0 else -1 # Activation function in this single line
    
    def train(self, n_input: NInput) -> tuple:

        guess: int = self.guess(n_input)
        err = n_input.label - guess

        for weight in self.weights: # Gradient decent
            w_idx = self.weights.index(weight)
            self.weights[w_idx] += err * n_input.data[w_idx] * self.lr
        
        return n_input.label, guess, self.weights

    @staticmethod
    def __randomize_weights(lenght: int) -> list:
        return [random.uniform(-1, 1) for i_ in range(lenght)]
