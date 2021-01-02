from exceptions import LengthError
import random


class Input(object):

    def __init__(self, data: list):
        self.data = data


class NInput(object):

        def __init__(self, data: list, label: int):

            self.data: list = data
            self.label: int = label
            self.weights: list = self.__randomize_weights(len(data))
        
        @staticmethod
        def __randomize_weights(lenght: int) -> list:
            return [random.uniform(-1,1) for i_ in range(lenght)]


class Perceptron(object):
            
    def __init__(self, input_size: int):

        self._input_size: int = input_size

    def guess(self, _input: NInput) -> int:

        if not len(_input.data) == self._input_size:
            raise ValueError("Invalid input")

        _sum: int = 0
        for inpt in _input.data:
            idx = _input.data.index(inpt)
            _sum += inpt * _input.weights[idx]
        
        return 1 if _sum > 0 else -1
    
    def train(self, inputs: list):
        
        for inpt in inputs:
            if type(inpt) is not NInput:
                raise TypeError("Invalid input type")

            while not self.guess(inpt) == inpt.label:
                continue # Back Probagation to Correct weights


if __name__ == "__main__":

    training_inputs: list = []
    for _ in range(100):
        data: list = [random.uniform(-1,1) for _ in range(2)]
        lbl = -1 if random.randint(0,1) == 0 else 1
        training_inputs.append(NInput(data, lbl))
    
    percept = Perceptron(2)
    print(percept.train(training_inputs))
