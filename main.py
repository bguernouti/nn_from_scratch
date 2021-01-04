from core.perceptron import NInput, Perceptron
from helpers.data_builder import DataManager
import json
import random


if __name__ == "__main__":

    TRAIN_INPUTS_FILE: str = "train_data.json"
    TEST_INPUTS_FILE: str = "test_data.json"
    
    Manager = DataManager()
    Neuron: Perceptron = Perceptron(3, lr=0.01)

    """ 
    Delete train_data.json, test_data.json files or clear there content's 
    and uncomments this two lines to generate new data. 
    (compare_file=file_name) is used to prevent duplication while generating them
    """
    #Manager.random_data_writer(TRAIN_INPUTS_FILE, count=10000)
    #Manager.random_data_writer(TEST_INPUTS_FILE, count=1000, compare_file=TRAIN_INPUTS_FILE)
    
    trn_inputs: list = Manager.data_reader(TRAIN_INPUTS_FILE)
    tst_inputs: list = Manager.data_reader(TEST_INPUTS_FILE)

    # Training
    for train_input in trn_inputs:
        trn_data: list = train_input.get("values")
        trn_item: NInput = NInput(trn_data)
        
        Neuron.train(trn_item)

    ################################

    fail: int = 0
    # Testing
    for test_input in tst_inputs:
        tst_data: list = test_input.get("values")
        tst_item: NInput = NInput(tst_data)

        # Calculating accuracy
        if not tst_item.label == Neuron.guess(tst_item):
            fail += 1
    ##################################

    accuracy = 100 - (fail*100)/1000
    Neuron.accuracy = accuracy

    print("Successfully built a perceptron with accuracy = {}%".format(Neuron.accuracy))
    