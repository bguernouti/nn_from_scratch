import random
import json

from core.perceptron import NInput


class DataManager(object):

    """
    Data manager for the perceptron. 
    It creates training and testing datasets based on the NInput class.
    Prevent inputs duplication, also allow to read the data after has been created.
    """

    def __write(self, data: list, file: str) -> None:

        _structured: list = []

        for item in data:
                
            if type(item) is not NInput:
                raise TypeError("Invalid input type")

            item: dict = {
                "label": item.label,
                "values" : item._values
            }
            _structured.append(item)

        with open(file, "w+") as f:

            json.dump(_structured, f)
            f.close()

    def random_data_writer(self, _file: str, count: int, compare_file: str = None) -> None:

        for_compare = self.data_reader(compare_file) if compare_file else None

        _all: list = []
        
        for _ in range(count):

            inpt: NInput = self.__random_single()

            if for_compare:
                item = {
                    "values": inpt._values,
                    "label": inpt.label
                }
                if item in for_compare:
                    print("We got a duplicated item")
                    continue
            
            _all.append(inpt)
        self.__write(_all, file=_file)

    @staticmethod
    def __random_single() -> NInput:
        inpt_data: list =  [random.uniform(-1,1) for _ in range(2)]
        return NInput(inpt_data)
    
    @staticmethod
    def data_reader(_file: str) -> list:

        with open(_file, "r") as f:
            data = json.load(f)
        
        return data
