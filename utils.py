from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod    
    def train(self, train_set):
        pass
    
    @abstractmethod 
    def save_to_pickle(self, filename):
        pass
    
    @abstractmethod 
    def load_from_pickle(self, filename):
        pass
    
    @abstractmethod 
    def generate_password(self, n): # n is used in Markov as n-gram param, might not be needed for RNN
        pass

    @abstractmethod 
    def generate_passwords(self, count):
        pass



