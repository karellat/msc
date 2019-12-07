import time
import numpy as np

class Timer: 
    def __init__(self, msg="", print_fnc=print): 
        self.msg = msg
        self.print_fnc = print_fnc

    def __enter__(self):
        self.f = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.print_fnc(self.msg + str(time.perf_counter() - self.f))
        
def max_class_accuracy(labels): 
    unique, counts = np.unique(labels, return_counts=True)
    return unique[np.argmax(counts)], np.max(counts)/labels.shape[0]
  