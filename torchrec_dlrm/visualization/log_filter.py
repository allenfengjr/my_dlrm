import json
import os
import sys
import string

class logFile:
    gpu_rank:int
    sort_rank:int
    time_stamp:int
    filename:str
    def __init__(self,filename) -> None:
        self.gpu_rank = int(filename.split('.')[0][4:])
        self.time_stamp = int(filename.split('.')[1][4:])
        self.filename = filename

def sort_rank():
    return None

l1 = logFile(filename="rank15.1673619674213.pt.trace.json")
