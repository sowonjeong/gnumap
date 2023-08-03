import sys
import os
from pprint import pprint


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
pprint(parent)
pprint(sys.path)

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../')

print("-----------------------")
print("After appending a path")
print("-----------------------")
pprint(sys.path)