import os 
import os
import itertools

# Define the tuple with four elements
elements = [True, False]
combinations = list(itertools.product(elements, repeat=4))

print(len(combinations))