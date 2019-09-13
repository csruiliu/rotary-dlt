import numpy as np

from random import random
from math import log, ceil
from time import time, ctime

class Hyperband:
	def __init__(self, get_params_function, try_params_function):

