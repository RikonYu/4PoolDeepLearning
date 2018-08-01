import numpy
from Learners import Learner

class GeneticNet:
    def __init__(self):
        self.boundRad=numpy.random.uniform(15-5, 15+5)

class GeneticLearner(Learner):
    def __init__(self, population):
        self.