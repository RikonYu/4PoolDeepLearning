import numpy
from Learners import Learner

class GeneticNet:
    def __init__(self):
        self.zerglingEff=numpy.random.random()*2-1.0
        self.zerglingRng=numpy.random.random()
class GeneticLearner(Learner):
    def __init__(self, population):
        self.