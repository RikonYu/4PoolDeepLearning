import numpy
from pybrood import game
import pybrood
class gameTask:
    def __init__(self, name, valueFunc, units, maxFrame, finalValue):
        self.name = name
        self.valueFunc = valueFunc
        self.unitTypes = units
        self.finalValueFunc=finalValue
        self.maxFrame = maxFrame
    def can_control(self, unit, playerMe):
        if(unit.getType() in self.unitTypes and unit.getPlayer()==playerMe):
            return True
        return False
    def is_terminal(self, unit):
        if(unit.exists()==False or game.getFrameCount()>=self.maxFrame):
            return True
        return False



def dragoonDefusalValue(unit):
    ans=0
    for u in game.getAllUnits():
        if(u.getType()==pybrood.UnitTypes.Terran_Vulture_Spider_Mine):
            ans+=1
    return unit.getKillCount()-ans*0.2
def dragoonDefusalFinalValue(player):
    return player.minerals()

def findEnemyBaseFinalValue(player):
    for u in game.getAllUnits():
        if(u.getType().getName() in ['Protoss_Nexus']):
                return 0
    loc=game.getStartLocations()
    myLoc=player.getStartLocation()
    ans=numpy.inf
    for i in loc:
        for u in game.getAllUnits():
            if(u.getType().getName()=='Zerg_Drone'):
                if(i!=myLoc and numpy.linalg.norm(numpy.array(u.getPosition())-i)<ans):
                    ans=numpy.linalg.norm(numpy.array(myLoc)-i)
    return -ans
def findEnemyBaseValue(unit):

    for u in game.getAllUnits():
        if(u.getType().getName() in ['Protoss_Nexus']):
            if(u.getPlayer()!=unit.getPlayer()):
                return 'found'
    loc=game.getStartLocations()
    myLoc=unit.getPlayer().getStartLocation()
    ans=numpy.inf
    for i in loc:
        if(i!=myLoc and numpy.linalg.norm(numpy.array(myLoc)-i)<ans):
            ans=numpy.linalg.norm(numpy.array(myLoc)-i)
    return -ans


taskDragoonDefuse=gameTask('DragoonDefusal', dragoonDefusalValue, [pybrood.UnitTypes.Protoss_Dragoon], 15000, dragoonDefusalFinalValue)
taskBaseScout=gameTask( 'findEnemyBase', findEnemyBaseValue, [pybrood.UnitTypes.Zerg_Drone], 2500,findEnemyBaseFinalValue)

