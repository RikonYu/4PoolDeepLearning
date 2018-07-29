import numpy
from pybrood import game
import pybrood

def dist(p1,p2):
    return numpy.linalg.norm(numpy.array(p1)-p2)

class gameTask:
    def __init__(self, name, valueFunc, units, maxFrame, finalValue, frameSkip):
        self.name = name
        self.valueFunc = valueFunc
        self.unitTypes = units
        self.finalValueFunc=finalValue
        self.maxFrame = maxFrame
        self.frameSkip=frameSkip
    def can_control(self, unit, playerMe):
        if(unit.getType() in self.unitTypes and unit.getPlayer()==playerMe and unit.getPosition()[0]<=game.mapHeight()*32 and unit.getPosition()[1]<=game.mapWidth()*32):
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
                return 0
    loc=game.getStartLocations()
    myLoc=unit.getPlayer().getStartLocation()
    ans=numpy.inf
    for i in loc:
        if(i!=myLoc and numpy.linalg.norm(numpy.array(myLoc)-i)<ans):
            ans=numpy.linalg.norm(numpy.array(myLoc)-i)
    return -ans

def findGasValue(unit):
    #gasPos=[11111,11111]
    ans=numpy.inf
    for u in game.getAllUnits():
        if(u.getType()==pybrood.UnitTypes.Resource_Vespene_Geyser):
            if(dist(u.getPosition(),unit.getPosition())<ans):
                ans=dist(u.getPosition(),unit.getPosition())
    return -ans/(pybrood.UnitTypes.Zerg_Drone.topSpeed()*10)

def findGasFinalValue(player):
    ans=0
    for u in player.getUnits():
        if(u.getType()==pybrood.UnitTypes.Zerg_Drone):
            ans+=findGasValue(u)
    return ans

def VultureKiteValue(unit):
    return unit.getKillCount() + unit.getHitPoints()*0.2

def VultureKiteFinalValue(player):
    return game.getFrameCount()


taskDragoonDefuse=gameTask('DragoonDefusal', dragoonDefusalValue, [pybrood.UnitTypes.Protoss_Dragoon], 15000, dragoonDefusalFinalValue, 10)
taskBaseScout=gameTask( 'findEnemyBase', findEnemyBaseValue, [pybrood.UnitTypes.Zerg_Drone], 2500,findEnemyBaseFinalValue, 10)
taskDebug=gameTask('findGas',findGasValue,[pybrood.UnitTypes.Zerg_Drone],150,findGasFinalValue, 10)
taskVultureKite=gameTask('VultureKite', VultureKiteValue, [pybrood.UnitTypes.Terran_Vulture], numpy.inf, VultureKiteFinalValue, 5)