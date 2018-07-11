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
def dragoonDefusalFinalValue():
    for u in game.getAllUnits():
        if(u.getType()==pybrood.UnitTypes.Terran_Vulture_Spider_Mine):
            ans+=1
    return ans

def findEnemyBaseFinalValue():
    for u in game.getAllUnits():
        if(u.getType().getName() in ['Protoss_Nexus']):
                return 1
    return 0

def findEnemyBaseValue(unit):
    for u in game.getAllUnits():
        if(u.getType().getName() in ['Protoss_Nexus', 'Terran_Command_Center', 'Zerg_Hatchery', 'Zerg_Hive', 'Zerg_Lair¶']):
            if(u.getPlayer()!=unit.getPlayer()):
                return 1
    return 0


taskDragoonDefuse=gameTask('DragoonDefusal', dragoonDefusalValue, [pybrood.UnitTypes.Protoss_Dragoon], 15000, dragoonDefusalFinalValue)
taskBaseScout=gameTask( 'findEnemyBase', findEnemyBaseValue, [pybrood.UnitTypes.Zerg_Drone], 25,findEnemyBaseFinalValue)

