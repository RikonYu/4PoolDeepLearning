from DragoonNet import DragoonNet
from DroneNet import DroneNet
from VultureNet import VultureNet
def getUnitClass(name,loading=False):
    print('name')
    if(name=='Protoss_Dragoon'):
        return DragoonNet(loading)
    if(name=='Zerg_Drone'):
        return DroneNet(loading)
    if(name=='Terran_Vulture'):
        return VultureNet(loading)
