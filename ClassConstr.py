from DragoonNet import DragoonNet
from DroneNet import DroneNet
def getUnitClass(name,loading=False):
    print('name')
    if(name=='Protoss_Dragoon'):
        return DragoonNet(loading)
    if(name=='Zerg_Drone'):
        return DroneNet(loading)

