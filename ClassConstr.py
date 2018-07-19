from DragoonNet import DragoonNet
from DroneNet import DroneNet
from VultureNet import VultureNet
def getUnitClass(name,loading=False, output_func='linear'):
    print('name')
    if(name=='Protoss_Dragoon'):
        return DragoonNet(loading, output_func)
    if(name=='Zerg_Drone'):
        return DroneNet(loading, output_func)
    if(name=='Terran_Vulture'):
        return VultureNet(loading, output_func)
