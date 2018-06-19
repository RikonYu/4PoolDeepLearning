import numpy
import os
from pybrood import BaseAI, run, game
import pybrood
import win32api
import win32con
import time
import subprocess
import shutil
from consts import WINDOW_SIZE
terrain=numpy.zeros([1,1])
hground=numpy.zeros([1,1])

def init():
    global terrain,hground
    terrain=numpy.zeros([game.mapHeight()*32,game.mapWidth()*32])
    hground=numpy.zeros([game.mapHeight()*32,game.mapWidth()*32])
    for i in game.getAllRegions():
        terrain[i.getBoundsTop():i.getBoundsBottom(),i.getBoundsLeft():i.getBoundsRight()]=i.isAccessible()
        hground[i.getBoundsTop():i.getBoundsBottom(),i.getBoundsLeft():i.getBoundsRight()]=i.isHigherGround()



def navigate():
    subprocess.Popen(['python','navigator.py'])
def inReach(unit,me):
    up=unit.getPosition()
    mp=me.getPosition()
    if(abs(up[0]-mp[0])<WINDOW_SIZE//2 and abs(up[1]-mp[1])<WINDOW_SIZE//2):
        return True
    return False
#0idle, 1move, 2build, 3gather, 4attack, 5returnCargoo
def ord2cmd(order):
    if(order==6):
        return 1
    if(order==30):
        return 2
    if(order in [85,81]):
        return 3
    if(order==10):
        return 4
    if(order in [90,84]):
        return 5
    return 0
def type2cmd(cmdType):
    if(cmdType.getName() in ['Attack_Move', 'Attack_Unit']):
        return 4
    if(cmdType.getName()== 'Gather'):
        return 3
    if(cmdType.getName() in ['Move','Follow']):
        return 1
    if(cmdType.getName()=='Build'):
        return 2
    if(cmdType.getName()=='Return_Cargo'):
        return 5
    if(cmdType.getName() in ['Stop', 'Hold_Position']):
        return 0
    raise Exception
    
#0terrain,1friendly ground,2friendly air,3friendly building,4enemy ground, 5enemy air,
#6enemy building, 7mineral, 8naked gas, 9geyser, 10highground, 11self hp, 12ally hp,
#13enemy hp, 14dmg dealt, 15dmg receive, 16has mineral, 17has gas
def reg2msg():
    ans=[[(game.mapHeight(),game.mapWidth())]]
    ans.append([(i.getBoundsTop(),i.getBoundsBottom(),i.getBoundsLeft(),i.getBoundsRight(),i.isAccessible(),i.isHigherGround()) for i in game.getAllRegions()])
    return ans
#0myPos,1[myHp,hasMineral,hasGas, groudWeaponCooldown],2enemies[coordinate, HP,isFlyer,isBuilding, dmgTo,dmgFrom],3Ally[coordinate, Hp, isFlyer, isBuilding]
#4resource[isMineral,coord], 5my_extractor[coord]
def game2msgDrone(me):
    ans=[]
    ans.append(me.getPosition())
    ans.append((me.getHitPoints(),me.isCarryingMinerals(),me.isCarryingGas(),me.getGroundWeaponCooldown()))
    enemy=[]
    ally=[]
    resource=[]
    extra=[]
    for u in game.getAllUnits():
        if(inReach(u,me)):
            coor=u.getPosition()
            tu=u.getType()
            #print(tu.getName())
            if(u.getType().getName() in ['Resource_Mineral_Field','Resource_Vespene_Geyser']):
                #print('mineral %s'%str(coor))
                resource.append((u.getType().getName()=='Resource_Mineral_Field',coor))
            elif(u.getType().getName()=='Zerg_Extractor' and u.getPlayer()==me.getPlayer()):
                extra.append(coor)
            elif(u.getPlayer()==me.getPlayer()):
                ally.append((coor,u.getHitPoints(),tu.isFlyer(),tu.isBuilding()))
            else:
                enemy.append((coor,u.getHitPoints(),tu.isFlyer(),tu.isBuilding(),game.getDamageTo(me.getType(),u.getType(),me.getPlayer(),u.getPlayer()),
                                                                                game.getDamageFrom(me.getType(),u.getType(),me.getPlayer(),u.getPlayer())))
    ans.append(enemy)
    ans.append(ally)
    ans.append(resource)
    ans.append(extra)
    return ans
    return ans
def command(unit,order):
    coord=unit.getPosition()
    coord[0]+=order[0]
    coord[1]+=order[1]
    lcmd=unit.getLastCommand()
    print('cmd',lcmd.getType(),lcmd.getTargetPosition())
    if(lcmd.getTargetPosition()==coord and type2cmd(lcmd.getType())==order[2]):
        return
    if(order[2]==0):
        unit.holdPosition()
    elif(order[2]==1):
        unit.move(coord)
    elif(order[2]==2):
        unit.build(pybrood.UnitTypes.Zerg_Spawning_Pool, coord)
    elif(order[2]==3):
        print('gathering ',unit.getPosition(),game.getClosestUnit(coord).getPosition())
        unit.gather(game.getClosestUnit(coord))
    elif(order[2]==4):
        print('attacking ',unit.getPosition(),game.getClosestUnit(coord).getPosition())
        unit.attack(game.getClosestUnit(coord))
    elif(order[2]==5):
        unit.returnCargo()
