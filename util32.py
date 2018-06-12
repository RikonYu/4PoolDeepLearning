import numpy
import os
from pybrood import BaseAI, run, game
import pybrood
import win32api
import win32con
import time
import subprocess
import shutil
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
    if(abs(up[0]-mp[0])<180 and abs(up[1]-mp[1])<180):
        return True
    return False
#idle, move, build, gather, attack, returnCargoo
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
            if(u.getType().getName() in['Resource_Mineral_Field','Resource_Vespene_Geyser']):
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
def game2stateDrone(me):
    global terrain, hground
    ans=numpy.zeros([360,360,18])
    x,y=me.getPosition()
    ax=max(0,180-x)
    ay=max(0,180-y)
    ans[:,:,16]=me.isCarryingMinerals()
    ans[:,:,17]=me.isCarryingGas()
    ans[:,:,11]=me.getHitPoints()
    #print(max(0,x-180),min(x+180,game.mapHeight()*32),max(0,y-180),min(y+180,game.mapWidth()*32))
    ans[ax:min(360,game.mapHeight()*32-x+180),
        ay:min(360,game.mapWidth()*32-y+180),10]=hground[max(0,x-180):min(x+180,game.mapHeight()*32),max(0,y-180):min(y+180,game.mapWidth()*32)]

    ans[ax:min(360,game.mapHeight()*32-x+180),
        ay:min(360,game.mapWidth()*32-y+180),0]=terrain[max(0,x-180):min(x+180,game.mapHeight()*32),max(0,y-180):min(y+180,game.mapWidth()*32)]
    for u in game.getAllUnits():
        if(inReach(u,me)):
            if(u.getType().getName()=='Resource_Mineral_Field'):
                ans[me.getPosition()[0]-u.getPosition()[0]+180,
                    me.getPosition()[1]-u.getPosition()[1]+180,7]=1
            elif(u.getType().getName()=='Resource_Vespene_Geyser'):
                ans[me.getPosition()[0]-u.getPosition()[0]+180,
                    me.getPosition()[1]-u.getPosition()[1]+180,8]=1
            elif(u.getPlayer()==me.getPlayer()):
                ans[me.getPosition()[0]-u.getPosition()[0]+180,
                    me.getPosition()[1]-u.getPosition()[1]+180,12]=u.getHitPoints()
                ans[me.getPosition()[0]-u.getPosition()[0]+180,
                    me.getPosition()[1]-u.getPosition()[1]+180,14]=game.getDamageTo(me.getType(),u.getType(),me.getPlayer(),u.getPlayer())
                ans[me.getPosition()[0]-u.getPosition()[0]+180,
                    me.getPosition()[1]-u.getPosition()[1]+180,15]=game.getDamageFrom(me.getType(),u.getType(),me.getPlayer(),u.getPlayer())     
                if(u.getType().getName()=='Zerg_Extractor'):
                    ans[me.getPosition()[0]-u.getPosition()[0]+180,
                        me.getPosition()[1]-u.getPosition()[1]+180,9]=1
                elif(u.getType().isFlyer()==True):
                    ans[me.getPosition()[0]-u.getPosition()[0]+180,
                        me.getPosition()[1]-u.getPosition()[1]+180,2]=1
                elif(u.getType().isBuilding()==True):
                    ans[me.getPosition()[0]-u.getPosition()[0]+180,
                            me.getPosition()[1]-u.getPosition()[1]+180,3]=1
                else:
                    ans[me.getPosition()[0]-u.getPosition()[0]+180,
                            me.getPosition()[1]-u.getPosition()[1]+180,1]=1
            else:
                ans[me.getPosition()[0]-u.getPosition()[0]+180,
                    me.getPosition()[1]-u.getPosition()[1]+180,13]=u.getHitPoints()
                if(u.getType().isFlyer()==True):
                    ans[me.getPosition()[0]-u.getPosition()[0]+180,
                        me.getPosition()[1]-u.getPosition()[1]+180,5]=1
                elif(u.getType().isBuilding()==True):
                    ans[me.getPosition()[0]-u.getPosition()[0]+180,
                        me.getPosition()[1]-u.getPosition()[1]+180,6]=1
                else:
                    ans[me.getPosition()[0]-u.getPosition()[0]+180,
                        me.getPosition()[1]-u.getPosition()[1]+180,4]=1

    return ans
def command(unit,order):
    coord=unit.getPosition()
    coord[0]+=order[0]
    coord[1]+=order[1]
    lcmd=unit.getLastCommand()
    #print(lcmd,lcmd.getX(),lcmd.getY())
    if(lcmd.getTargetPosition()==coord and type2cmd(lcmd.getType())==order[2]):
        return
    if(order[2]==0):
        unit.holdPosition()
    elif(order[2]==1):
        unit.move(coord)
    elif(order[2]==2):
        unit.build(pybrood.UnitTypes.Zerg_Spawning_Pool, coord)
    elif(order[2]==3):
        unit.gather(game.getClosestUnit(coord))
    elif(order[2]==4):
        unit.attack(game.getClosestUnit(coord))
    elif(order[2]==5):
        unit.returnCargo()
