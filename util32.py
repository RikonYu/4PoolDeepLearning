import numpy
import os
from pybrood import BaseAI, run, game
import pybrood
import win32api
import win32con
import time
import subprocess
import struct
import shutil
import random
from consts import WINDOW_SIZE

terrain = numpy.zeros([1, 1])
hground = numpy.zeros([1, 1])

def collide(pos,unit):
    if(unit.getLeft() < pos[1] and pos[1]< unit.getRight() and unit.getTop() < pos[0] and pos[0]<unit.getBottom()):
        return True
    return False
def init():
    global terrain, hground
    terrain = numpy.zeros([game.mapHeight() * 32, game.mapWidth() * 32])
    hground = numpy.zeros([game.mapHeight() * 32, game.mapWidth() * 32])
    for i in game.getAllRegions():
        terrain[i.getBoundsTop():i.getBoundsBottom(), i.getBoundsLeft():i.getBoundsRight()] = i.isAccessible()
        hground[i.getBoundsTop():i.getBoundsBottom(), i.getBoundsLeft():i.getBoundsRight()] = i.isHigherGround()


def navigate():
    subprocess.Popen(['python', 'navigator.py'])


def inReach(unit, me):
    up = unit.getPosition()
    mp = me.getPosition()
    if (abs(up[0] - mp[0]) < WINDOW_SIZE // 2 and abs(up[1] - mp[1]) < WINDOW_SIZE // 2):
        return True
    return False


# Y:0idle, 1move, 2build, 3gather, 4attack, 5returnCargoo
def ord2cmd(order):
    if (order == 6):
        return 1
    if (order == 30):
        return 2
    if (order in [85, 81]):
        return 3
    if (order == 10):
        return 4
    if (order in [90, 84]):
        return 5
    return 0


def type2cmd(cmdType):
    if (cmdType.getName() in ['Attack_Move', 'Attack_Unit']):
        return 4
    if (cmdType.getName() == 'Gather'):
        return 3
    if (cmdType.getName() in ['Move', 'Follow']):
        return 1
    if (cmdType.getName() == 'Build'):
        return 2
    if (cmdType.getName() == 'Return_Cargo'):
        return 5
    if (cmdType.getName() in ['Stop', 'Hold_Position']):
        return 0
    raise Exception


# Drones:
# X:
# 0terrain,1friendly ground,2friendly air,3friendly building,4enemy ground, 5enemy air,
# 6enemy building, 7mineral, 8naked gas, 9geyser, 10highground, 11self hp, 12ally hp,
# 13enemy hp, 14dmg dealt, 15dmg receive, 16has mineral, 17has gas
# Dragoons:
# 0terrain,1friendly ground,2enemy ground,3selfhp+sheild, 4allyhp+shield, 5enemyhp, 6dmg dealt, 7dmg redeive
def reg2msg():
    # ans=[[(game.mapHeight(),game.mapWidth())]]
    # ans.append([(i.getBoundsTop(),i.getBoundsBottom(),i.getBoundsLeft(),i.getBoundsRight(),i.isAccessible(),i.isHigherGround()) for i in game.getAllRegions()])
    ans = numpy.zeros([game.mapHeight() * 4, game.mapWidth() * 4])
    for i in range(game.mapHeight() * 4):
        for j in range(game.mapWidth() * 4):
            ans[i, j] = game.isWalkable([i, j])
    return ans


# msg:
# 0myPos,1[myHp,hasMineral/killCount,hasGas, groudWeaponCooldown],
# 2enemies[coordinate, HP,isFlyer,isBuilding, dmgTo,dmgFrom，(top,bot,left,right).(gminRange,gmaxRange,aminRange,amaxrange)],
# 3Ally[coordinate, Hp, isFlyer, isBuilding,(top,bot,left,right)]
# 4resource[isMineral,coord,(top,bot,left,right)],
# 5my_extractor[coord]
# 6 minimap explored
def game2msg(me):
    ans = []
    typeMe = me.getType()
    enemyCount=0
    for u in game.getAllUnits():
        if(u.getType()==pybrood.UnitTypes.Terran_Vulture_Spider_Mine):
            enemyCount+=1
    ans.append(me.getPosition())
    ans.append((me.getHitPoints() + me.getShields(),
                me.getKillCount(),
                me.isCarryingGas(),
                (typeMe.groundWeapon() != pybrood.WeaponTypes.None_, me.getGroundWeaponCooldown(),
                 typeMe.groundWeapon().maxRange()),
                (typeMe.airWeapon() != pybrood.WeaponTypes.None_, me.getAirWeaponCooldown(),
                 typeMe.airWeapon().maxRange()),
                enemyCount
                ))
    enemy = []
    ally = []
    resource = []
    extra = []
    for u in game.getAllUnits():
        if (inReach(u, me) and u.getID()!=me.getID()):
            coor = u.getPosition()
            tu = u.getType()
            # print(tu.getName())
            if (u.getType().getName() in ['Resource_Mineral_Field', 'Resource_Vespene_Geyser']):
                # print('mineral %s'%str(coor))
                if(u.isVisible(me.getPlayer())):
                    resource.append((u.getType().getName() == 'Resource_Mineral_Field', coor,
                                     (u.getTop(), u.getBottom(), u.getLeft(), u.getRight())
                                     ))
            elif (u.getType().getName() == 'Zerg_Extractor' and u.getPlayer() == me.getPlayer()):
                extra.append(coor)
            elif (u.getPlayer() == me.getPlayer()):
                ally.append((coor, u.getHitPoints() + u.getShields(), tu.isFlyer(), tu.isBuilding(),
                             (u.getTop(), u.getBottom(), u.getLeft(), u.getRight()),
                             ))
            else:
                enemy.append((coor, u.getHitPoints() + u.getShields(), tu.isFlyer(), tu.isBuilding(),
                              game.getDamageTo(me.getType(), u.getType(), me.getPlayer(), u.getPlayer()),
                              game.getDamageFrom(me.getType(), u.getType(), me.getPlayer(), u.getPlayer()),
                              (u.getTop(), u.getBottom(), u.getLeft(), u.getRight()),
                              (u.getType().groundWeapon().minRange(), u.getType().groundWeapon().maxRange()),
                              (u.getType().airWeapon().minRange(), u.getType().airWeapon().maxRange()),
                              ))
    ans.append(enemy)
    ans.append(ally)
    ans.append(resource)
    ans.append(extra)
    explored=numpy.zeros([game.mapHeight(),game.mapWidth()])
    for i in range(game.mapHeight()):
        for j in range(game.mapWidth()):
            explored[i,j]=game.isExplored(i,j)
    ans.append(explored)
    return ans


def get_all_mineral():
    ans = []
    for i in game.getAllUnits():
        if (i.getType().getName() == 'Resource_Mineral_Field'):
            ans.append(i.getPosition())
    return ans


def get_all_drones():
    ans = []
    for i in game.getAllUnits():
        if (i.getType().getName() == 'Zerg_Drone'):
            ans.append(i.getPosition())
    return ans

def command(unit, order):
    showCommands=0
    # order=[random.randint(-239,239),random.randint(-239,239),random.randint(0,5)]
    coord = unit.getPosition()
    coord[0] += order[0] - WINDOW_SIZE//2
    coord[1] += order[1] - WINDOW_SIZE//2
    lcmd = unit.getLastCommand()

    if(showCommands):
        for uu in game.getAllUnits():
            if (order[2]==1 and uu.getID()!=unit.getID() and collide(order, unit)):
                print('conflict')
    # print('cmd',lcmd.getType().getName(),lcmd.getTargetPosition())
    if (lcmd.getTargetPosition() == coord and type2cmd(lcmd.getType()) == order[2]):
        return
    if (order[2] == 0):
        if (showCommands):
            print('holding')
        unit.holdPosition()
    elif (order[2] == 1):
        if (showCommands):
            print('moving to', coord)
        unit.move(coord)
    elif (order[2] == 2):
        if (showCommands):
            print('building')
        unit.build(pybrood.UnitTypes.Zerg_Spawning_Pool, coord)
    elif (order[2] == 3):
        if (showCommands):
            print('gathering ', coord, get_all_mineral())
        unit.gather(game.getClosestUnit(coord))
    elif (order[2] == 4):
        if (showCommands):
            print('attacking ', coord, get_all_drones())
        unit.attack(game.getClosestUnit(coord, radius=1))
    elif (order[2] == 5):
        unit.returnCargo()

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


