import numpy
class unitInfo:
    def __init__(self, unit):

        self.coord=unit.getPosition()
        self.bounds=[unit.getTop(),unit.getBottom(),unit.getLeft(),unit.getRight()]
        self.type=unit.getType().getName()
        self.ID=unit.getID()
        self.HP=unit.getHitPoints()
        self.shield=unit.getShields()
        self.playerID=unit.getPlayer().getID()
        self.killCount=unit.getKillCount()
        self.rangeGround=(unit.getType().groundWeapon().minRange(),unit.getType().groundWeapon().maxRange())
        self.rangeAir = (unit.getType().airWeapon().minRange(), unit.getType().airWeapon().maxRange())
        self.canFireGround=unit.getGroundWeaponCooldown()
        self.canFireAir=unit.getAirWeaponCooldown()
        self.hasGas=unit.isCarryingGas()
        self.hasMineral=unit.isCarryingMinerals()
        self.isFlyer=unit.getType().isFlyer()
        self.isBuilding=unit.getType().isBuilding()
    def addDamage(self, unitMe, unit):
        from pybrood import game
        self.damageTo= game.damagaTo(unitMe.getType(), unit.getType(), unitMe.getPlayer(), unit.getPlayer())
        self.damageFrom= game.damagaFrom(unitMe.getType(), unit.getType(), unitMe.getPlayer(), unit.getPlayer())

class gameMessage:
    def __init__(self, unit):
        from pybrood import game
        self.myInfo=unitInfo(unit)
        self.enemies=[]
        self.allies=[]
        self.resources=[]
        self.extractors=[]
        for u in game.getAllUnits():
            if(u.getType().getName() in ['Resource_Mineral_Field', 'Resource_Vespene_Geyser']):
                self.resources.append(unitInfo(u))
            elif (u.getType().getName() == 'Zerg_Extractor' and u.getPlayer() == unit.getPlayer()):
                self.extractors.append(unitInfo(u))
            elif (u.getPlayer() == unit.getPlayer()):
                self.allies.append(unitInfo(u))
            else:
                self.enemies.append(unitInfo(u))
        self.explored=numpy.zeros([game.mapHeight(), game.mapWidth()])

class mapMessage:
    def __init__(self):
        from pybrood import game
        self.ans = numpy.zeros([game.mapHeight() * 4, game.mapWidth() * 4])
        for i in range(game.mapHeight() * 4):
            for j in range(game.mapWidth() * 4):
                self.ans[i, j] = game.isWalkable([i, j])

class gameState:
    def __init__(self, msg_type, msg, value):
        self.type=msg_type
        self.msg=msg
        self.value=value
class mapState:
    def __init__(self, msg, unitType, mapName):
        self.type='reg'
        self.msg=msg
        self.unitType=unitType
        self.mapName=mapName