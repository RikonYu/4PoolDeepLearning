from pybrood import game, run, BaseAI
import matplotlib.pyplot as plt
import numpy
class DrawerAI(BaseAI):
    def prepare(self):
        print(game.mapHeight()*32,game.mapWidth()*32)
        ans=numpy.zeros([game.mapHeight()*32,game.mapWidth()*32])
        for i in range(game.mapHeight()*4):
            for j in range(game.mapWidth()*4):
                ans[i*8:i*8+8,j*8:j*8+8]=game.isWalkable([i,j])
        #fig=plt.figure(figsize=(1,2))
        #fig.add_subplot(1,2,1)
        plt.imshow(ans*255,cmap=plt.cm.gray)
        plt.show()
        ans=numpy.zeros([game.mapHeight()*32,game.mapWidth()*32])
        for r in game.getAllRegions():
            ans[r.getBoundsTop():r.getBoundsBottom(),r.getBoundsLeft():r.getBoundsRight()]=r.isHigherGround()
        #fig.add_subplot(1,2,2)
        plt.imshow(ans * 255, cmap=plt.cm.gray)
        plt.show()
        #fig.show()

    def frame(self):
        pass

if __name__=='__main__':
    run(DrawerAI)