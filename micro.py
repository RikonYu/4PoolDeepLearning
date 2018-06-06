from time import sleep
from pybrood import BaseAI, run, game, Color


class DragoonDefuseAI(BaseAI):
    playerMe=None
    def prepare(self):
        force = game.getForce(0)
        playerMe=game.

    def frame(self):
        sleep(0.05)
    def finished(self):
        print


if __name__ == '__main__':
    run(HelloAI)
