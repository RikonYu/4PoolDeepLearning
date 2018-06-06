import urllib.request
import os
for i in range(638,2101):
    try:
        urllib.request.urlretrieve('http://www.teamliquid.net/replay/download.php?replay=%d'%i,'E:/sc/1.16/maps/replays/liquid/%d.rep'%i)
    except:
        pass

os.system('shutdown.exe -s -t 0')
