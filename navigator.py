import time
import win32api
import win32con
def keypress(key):
    win32api.keybd_event(key, 0,0,0)
    time.sleep(.05)
    win32api.keybd_event(key,0 ,win32con.KEYEVENTF_KEYUP ,0)
    time.sleep(0.2)

if(__name__=='__main__'):
    time.sleep(3)
    keypress(0x28)
    keypress(0x28)
    keypress(0x28)
    
    keypress(0x4f)
