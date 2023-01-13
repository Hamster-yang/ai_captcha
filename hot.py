import keyboard
idx=0
while True:
    idx+=1
    a=''
    a=keyboard.read_key()
    if a!='' and idx%2==0:
        print("You pressed "+a)
        
