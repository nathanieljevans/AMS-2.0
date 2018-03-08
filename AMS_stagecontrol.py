import time 
import serial 
import subprocess as sp 
from os import listdir, makedirs, path, remove
import sys

capture_img_path = "AMS_SynchronousGrab/Console/Build/Make/binary/x86_64bit/SynchronousGrabConsole"
captured_img_save_path = "temp_imgs/"

counting = False  
global t # image analysis thread holder
 
# --------------------------CRTITICAL VALUES TO ELIMINATE OVERLAP OR MISS AREA -------------------------
# DO NOT CHANGE MAGNIFICATION OF HARDWARE
# x and y should be proportional to the pixel dimesions (assuming pixels are square)
# these were calculated by known bar pattern dimensions, DO NOT CHANGE
IMG_X = 556 #number of stage user steps for image's X length --- repeat 556
IMG_Y = 445 #number of stage user steps for image's Y height --- repeat 445
# ----------------------------------------------------------------------------------------------------------------------------

'''
INPUT(S): 
    None
OUTPUT(S): 
    None
SUMMARY: 
    Takes user input via command line and performs motor stage control to image desired area
'''
def main():  
    ser = open_port() 
    
    # Synchronize port serial baud and motor controller baud 
    send("BAUD 38", ser) # change prior controller baud rate 
    ser.baudrate = 38400    # change python serial port baud rate 

    print ('Follow the setup instructions.\nInsert "exit" to leave the application.\nCommands are not cap sensitive.\n')
    
    #send/write a command to the serial
    send('JXD 1', ser) #sets correlation between joystick direction and stage's X axis direction of movement 
                       #joystick right, stage moves mechanically right, image on screen moves right
    send('JYD 1', ser) #changes correlation between joystick direction and stage's Y axis direction of movement 
    				 #joystick forward, stage moves mechanically back, image on screen moves forward
    send('H', ser) # disable joystick
    
    col, row = set_boundary(ser)
    
    in1 = input ("To begin stage pattern and image acquisition, press enter. Or EXIT to quit: ")
    if (in1 == 'exit' or in1 == 'EXIT'):
        ser.close()
        exit()
        
    stage_pattern(col, row, ser)
    
    send('J', ser)
    
'''
INPUT(S): 
    columns (int) number of columns in imaging region 
    rows (int) number of rows in imaging region 
    ser (serial object) motor controller communication 
OUTPUT(S): 
    None
SUMMARY: 
    This method is responisble for communicating with the proscan controller to create a serpentine pattern of images covering the slide
'''
def stage_pattern(columns, rows, ser): # this function is called after the setup (below)--it snakes the stage back and forth
    global capture_img_path 
    global counting
    global captured_img_save_path
    
    global IMG_X
    global IMG_Y
    
    tic = time.time()    
    
    ser.reset_output_buffer() #make sure buffer is cleared
    ser.reset_input_buffer()
    
    imgNum = 1 #initialize the counting system for labeling image files
    
    for r in range (0, int(rows) + 1): #rows --- add +1 to ensure that we make it all the way across the slide
 
        for c in range (0, int(columns) + 1): #columns ||| 

            #subprocess.Popen runs the c++ exe as a separate thread, stdin=PIPE opens a pipe to send input to exe
            p = sp.Popen(capture_img_path, stdin = sp.PIPE, universal_newlines = True) #universal_newlines means stdin can be a string instead of bytes
            			
            p.communicate(captured_img_save_path + str(imgNum) + ".bmp") #"communicate" sends the imgStr through stdin PIPE so exe can label image file 
            #if images are turning out blurry, add "time.sleep()" here
            
            if (not c == columns):  # if we're getting a strip of images (single row) we don't want the stage to move 
                send('L ' + str(IMG_X), ser) #"L" tells the stage to move left by imgX number of steps--need space between 'L' and imgX 
            
            percent = (float(imgNum) / ((rows+1)*(columns+1)))*100
            sys.stdout.write("Image acquistion: %d%% \r" % (percent) )
            sys.stdout.flush()
                    
            imgNum += 1 #update the image number so that most recent img file doesn't get overwritten
            
        if (not r == rows): 
            ser.write(str.encode('B ' + str(IMG_Y) + '\r\n')) #"B" tells stage to move back by imgY number of steps

        IMG_X = IMG_X * -1 #this changes the sign of imgX so that next time thru the loop the stage moves R (negative L)

    print("\nImages Acquired")
    print("Time to acquire images: " + str(time.time()-tic))

'''
INPUT(S): 
    ser (serial object) motor controller communication 
OUTPUT(S): 
    columns (int) number of columns in imaging region 
    rows (int) number of rows in imaging region
SUMMARY: 
    This method is responisble for getting user input that defines the imaging regions and converting it to # of img rows and cols. 
'''
def set_boundary(ser):
    
    global IMG_X
    global IMG_Y
    send('J', ser) #"J" activates the joystick  
    
    print("Move to one corner of the slide cover (or desired bounding box to be imaged)") #end should be lower right corner of physical slide, upper right of camera image of slide
    
    in1 = input("press any key when ready or EXIT to quit: ")	#operator should literally type "px" (NOT cap sensitive)
    if in1 == 'exit': 
        ser.close() 
        exit()
        
    endX, endY = get_xy(ser)
    
    print("Move to opposite corner of slide cover (or desired bounding box to be imaged)") #should be upper left corner of physical slide, lower left corner of camera image
    in1 = input("press any key when ready or EXIT to quit: ")	#operator should literally type "px" (NOT cap sensitive)
    if in1 == 'exit': 
        ser.close() 
        exit()
        
    startX, startY = get_xy(ser)
    
    stageDim = [abs(endX - startX), abs(endY - startY)] # dimensions of the bounding box to be imaged -> [x,y]
    send('H', ser) #disable joystick 

    columns = int(stageDim[0]/IMG_X + 1) #calculate the number of columns the slide is to be divided into, round up
    rows = int(stageDim[1]/IMG_Y + 1)  #calculate number of rows, round up
    
    print("Slide width in stage steps: " + str(stageDim[0]) + ", Slide height in stage steps: " + str(stageDim[1]))
    print("Number of columns: " + str(columns) + ", Number of rows: " + str(rows)) #this is optional--lets user see the values
    
    xStart = max([endX, startX])
    yStart = max([endY, startY])
    
    send("G " + str(xStart) + " " + str(yStart), ser)
    
    return columns, rows # returns values necessary for the "pattern" function (written above)
    
'''
INPUT(S): 
    ser (serial object) motor controller communication 
OUTPUT(S): 
    X (int) absolute x position in motor stage steps 
    Y (int) absolute y position in motor stage steps 
SUMMARY: 
    This method is responisble for retrieving the current motor stage position in steps 
'''
def get_xy(ser):
    ser.reset_input_buffer() #clear the buffer so controller can get command
    send('px', ser) #"PX" means "get the current x coordinate of stage"
    out = str.encode('') #initialize out variable to store controller's return later
    time.sleep(0.5)
    while (ser.inWaiting() > 0): #wait for serial to reply
    	out += ser.read(1).strip() #read whatever the controller returns (px = x position) and strip
    print ("End position x: " + str(out))
    	
    X = int(out.strip()) #store the x position in int variable (for use in calculations)
    
    send('py', ser) #"PY" means "get the current y coordinate of stage"
    
    out = str.encode('')
    time.sleep(0.5)
    while ser.inWaiting() > 0:
    	out += ser.read(1) 
    print ("End position y: " + str(out))
    
    Y = int(out.strip())
    return X,Y

'''
INPUT(S):
    cmd (str) command to be sent 
    ser (serial object) motor controller communication 
OUTPUT(S): 
    None
SUMMARY: 
    This method sends a command string to the priror controller 
'''
def send(cmd, ser):
    ser.write(str.encode(cmd +'\r\n'))
    time.sleep(0.1)
    
'''
INPUT(S): 
    None
OUTPUT(S): 
    ser (serial object) connection to prior motor controller 
SUMMARY: 
    This method opens a serial connection to the prior motor controller 
''' 
def open_port():
    # configure the serial connections 
    try: 
        ser = serial.Serial( 
            port='/dev/ttyACM0', #open the port
            baudrate=9600, 
            parity=serial.PARITY_ODD, #serial will return a string -- lines 12, 13, & 14 were in original starter code, this is what I found on google
            stopbits=serial.STOPBITS_TWO, #convert num or str to int
            bytesize=serial.SEVENBITS #similar to STOPBITS_TWO
        ) 
    except: 
        try: 
            ser = serial.Serial( 
            port='/dev/ttyACM1', # try opening a different port (in case ttyACM0 was used)
            baudrate=9600, 
            parity=serial.PARITY_ODD, #serial will return a string -- lines 12, 13, & 14 were in original starter code, this is what I found on google
            stopbits=serial.STOPBITS_TWO, #convert num or str to int
            bytesize=serial.SEVENBITS #similar to STOPBITS_TWO
        ) 
        except: 
            print ("Could not open port!")
            print ("Check that PRIOR controller is connected and powered on")
            print ("Check user group permissions for port: ttyACM0,ttyACM1")
            raise
            exit
     
    print ("port check...") 
    if (not ser.isOpen()):           #close if the port is not open
        print ("...port is not open...") 
        exit 
    else: 
        print ("port is open")
        
    return ser

'''
INPUT(S): 
    pth (str) path to directory whose contents are to be deleted 
OUTPUT(S): 
    None
SUMMARY: 
    Deletes the contents of a given directory  
''' 
def delete_temp(pth): 
    try: 
        if (path.exists(pth)): # os module
        
            img_files = listdir(pth)
            time.sleep(0.5)
            for f in img_files: 
                remove(pth+'/' + f) #os module
        else: 
            makedirs(pth) #os module
    except: 
        print("clearing temp files failed")
        raise
            
        
    