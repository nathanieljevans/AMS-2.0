import AMS_stagecontrol as con
import AMS_segmentation as vis
import time
import AMS_predictor as pred
import subprocess as sp

_TEMP_IMGS = "temp_imgs"
_ANNO_IMGS = "image_segmentation_verification"
_NEW_OBJ_IMGS = "new_data_obj_imgs"
# --------------------------------------------------- RUN -------------------------------------------------------------
try: 
    print(' This is version 2.0, last edited: 3/7/18 ' )
    tic = time.time()
    print("clearing temp files")
    con.delete_temp(_TEMP_IMGS) # make sure there are no old images in the folder
    con.delete_temp(_ANNO_IMGS)
    con.delete_temp(_NEW_OBJ_IMGS)
    print("Time to remove old files: " + str(time.time() - tic))
    
    
    username = input("For email notification of results, type username now (For no notification, press enter): ")
    if (username is ''): 
        username = None 
    
    print()
    for lab in pred.get_label_names(): 
        print(lab)
    
    print("\nTo add a bias, list the expected object types (use integer labels from above)\nto include all object types, leave blank and press enter.\nExample: 0 1 5 6 7 8\n")   
    try: 
        bias = [int(x) for x in input('Expected labels: ').split()]  
        if (bias == []): 
            bias = None  
            print('proceeding unbiased')              
    except: 
        print('failed to add bias, proceeding unbiased')
    
    if (bias == []): 
        bias = None
        
    con.main()
    tic = time.time()
    vis.image_analysis()
    pred.classify_new_data(user=username, bias=bias)
    
    print("Time for image processing: " + str(time.time() - tic))
    
    print( 'To copy files to Lancer (/192.168.8.2/projects/Epperson/AMS_images/): Run move_images.sh ')
    #sp.call('/home/vgdev/Desktop/move_images.sh', shell=True)
    
except:   
    print("Program Failed")
    raise
    
