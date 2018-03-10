"""
Created on Mon Aug 28 17:13:33 2017

@author: Nate Evans 
"""

from scipy import misc
from skimage import measure
from skimage.filters import threshold_local
from os import listdir
import matplotlib.patches as patches 
import sys
from matplotlib import pyplot as plt
 
# threshold_local parameters - segmentation - 
_BLOCKSIZE = 299 
_OFFSET = 10 

# add N pixels of padding to object images when saving 
_PADDING = 5

# object area thresholds
_MIN_AREA = 50
_MAX_AREA = 2000

_IMAGE_DIR = ""

'''
INPUT(S): 
    None
OUTPUT(S): 
    count (int) is the sum of objects segmented from images in the directory 'temp_imgs' 
SUMMARY: 
    Reads the microscope slide images file names in the directory _IMAGE_DIR and process
    each image, one at a time.
'''
def image_analysis(): 
    global _IMAGE_DIR
    img_files = listdir(_IMAGE_DIR + 'temp_imgs')
    count = 0
    ID = None
    for i,name in enumerate(img_files) : 
        cnt, id_full = single_img_proc(name, id_full=ID)
        ID = id_full
        count += cnt
        percent = (float(i)/len(img_files))*100
        sys.stdout.write("Image analysis progress: %d%% \r" % (percent) )
        sys.stdout.flush
          
    print("\nNumber of ixels)new data objects saved: " + str((count))+'\n')

    return count

'''
INPUT(S): 
    f_name (str) is the path to the image to be processed 
    id_full (int) is the next unique object ID. If id_full=None then the next unique ID 
        will be pulled from the AMP_IMAGE_DATABASE directory. 
OUTPUT(S): 
    returns a tuple (count, id_full) where 
    count (int) represents the # of objects segmented in the image 
    id_full (int) represents the last object ID assigned to a segmented object 
SUMMARY: 
    image processing for an image containing wafer objects. Adaptive thresholding
    to segment objects and a general size and focus filter to generate likely objects. 
    Logs the resulting objects. 
'''
def single_img_proc(f_name, id_full=None): 
    global image_dir_path
    global _BLOCKSIZE
    global _OFFSET 
    global _PADDING
    global _MIN_AREA
    global _MAX_AREA
    global _IMAGE_DIR
    
    if (id_full is None): 
        id_full = get_last_id(_IMAGE_DIR + "AMS_savedobjects/") # unique ID based off full database
    
    # load image and but don't flatten to greyscale, seems to save better quality with flattening
    img = misc.imread(_IMAGE_DIR + 'temp_imgs/' + f_name, flatten=False)

    # create a binary mixels)ask for object segmentation 
    binary_adaptive = img > threshold_local(img, _BLOCKSIZE, offset=_OFFSET, method='gaussian')   

    # label blobs and get value
    blob_labels = measure.label(binary_adaptive, background = 1, return_num = False, connectivity = 2)

    # get blob properties 
    properties = measure.regionprops(blob_labels)

    # create figure to annotate with recognized object locations 
    ann_fig, ann_ax = plt.subplots()
    plt.imshow(img)
    
    # segment objects for later classification 
    count = 0
    for j,obj_prop in enumerate(properties): 
        if (obj_prop.area > _MIN_AREA and obj_prop.area < _MAX_AREA): #general size filter
            count+=1
            bbox = obj_prop.bbox
            rect_width = bbox[3] - bbox[1] + _PADDING*2
            rect_height = bbox[2] - bbox[0] + _PADDING*2
            X1 = max(bbox[1] - _PADDING, 0)
            Y1 = max(bbox[0] - _PADDING, 0)
            X2 = min(bbox[3] + _PADDING, 1279)
            Y2 = min(bbox[2] + _PADDING, 1023)
            waf = img[Y1:Y2, X1:X2]
            ann_ax.add_patch(patches.Rectangle((X1,Y1),rect_width,rect_height,linewidth=1,edgecolor='r',facecolor='none'))
            try: 
                misc.imsave(_IMAGE_DIR + "new_data_obj_imgs/" + str(id_full) +".bmp", waf)             #save in temp folder for testing purposes
                misc.imsave(_IMAGE_DIR + "AMS_savedobjects/" + str(id_full) +".bmp", waf)   # save in full obj database folder
                id_full+=1 
                
            except:
                print("couldn't log objects")
                raise

    ann_fig.savefig(_IMAGE_DIR + 'image_segmentation_verification/'+ f_name[0:-4] + '_anno.png')
    plt.close(ann_fig)
    return (count, id_full)

'''
INPUT(S):
    pth (str) is the path to a directory containing the full collection of segmented 
    data objects.
OUTPUT(S): 
    i_d (int) is the next greatest unique ID 
SUMMARY: 
    sorts through the file names in the passed directory and sorts them by the greatest
    value and returns the next greatest integer. e.g. 
    DIR :            Return : 
        <25.bmp>            250 
        <27.bmp> 
        <249.bmp> 
'''
def get_last_id(pth): 
    f = listdir(pth)
    if len(f) == 0: 
        i_d = 1
    else: 
        fi = list(map(lambda x: int(x[0:-4]), f))
        fi.sort()
        i_d = (fi[-1] + 1) #remove the last four characters '.bmp'
    return i_d 