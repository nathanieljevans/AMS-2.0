#!/bin/bash 
# moving AMS images into P://Epperson 

echo Input Accession Number\:
read varname 
name=$(date '+%d-%b-%Y_%H:%M__acc-')$varname
echo Images will be saved to P\:\/\/Epperson\/AMS_images\/$name
cd /mnt/projects/Epperson/AMS_images/
rm -rf $name
mkdir $name
cd $name
cp -a /home/vgdev/local/spyder_working_directory/wafer_objects/temp_imgs/ .
cp -a /home/vgdev/local/spyder_working_directory/wafer_objects/image_segmentation_verification/ .
cp -a /home/vgdev/local/spyder_working_directory/wafer_objects/classified_objects/ . 
