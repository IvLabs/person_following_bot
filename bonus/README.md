# Bonus Functions
### 1.change_res.py
Execute ```python change_res.py -h [help]```

Prints help options to terminal

Execute ```python change_res.py -d [output_directory] -i [input_directory] -hd [height_output_img] -wd [width_output_img]```

Change resolution of input images (from input_directory) to images of resolution (hd, wd) (to output_directory)

### 2.extract_images.py
Execute ```python extract_images.py -h [help]```

Prints help options to terminal

Execute ```python extract_images -d [output_directory] -i [input_bag_file]```

Extract images from input bag file.

### 3.xmlparser.py
Execute ```python xmlparser.py -h [help]```

Prints help options to terminal

Execute ```python xmlparser.py -d [output_directory] -i [input_directory]```

Parse xml files to csv files and write (to output_directory). Change child address to parse specific end.

## Modules:
### imagehelpers:
1. show(image, bounding_box='None'):displays image using matplotlib with bounding box(optional).

coordinates of bounding box are optional it should it in the form:[Xmin,Ymin,Xmax,Ymax]