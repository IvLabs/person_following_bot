import cv2 as cv
import argparse
import os

def main():
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)
        print('Created out_dir')
	files = os.listdir(args.input_dir)
	for name in files:
		try:
			img = cv.imread( args.input_dir + '/'+ name);

		except:
			print('Error reading: ' + name);
		try:

			resized = cv.resize(img, (args.width,args.height));
			cv.imwrite( args.out_dir + '/' + name, resized);
		except:
			print('Error writing: ' + name);

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--out_dir", type=str, help="Path to save the images")
    parser.add_argument("-i", "--input_dir", type=str, help="Bag file to read")
    parser.add_argument("-hd", "--height", type=int, help="op height")
    parser.add_argument("-wd", "--width", type=int, help="op width")
    args = parser.parse_args()
    main()
