import xml.etree.ElementTree as ET
import numpy as np
import argparse
import os


def main():
	tmp = [1, 0, 0, 0, 0]
	cntr = 0
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)
		print('Created out_dir')
	files = os.listdir(args.input_dir)
	for f_name in files:
		try:
			tree = ET.parse(args.input_dir + '/' + f_name)
			root = tree.getroot()
			data = [ root[6][4][0].text, root[6][4][1].text, root[6][4][2].text, root[6][4][3].text]
			data = list(map(int, data))
			tmp[1:5] =  data
			np.savetxt(args.out_dir + '/' + f_name.strip('_rgb.xml') + '.csv', tmp, delimiter=',')
			cntr = cntr+1
		except:
			print('Error handling: ' + f_name)
	print('');print('                 Total files: '+str(len(files)) + ' | Parsed files: '+str(cntr));print('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--out_dir", type=str, help="Path to save the images")
    parser.add_argument("-i", "--input_dir", type=str, help="Path to directory containg xml files")
    args = parser.parse_args()
    main()
