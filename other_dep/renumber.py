import argparse
import os


def main():
	cntr = 0
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)
        print('Created out_dir')
    if args.inc:
        inc = args.inc
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
    parser.add_argument("-i", "--input_dir", type=str, help="Path to directory containg files")
    parser.add_argument("-n", "--start_num", type=int, help="Starting number")
    parser.add_argument("-inc", "--increment", type=int, help="increment number")
    args = parser.parse_args()
    main()


# Function to renumber files
import os

z =534;
lis = os.listdir('../annotations3')
for i in lis:
   z = z+1;
   os.rename('../annotations3/'+i,'../annotationsB/'+str(z)+'.xml')
   os.rename('../depth3/'+i.strip('_rgb.xml')+'_depth.jpg','../depthB/'+str(z)+'.jpg')
   os.rename('../rgb3/'+i.strip('.xml')+'.jpg','../rgbB/'+str(z)+'.jpg')
# i = lis[0]
# os.rename('../annotations/'+str(1000000+1000001)+'_rgb.xml','../annotations/'+'416_rgb.xml',)
