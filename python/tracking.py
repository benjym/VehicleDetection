import os, sys, glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
cmap = matplotlib.cm.get_cmap('Spectral')

def dist(x0,x1):
    return np.sqrt(np.sum(np.square(x0-x1)))

data_dir = './yolov5/runs/detect/exp3/labels/'
homography = False
classtype = 0 # see coco-classes.txt for what each class is, although you need to subtract 1 from that list to get to this number
distance_threshold = 0.1

files = natsorted(glob.glob(data_dir + '*.txt'))
nframes = int(files[-1].split('_')[-1][:-4])
filename = '_'.join(files[-1].split('_')[:-1])

alive = []

for i in range(nframes):
    # if we found anything in this frame
    if os.path.exists(filename + '_' + str(i) + '.txt'):
        # load that data
        data = np.loadtxt(filename + '_' + str(i) + '.txt')
        for row in data:
            # each row contains classtype, x, y, width, height in normalised units
            if row[0] == classtype:
                found = False # by default it is not found to exist in the list already
                for object in alive:
                    if i-1 in object: # check the previous timestep
                        if dist(row[1:3],object[i-1][1:3]) < distance_threshold and not found: # if there is something close
                            if dist(row[3:5],object[i-1][3:5]) < distance_threshold: # _VERY_ strange way to check that the bounding boxes are similar sizes, my brain is too tired for this right now
                                found = True # bingo we found it
                                object[i] = row # add it to that dict
                if not found: # well then it must be a new vehicle
                    alive.append( {i :row} ) # make a new dict in the list

# now do some plotting
for i,object in enumerate(alive):
    # t = object.keys()
    x = np.array(list(object.values()))[:,1:3]
    plt.plot(x[:,0],x[:,1],'.',c=cmap(np.random.rand()))

plt.title('colour represents unique pedestrian ID')
plt.ylabel('vertical position on screen')
plt.xlabel('horizontal position on screen')
plt.savefig('trajectories.png')


# UNTESTED SO FAR, JUST A COPY/PASTE FROM STACKOVERFLOW
if homography:
    import cv2 # import the OpenCV library

    # provide points from image 1
    pts_src = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])
    # corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
    pts_dst = np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])

    # calculate matrix H
    h, status = cv2.findHomography(pts_src, pts_dst)

    # finally, get the mapping
    # now do some plotting
    for i,object in enumerate(alive):
        # t = object.keys()
        x = np.array(list(object.values()))[:,1:3]
        x_out = cv2.perspectiveTransform(x, h)
        plt.plot(x_out[:,0],x_out[:,1],'.',c=cmap(np.random.rand()))

    plt.title('colour represents unique pedestrian ID')
    plt.ylabel('vertical position on screen')
    plt.xlabel('horizontal position on screen')
    plt.savefig('trajectories_tranformed.png')
