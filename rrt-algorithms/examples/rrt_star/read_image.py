# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import sys
from keras_preprocessing.image.utils import img_to_array

from numpy.core.defchararray import center
sys.path.append('/home/lift/wzw/segement/planning/rrt-algorithms')

import os
import cv2
import numpy as np
from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

path = "/home/lift/wzw/segement/planning/image_preprocess/image/"
filename = "mos1_heat_map.png"

original_item = "mosRes_2021_12_06_22_41_37_444.png"
original_img = cv2.imread(os.path.join(path,original_item))
original_img = cv2.resize(original_img,(500,300))

image = cv2.imread(os.path.join(path,filename),0)

# if image.shape[0] != 300:
#     image = cv2.resize(image,(500,300))
print(image.shape[0],image.shape[1])

pixels = []
start_x = 0
start_y = 0



def catch_point(event,x,y,flags,param):
    global pixels,start_x,start_y

    center = []

    if event == cv2.EVENT_LBUTTONDOWN:

        start_x = x
        start_y = y
    elif event == cv2.EVENT_LBUTTONUP:

        end_x = x
        end_y = y
        cv2.rectangle(original_img, (start_x,start_y), (end_x,end_y), (125, 125, 125), thickness=2)
        cv2.rectangle(image, (start_x,start_y), (end_x,end_y), 0, thickness=2)
        cv2.imshow("image",image)
        if start_x != end_x:
            min_x = min(start_x,end_x)
            min_y = min(start_y,end_y)
            width = abs(start_x - end_x)
            height = abs(start_y - end_y)
            center = np.append((int)(min_x+width/2),(int)(min_y+height/2))
            
        else:
            center = np.append(x,y)
        print(center)
        pixels.append(center)
        print(len(pixels))


def vanish(event,x,y,flags,param):
    global pixels,start_x,start_y

    center = []

    if event == cv2.EVENT_LBUTTONDOWN:

        start_x = x
        start_y = y
    elif event == cv2.EVENT_LBUTTONUP:

        end_x = x
        end_y = y
        cv2.rectangle(image, (start_x,start_y), (end_x,end_y), (255), thickness=-1)
        cv2.imshow("image",image)





X_dimensions = np.array([(0, image.shape[1]), (0, image.shape[0])])  # dimensions of Search Space
# obstacles
Obstacles = []
obstacle = []
cv2.namedWindow("image")

while True:
    cv2.setMouseCallback('image', vanish)
    cv2.imshow("image",image)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
cv2.imwrite(os.path.join("/home/lift/wzw/segement/planning/image_preprocess/image",filename),image)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):

        if image[i,j] < 100:
            obstacle = np.append(j,i)

            Obstacles.append(tuple(obstacle))
print(len(Obstacles))



while True:
    cv2.setMouseCallback('image', catch_point)
    cv2.imshow("image",image)
    k = cv2.waitKey(0)
    if k == ord('q') or len(pixels) >=2:
        break


# cv2.destroyAllWindows()
x_init = (0, 0)  # starting location
x_goal = (100, 100)  # goal location
x_init = tuple(pixels[0])
x_goal = tuple(pixels[1])




# if Obstacles.count(x_init)!=0 or Obstacles.count(x_goal)!=0:
#     raise Exception("selected point in the obstacles")

Q = np.array([(12, 4)])  # 8,4 length of tree edges Q[0]represent the distance between x_rand and x_new ,x_new generated from the direction x_rand to x_new 代表x_rand到x_nearst方向上取x_new的距离大小

r = 2  # length of smallest edge to check for intersection with obstacles
max_samples = 8000  # max number of samples to take before timing out
rewire_count = 80  # optional, number of nearby branches to rewire
prc = 1  # probability of checking for a connection to goal

# create Search Space

X = SearchSpace(X_dimensions, Obstacles, x_init, x_goal)
# create rrt_search
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)

path = rrt.rrt_star()


last = (0,0)
for point in path:
    if int(last[0]) == 0 & int(last[1]) == 0:
        last = pixels[0]
    cv2.line(image,(int(last[0]),int(last[1])),(int(point[0]),int(point[1])),(125, 125, 125),1)
    cv2.line(original_img,(int(last[0]),int(last[1])),(int(point[0]),int(point[1])),(0, 0, 0),2)
    last = point
last = path[-1]
x_goal = pixels[1]
cv2.line(image,(int(last[0]),int(last[1])),(x_goal[0],x_goal[1]),(125, 125, 125),1)
cv2.line(original_img,(int(last[0]),int(last[1])),(x_goal[0],x_goal[1]),(0, 0, 0),2)
cv2.imwrite(os.path.join("/home/lift/wzw/segement/planning/rrt-algorithms/output",filename),image)
cv2.imwrite(os.path.join("/home/lift/wzw/segement/planning/rrt-algorithms/output",original_item),original_img)


# plot
plot = Plot("rrt_star_2d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles_self(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
