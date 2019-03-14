__author__='Travix'
import numpy as np
import cv2


color = [
(241,242,224), (196,203,128), (136,150,0), (64,77,0), 
(201,230,200), (132,199,129), (71,160,67), (32,94,27),
(130,224,255), (7,193,255), (0,160,255), (0,111,255),
(220,216,207), (174,164,144), (139,125,96), (100,90,69),
(252,229,179), (247,195,79), (229,155,3), (155,87,1),
(231,190,225), (200,104,186), (176,39,156), (162,31,123),
(210,205,255), (115,115,229), (80,83,239), (40,40,198)]
# Color Names
color_name = [
'teal01', 'teal02', 'teal03', 'teal04',
'green01', 'green02', 'green03', 'green04',
'amber01', 'amber02', 'amber03', 'amber04',
'bluegrey01', 'bluegrey02', 'bluegrey03', 'bluegrey04',
'lightblue01', 'lightblue02', 'lightblue03', 'lightblue04',
'purple01', 'purple02', 'purple03', 'purple04',
'red01', 'red02', 'red03', 'red04']
color_map={}
for i in range(len(color_name)):
    color_map[color_name[i]]=color[i]
color_map['yellow']=(255,255,0)
color_map['violet']=(148,0,211)
color_map['PaleTurquoise']=(175,238,238)
def plot_arrow(img,orgin, destination, angle=25., fraction=0.2,color1='purple02', color2='purple03',thicken =2):
    orgin = np.array(orgin,dtype=np.int)
    destination = np.array(destination,dtype=np.int)
    angle =angle/180.*np.pi
    length = np.linalg.norm(orgin- destination)
    short_length = length*fraction
    vector = orgin - destination
    vector_rotate = np.dot(np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]),vector)*fraction
    end_point1 =vector_rotate+destination
    angle = -angle
    vector_rotate = np.dot(np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]),vector)*fraction
    end_point2 =vector_rotate+destination
    cv2.line(img,tuple(orgin),tuple(destination),color_map[color1],thicken)
    cv2.line(img,tuple(end_point1.astype(np.int)),tuple(destination),color_map[color2],thicken)
    cv2.line(img,tuple(end_point2.astype(np.int)),tuple(destination),color_map[color2],thicken)
    return img
def add_arrow(img, skeleton):
    middle = skeleton[24]
    head = skeleton[0]
    tail = skeleton[48]
    img = plot_arrow(img,middle,head,color1='violet',color2='PaleTurquoise')
    img = plot_arrow(img,middle,tail,color1='yellow',color2='PaleTurquoise')
    return img