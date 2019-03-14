__author__='Travix'
import configparser
import cv2 
import numpy as np
import SimpleITK as sitk
from lib.myshow import myshow, myshow3d
from matplotlib import pyplot as plt
from  PIL import Image,ImageFilter
from math import atan2#参数y在前，x在后

def process_config(conf_file):
	"""
		read configuration from files and saved to a dict
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'Global':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'ActiveContour':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params
	
def frame_diff(gray_img,*args):
	diff_frame1=cv2.absdiff(gray_img,args[0])
	diff_frame0=cv2.absdiff(args[1],gray_img)
	return cv2.bitwise_and(diff_frame1,diff_frame0)
	
def concatenate(img1,gray_img):
	gray_img_=np.stack([gray_img,gray_img,gray_img],axis=2)
	concat=np.concatenate([img1,gray_img_],axis=1)#数组拼接
	return concat
	
def thresh_otsu(img,*args):
	#th=cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,4)
	ret,th=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#基于直方图的二值化
	return th
	
def resize_and_gray(img,return_color=False,scale=2.):
	w,h,c=img.shape#width,height,color，缩放范围scale为原来一半大小
	shape=(int(h/scale),int(w/scale))
	img=cv2.resize(img,shape)
	gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#颜色空间转换，BGR2GRAY彩色转为黑白
	if return_color:
		return img,gray_img
	else:
		return gray_img

def plt_show(img,title=None):
	plt.figure(figsize=(10,10)) 
	if title!=None:
		plt.title(title)
	if len(img.shape)==2:
		cmap=plt.get_cmap('gray')#cmap是colormap的缩写。颜色图谱设为黑白
		plt.imshow(img,cmap=cmap)
	else:
		plt.imshow(img)
	plt.axis('off')
    
def parse_image(img,min_area= 500):
    (_,contours, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#只检测外轮廓
    goodIndex =[]
    gravity= []
    areas = []
    cnts =[]
    boundboxs=[]
    for ii, contour in enumerate(contours):
        area = cv2.contourArea(contour)#计算轮廓的面积
        if area > min_area:
            goodIndex.append(ii)
            M = cv2.moments(contour)#得到图像的矩
            gravity.append(( int(M['m10']/M['m00']),int(M['m01']/M['m00'])))#计算中心
            areas.append(area)
            cnts.append(np.squeeze(contour, axis=1))#删除数组形状中的单维度条目?
            boundboxs.append(cv2.boundingRect(contour))#矩形边框，img是一个二值图，返回的是矩阵左上角的坐标x,y以及宽高wh
                                         # 然后可以利用cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩形，2是线宽
            
    mask = np.zeros(img.shape, dtype=img.dtype)
    mask=np.stack([mask,mask,mask],axis=2)
    for i,ii in enumerate(goodIndex):#i是索引号，每次都不相同；ii是之前记录的索引
        cv2.drawContours(mask, contours, ii, (255,0,0), cv2.FILLED)#以填充模式绘制轮廓
        x,y,w,h= boundboxs[i]
        x0,y0 = gravity[i]
        #cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(mask,(x0,y0),2,(30,125,255),-1)  
    return len(goodIndex),np.array(gravity),np.array(areas),np.array(cnts),np.array(boundboxs),mask


def worm_tracking(last_frame_data,cur_frame_data,max_dist_thresh=17):     
    
    count_1, gravity_1, areas_1 = last_frame_data
    count_2, gravity_2, areas_2 = cur_frame_data
    diff_gravity = [np.linalg.norm(gravity_2-it,axis=1) for it in gravity_1]#求范数,axis=1是按照行向量求范数。axis=0是按照列向量求范数
    diff_gravity = np.array(diff_gravity)#线虫质心距离
    min_dist = np.min(diff_gravity,axis=1)
    min_dist_index = np.argmin(diff_gravity,axis=1)
    is_valid_track = (areas_2[min_dist_index]>0.7*areas_1)&(areas_2[min_dist_index]<1.3*areas_1)&(min_dist<max_dist_thresh)
    #分割不好时，设置最大阈值 max_dist_thresh，防止跟踪错误
    #print(areas_2[min_dist_index]>0.65*areas_1,areas_2[min_dist_index]<1.35*areas_1,min_dist<max_dist_thresh)
    #print("-"*20)
    i= 0
    min_dist_index_dict={}
    for it,iu in zip(is_valid_track,min_dist_index):
        min_dist_index_dict[i] = iu if it else 100#如果有最小距离的索引(质心索引)，就将他给字典{当前索引:最小距离索引}；若无，值赋为100
        i+=1
    new_track = list(set(range(count_2)).difference(min_dist_index_dict.values()))#当前帧图像中的线虫，多于前一帧的，作差;该操作是对set集合作差
                                                        #x.difference(y),返回包含在集合x中，但不在集合y中的集合
    return min_dist_index_dict, new_track

def get_angle(skeleton):
    vec1 = skeleton[0]-skeleton[24]
    vec2 = skeleton[48]-skeleton[24]
    angle= atan2(vec1[1],vec1[0])- atan2(vec2[1],vec2[0])
    angle = angle if angle>0 else angle+2*np.pi
    return angle/np.pi*180.

class geodesicActiveContourSegementation:

	def __init__(self,params):
		self.params=params
		self.PropagationScaling=params['propagationscaling']
		self.CurvatureScaling=params['curvaturescaling']
		self.AdvectionScaling=params['advectionscaling']
		self.MaximumRMSError=params['maximumrmserror']
		self.NumberOfIterations=params['numberofiterations']
		self.Dist_threshold=params['dist_threshold']
		self.Sigma=params['sigma']
		self.geodesicActiveContour = sitk.GeodesicActiveContourLevelSetImageFilter()
		self.geodesicActiveContour.SetPropagationScaling(self.PropagationScaling)
		self.geodesicActiveContour.SetCurvatureScaling(self.CurvatureScaling)
		self.geodesicActiveContour.SetAdvectionScaling(self.AdvectionScaling)
		self.geodesicActiveContour.SetMaximumRMSError(self.MaximumRMSError)
		self.geodesicActiveContour.SetNumberOfIterations(self.NumberOfIterations)
		self.gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
		self.gradientMagnitude.SetSigma(self.Sigma)
		
	def get_init_contour(self,img): 
		reverse=255-img
		dist_transform = cv2.distanceTransform(reverse,cv2.DIST_L2,3)
		_,ret=cv2.threshold(dist_transform,self.Dist_threshold,255,cv2.THRESH_BINARY)
		ret=255-ret
		shape_filtered=shape_filter(ret.astype(np.uint8))
		shape_filtered=shape_filtered.astype(np.uint8)
		holes=fill_holes(shape_filtered)
		init_contour=cv2.morphologyEx(holes,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations = 7)
		return init_contour
		
	def get_init_contour_v2(self,img):
		'''
			params:
				img: img is a the diff of original img and backgroud
		'''
		edge=cv2.Canny(img,7,60)
		reverse=255-edge
		dist_transform = cv2.distanceTransform(reverse,cv2.DIST_L2,3)
		_,ret=cv2.threshold(dist_transform,4,255,cv2.THRESH_BINARY_INV)
		shape_filtered=shape_filter(ret.astype(np.uint8),thresh=2000)
		shape_filtered=shape_filtered.astype(np.uint8)
		reverse=255-shape_filtered
		dist_transform = cv2.distanceTransform(reverse,cv2.DIST_L2,3)
		_,init_contour=cv2.threshold(dist_transform,4,255,cv2.THRESH_BINARY_INV)
		init_contour=init_contour.astype(np.uint8)
		return init_contour
	def execute_v1(self,img,foreground):
		init_contour=self.get_init_contour(foreground)
		init_contour = sitk.GetImageFromArray(init_contour)
		init_contour = sitk.SignedMaurerDistanceMap(init_contour, 
						insideIsPositive=False, useImageSpacing=False)
		featureImage=sitk.GetImageFromArray(img)
		featureImage = sitk.BoundedReciprocal(self.gradientMagnitude.Execute(featureImage))
		featureImage = sitk.Cast( featureImage, init_contour.GetPixelID()) 
		levelset = self.geodesicActiveContour.Execute( init_contour, featureImage )
		contour=levelset<0
		return contour
	def execute(self,diff):
		featureImage=sitk.GetImageFromArray(diff)
		featureImage = sitk.BoundedReciprocal(self.gradientMagnitude.Execute(featureImage))
		bk=sitk.GetArrayFromImage(featureImage)
		bk=((bk/bk.max())*255).astype(np.uint8)
		otsu=thresh_otsu(bk)
		dist_transform = cv2.distanceTransform(otsu,cv2.DIST_L2,3)
		_,retu=cv2.threshold(dist_transform,5,255,cv2.THRESH_BINARY_INV)
		retu=retu.astype(np.uint8)
		(_,cnts, hier) = cv2.findContours(retu.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		areas=[(int(cv2.contourArea(c)),h[3]==-1) for (c,h) in zip(cnts,hier[0])]
		areas=np.array(areas)
		outer_contour_idx=np.where((areas[:,0]>self.params['min_contour_area'])&(areas[:,1]>0))
		inter_contour_idx=np.where((areas[:,0]>self.params['min_hole_area'])&(areas[:,1]<1))
		outer_contour=[cnts[i] for i in outer_contour_idx[0]]
		inter_contour=[cnts[i] for i in inter_contour_idx[0]]
		mask=np.zeros(diff.shape)
		mask=cv2.drawContours(mask,outer_contour,-1,255,-1) 
		mask=cv2.drawContours(mask,inter_contour,-1,0,-1)
		init_contour_ = sitk.GetImageFromArray(mask.astype(np.uint8))
		init_contour_1 = sitk.SignedMaurerDistanceMap(init_contour_, insideIsPositive=False, useImageSpacing=False)
		featureImage = sitk.Cast( featureImage, init_contour_1.GetPixelID())
		levelset = self.geodesicActiveContour.Execute(init_contour_1, featureImage )
		contour=levelset<0
		return contour,otsu
class PIL_filter:
	def __init__(self,filter_type):
		filter_type_dict={'EDGE_ENHANCE':ImageFilter.EDGE_ENHANCE,
							'EDGE_ENHANCE_MORE':ImageFilter.EDGE_ENHANCE_MORE,
							'FIND_EDGES':ImageFilter.FIND_EDGES}
		self.filter_type=filter_type_dict[filter_type]
	def filter(self,img):
		shape=img.shape
		pil_im = Image.fromarray(img)  
		pil_im=pil_im.filter(self.filter_type)
		pil_im=pil_im.convert("L")
		data=pil_im.getdata()
		data=np.matrix(data)
		img=np.reshape(data,shape)
		return img
	def filter_type(self,filter_type):
		self.filter_type=filter_type_dict[filter_type]

class frame_factory:
	def __init__(self,path):
		self.cap=cv2.VideoCapture(path)
		self.map={}
		self.num_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))#get用来获取视频的信息，CAP_PROP_FRAME_COUNT是指视频文件的帧数
		_,self.map[0]=self.cap.read()#做了一个图像缓存区，缓存20张图片用以前后的处理
		self.tail=0
		self.head=0
		self.size=(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		self.fps = self.cap.get(cv2.CAP_PROP_FPS)
	def __getitem__(self,index):
		if index>=self.tail and index<=self.head:#tail在左，head在右，假定序列往右读取
			return self.map[index]
		elif index>self.head:
			num_add=(index-self.head)
			assert num_add<20 and index<self.num_frames#如果有意外状况则抛出断言
			for i in range(1,num_add+1):
				_,self.map[self.head+i]=self.cap.read()	
			self.head=index
			if len(self.map)>20:
				num_del=len(self.map)-20
				for i in range(num_del):
					self.map.pop(self.tail+i)
				self.tail+=num_del
			return self.map[index]
		else:
			raise Exception("error")
	
	def reset(self):
		self.cap=cv2.VideoCapture(path)
		self.map={}
		_,self.map[0]=self.cap.read()
		self.tail=0
		self.head=0
	def set_file_name(self,name):
		self.writer=cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'XVID'), self.fps, self.size)
	def write(self,frame):
		self.writer.write(frame)
	def write_done(self):
		self.writer.release()
class track:
    track_id = 0
    def __init__(self, start_index,inter_frame_index):
        self.begin = start_index
        self.track_id = track.track_id
        track.track_id +=1
        self.inter_frames_index = [inter_frame_index]
    def add_track(self,inter_frame_index):
        self.inter_frames_index.append(inter_frame_index)
    
    def stop(self,end_index):
        self.end = end_index
        self.num_frames =self.end - self.begin+1
        
class watershed_segement:
	def __init__(self):
		pass
	