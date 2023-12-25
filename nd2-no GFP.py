from matplotlib import image
from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
threshold1 = 19  #细胞轮廓图像阈值
threshold2 = 61 #溶酶体位置图像阈值
threshold_l = 0.3 #细胞质范围比例阈值
key2, key3 = 1,2 #input the two channels your need used (0,1,2,3...) 1是细胞轮廓，2是主要参考细胞，3是溶酶体位置参考细胞
key1 = key2
###############################################################################################################################################
lsPointsChoose = []
tpPointsChoose = []
pointsCount = 0
count = 0
pointsMax = 10 #圈细胞点数
global img_cell,mask2
def on_mouse(event, x, y, flags, param):
  global img_cell, point1, point2, count, pointsMax
  global lsPointsChoose, tpPointsChoose # 存入选择的点
  global pointsCount # 对鼠标按下的点计数
  global init_img_cell, ROI_bymouse_flag
  init_img_cell = img_cell.copy() # 此行代码保证每次都重新再原图画 避免画多了

  if event == cv2.EVENT_LBUTTONDOWN: # 左键点击
    pointsCount = pointsCount + 1
    # 为了保存绘制的区域，画的点稍晚清零
    if(pointsCount == pointsMax + 1):
      pointsCount = 0
      tpPointsChoose = []
    # print('pointsCount:', pointsCount)
    point1 = (x, y)
    # print (x, y)
    # 画出点击的点
    cv2.circle(init_img_cell, point1, 2, (0, 255, 0), 2)
 
    # 将选取的点保存到list列表里
    lsPointsChoose.append([x, y]) # 用于转化为darry 提取多边形ROI
    tpPointsChoose.append((x, y)) # 用于画点

    # 将鼠标选的点用直线链接起来
    #print(len(tpPointsChoose))
    for i in range(len(tpPointsChoose) - 1):
      cv2.line(init_img_cell, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
    # 点击到pointMax时可以提取去绘图
    if(pointsCount == pointsMax):
      # 绘制感兴趣区域
      ROI_byMouse()
      ROI_bymouse_flag = 1
      lsPointsChoose = []
 
    cv2.imshow('src', init_img_cell)
    
  # 右键按下清除轨迹
  if event == cv2.EVENT_RBUTTONDOWN: # 右键点击
    #print("right-mouse")
    pointsCount = 0
    tpPointsChoose = []
    lsPointsChoose = []
    #print(len(tpPointsChoose))
    for i in range(len(tpPointsChoose) - 1):
    #   print('i', i)
      cv2.line(init_img_cell, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 5)
    cv2.imshow('src', init_img_cell)
def ROI_byMouse():
  global src, ROI, ROI_flag, mask2
  mask = np.zeros(img_cell.shape, np.uint8)
  pts = np.array([lsPointsChoose], np.int32)
  pts = pts.reshape((-1, 1, 2)) # -1代表剩下的维度自动计算
  # 画多边形
  mask = cv2.polylines(mask, [pts], True, (0, 255, 255))
  # 填充多边形
  mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
  #cv2.imshow('mask',mask2)
  #mask_gray = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)
  #print(mask_gray.shape)
  #cv2.imwrite('mask.png', mask2)
  ROI = cv2.bitwise_and(mask2, img_cell)
  cv2.imshow('ROI', ROI)
def choose_cell():
  global img_cell,init_img_cell, ROI
  # 图像预处理，设置其大小    
  ROI = img_cell.copy()
  cv2.namedWindow('src')
  cv2.setMouseCallback('src', on_mouse)  
  cv2.imshow('src', img_cell)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
##########################################################################################
import tkinter.filedialog
img_path = tkinter.filedialog.askopenfilename(title='选择一个文件', filetypes=[('所有文件','.*')])

img =  np.array(ND2Reader(img_path))
img_array = np.array(img,dtype=np.float64).transpose(1,2,0)
height, width ,channels = img_array.shape  
size = (int(width * 0.5), int(height * 0.5)) 

for i in range(channels):
    plt.subplot(2,3,i+1)#channel number
    plt.imshow(img[i])
plt.show()

img_copy = np.zeros_like(img,dtype=np.float64)
image_copy = np.zeros_like(img, dtype=np.uint8)
equ_img = np.zeros_like(img, dtype=np.uint8)
for i in range(channels):
    img_copy[i] = img[i]
    image_copy[i] = img_copy[i]/16
    #image_copy[i] = (img_copy[i] - np.min(img_copy[i])) / (np.max(img_copy[i]) - np.min(img_copy[i]))*255
    equ_img[i] = cv2.equalizeHist(image_copy[i])
    #equ_img[i] = image_copy[i]

R_img_array = np.zeros((height,width,3),dtype=np.uint8) #初始化Red图片，全部值为0
G_img_array = np.zeros((height,width,3),dtype=np.uint8) #初始化Green图片，全部值为0
B_img_array = np.zeros((height,width,3),dtype=np.uint8) #初始化Blue 图片，全部值为0

#key1, key2 = eval(input("input the two channels your need used:"))

def nothing(x):
    pass

cv2.namedWindow("image1") 
cv2.namedWindow("image2")      
cv2.createTrackbar("threshold1", "image1", threshold1, 255, nothing) 
cv2.createTrackbar("threshold2", "image2", threshold2, 255, nothing) 
    
while True:   
    mythreshold1 = cv2.getTrackbarPos("threshold1", "image1")    
    ret1, image_bin1 = cv2.threshold(image_copy[key1], mythreshold1, 255, 
                                cv2.THRESH_BINARY)    
    R_img_array[:,:,2] = image_bin1
    raw_image1 = img_copy[key1]
    image1 = cv2.resize(R_img_array, size, interpolation=cv2.INTER_AREA)
    cv2.imshow("image1",image1)   
    #cv2.imwrite("image1.png",image1)

    mythreshold2 = cv2.getTrackbarPos("threshold2", "image2")  
    ret, image_bin2 = cv2.threshold(image_copy[key3], mythreshold2, 255, 
                                cv2.THRESH_BINARY) 
    B_img_array[:,:,0] = image_bin2
    raw_image2 = img_copy[key3]
    image2 = cv2.resize(B_img_array, size, interpolation=cv2.INTER_AREA)
    cv2.imshow("image2",image2) 
    #cv2.imwrite("image2.png",image2)

    image_dis = image_bin1 - image_bin2
    image3 = cv2.resize(image_dis, size, interpolation=cv2.INTER_AREA)
    cv2.imshow('image_dis', image3)
    #cv2.imwrite("image_dis.png", image3)
  
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()

noise_image = np.copy(img_copy[key2]) #the main image
noise_image[np.where(image_bin1>0)] = 0
threshold = int(noise_image.mean())
img_copy[np.where(img_copy<threshold)] = 0
img_copy[np.where(img_copy>=threshold)] -= threshold
print("the noise threshold",threshold)

key_cell_count = eval(input("Input the count of cells:"))

raw_image = img_copy[key2] #the main image
raw_lysosomal = np.copy(raw_image)
image_bin = image_bin2 #溶酶体区域
raw_lysosomal[np.where(image_bin<1)] = 0 #只选择溶酶体区域
area_raw_lysosomal = len(np.where(image_bin>0)[0]) #溶酶体区域
mean_lysosomal = raw_lysosomal.sum()/area_raw_lysosomal
threshold_lysosomal = threshold_l*mean_lysosomal

raw_image_dis = np.copy(raw_image) #the main image
raw_image_dis[np.where(image_bin>0)] = 0 #删除溶酶体区域
raw_image_dis[np.where(raw_image_dis<threshold_lysosomal)] = 0 #删除非细胞区域
area_image_dis = len(np.where(raw_image_dis>0)[0]) #细胞区域

whole_img_ratio = (raw_image_dis.sum()/area_image_dis)/mean_lysosomal
print("整张图片的比值(差值图/选择图)：", whole_img_ratio)
#print("区域大小",len(np.where(image_bin1>0)[0]),area_image_dis,area_raw_lysosomal)
# print("亮度值",raw_image1.sum(),raw_image_dis.sum(), raw_lysosomal.sum())
# cv2.imshow('mask_dis',raw_image_dis)
# cv2.imshow('mask_lysosomal',raw_lysosomal)
# cv2.waitKey(0)
img_name = img_path.split("/")[-1].split(".")[0]
csv_path = img_name + "_results.csv"
with open(csv_path,"w+",encoding='utf-8',newline='') as f:
  writer = csv.writer(f)
  writer.writerow(["细胞序号","比值"])
  writer.writerow(["all", str(whole_img_ratio)])
  for i in range(key_cell_count):
      img_cell = equ_img[key1]
      choose_cell()
      mask_dis = np.copy(raw_image_dis)
      mask_dis[np.where(mask2<1)] = 0
      area_dis = np.where(mask_dis>0)[0]

      mask_lysosomal = np.copy(raw_lysosomal)
      mask_lysosomal[np.where(mask2<1)] = 0
      area_lysosomal = np.where(mask_lysosomal>0)[0]
      ratio = (mask_dis.sum()/len(area_dis))/(mask_lysosomal.sum()/len(area_lysosomal))
      ratio = round(ratio, 4)
      print('第几个细胞数:',i,'   比值:',ratio)
      writer.writerow([str(i), str(ratio)])
      # cv2.imshow('mask_dis',mask_dis)
      # cv2.imshow('mask_lysosomal',mask_lysosomal)
      # cv2.waitKey(0)

