
from script.colordescriptor import ColorDescriptor
from script.search import Searcher
import argparse
import cv2
import numpy as np

from input_data_test import InputData
import matplotlib.pyplot as plt
import heapq


batch_size = 1


def validate_acc(dist_array, top_k):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy
def validate(grd_descriptor, sat_descriptor):
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    print(dist_array.shape[0])
    top1_percent = 10
    top_k = []
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        """prediction = np.sum(dist_array[:, i] < gt_dist)
        print(prediction)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount"""
        per_grd = dist_array[:,i].tolist()
        per_grd_top_k = list(map(per_grd.index,heapq.nsmallest(top1_percent,per_grd)))
        top_k.append(per_grd_top_k)
    print(top_k)

    return top_k





input_data = InputData()

input_data.reset_scan()




sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 960])
map_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 960])
cd = ColorDescriptor((8, 8, 3))

descriptor = cv2.xfeatures2d.SIFT_create()


def extract_features(image, vector_size=32):
    #image = imread(image_path, mode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        #此处为了简化安装步骤，使用KAZE，因为SIFT/ORB以及其他特征算子需要安
#装额外的模块
        alg = cv2.xfeatures2d.SIFT_create()
        # Finding image keypoints
        #寻找图像关键点
        ###kps = alg.detect(image)
        #print(kps.shape)
        # Getting first 32 of them. 
        #计算前32个
        # Number of keypoints is varies depend on image size and color pallet
        #关键点的数量取决于图像大小以及彩色调色板
        # Sorting them based on keypoint response value(bigger is better)
        #根据关键点的返回值进行排序（越大越好）
        ###kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        #计算描述符向量
        kps, dsc = alg.detectAndCompute(image, None)
        #print(dsc)
        # Flatten all of them in one big vector - our feature vector
        # 将其放在一个大的向量中，作为我们的特征向量
        dsc = dsc[:vector_size,:]
        dsc = dsc.flatten()
        #print(dsc.size)
        # Making descriptor of same size
        # 使描述符的大小一致
        # Descriptor vector size is 64
        #描述符向量的大小为64
        needed_size = (vector_size * 128)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros 
            # at the end of our feature vector
#如果少于32个描述符，则在特征向量后面补零
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
            print('+++++++++++++++++++++++++++++++++++++')
    except cv2.error as e:
        print( 'Error: ', e)
        return None

    #print('finish')
    return dsc
    

val_i = 0
while True:
    print('      progress %d' % val_i)
    batch_sat, batch_map = input_data.next_batch_scan(batch_size)
    if batch_sat is None:
        break
    #print(batch_map.shape)
    for i  in range(batch_sat.shape[0]):
        
        """#sat8bit = cv2.normalize(batch_sat[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        #map8bit = cv2.normalize(batch_map[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        #print(map8bit)
        #sat_global_val = cd.describe(batch_sat[i])
        #map_global_val = cd.describe(batch_map[i])
        
        #_,sat_global_val =  descriptor.detectAndCompute(sat8bit,None)
        #print(sat_global_val.shape)
        #_,map_global_val =  descriptor.detectAndCompute(map8bit,None)
        
        sat_global_val =   extract_features(batch_sat[i])
        map_global_val =   extract_features(batch_map[i])
    
        sat_global_descriptor[val_i: val_i + i, :] = sat_global_val
        map_global_descriptor[val_i: val_i + i, :] = map_global_val
        #print(val_i)"""
    
    sat = batch_sat.squeeze(-4)
    mapl = batch_map.squeeze(-4)
    
    sat_global_val = cd.describe(sat)
    map_global_val = cd.describe(mapl)
    
    sat_global_descriptor[val_i: val_i + 1, :] = sat_global_val
    map_global_descriptor[val_i: val_i + 1, :] = map_global_val
    
    val_i += batch_sat.shape[0]

"""d = 0.5 * np.sum([(sat_global_descriptor - map_global_descriptor)**2.0)/(sat_global_descriptor + map_global_descriptor + 1e-10)]

top1_percent = int(dist_array.shape[0] * 0.01) + 1
#val_accuracy = np.zeros((1, top1_percent))
print('start')

fig1, ax1 = plt.subplots(figsize=(11, 8))

top_k_recall = []
epoches = []
for i in range(top1_percent):
    epoches.append(i)
    val_accuracy = validate_acc(dist_array, i)
    top_k_recall.append(val_accuracy)
    ax1.plot(epoches, top_k_recall,color='green')
ax1.set_title("Beijing")
ax1.set_xlabel("Top_K")
ax1.set_ylabel("Recall")
plt.savefig('./recall_Beijing84.png')
print('top1', ':', top_k_recall[1])
print('top5', ':', top_k_recall[5])
#print(top1_percent)
print('top10', ':', top_k_recall[10])
print('top1%', ':', top_k_recall[top1_percent-1])"""




dist_array = 2 - 2 * np.matmul(sat_global_descriptor, np.transpose(map_global_descriptor))
top1_percent = int(dist_array.shape[0] * 0.01) + 1
#val_accuracy = np.zeros((1, top1_percent))
print('start')

fig1, ax1 = plt.subplots(figsize=(11, 8))

top_k_recall = []
epoches = []
for i in range(top1_percent):
    epoches.append(i)
    val_accuracy = validate_acc(dist_array, i)
    top_k_recall.append(val_accuracy)
    ax1.plot(epoches, top_k_recall,color='green')
ax1.set_title("Beijing")
ax1.set_xlabel("Top_K")
ax1.set_ylabel("Recall")
plt.savefig('./recall_Beijing84.png')
print('top1', ':', top_k_recall[1])
print('top5', ':', top_k_recall[2])
#print(top1_percent)
print('top10', ':', top_k_recall[10])
print('top1%', ':', top_k_recall[top1_percent-1])

with open('./Seattle_67.csv', 'w',newline='') as csvfile:
    for i in top_k_recall:
        csvfile.writelines(str(i))
        csvfile.writelines('\n')

"""img = cv2.imread ("E:/Program/Beijing/map/002076.png")
descriptor = cv2.xfeatures2d.SIFT_create()
ps, des = descriptor.detectAndCompute(img,None)
print(des)"""