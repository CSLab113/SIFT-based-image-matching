
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




val_i = 0
while True:
    print('      progress %d' % val_i)
    batch_sat, batch_map = input_data.next_batch_scan(batch_size)
    if batch_sat is None:
        break
    #print(batch_map.shape)
    
    sat = batch_sat.squeeze(-4)
    mapl = batch_map.squeeze(-4)
    
    sat_global_val = cd.describe(sat)
    map_global_val = cd.describe(mapl)
    
    sat_global_descriptor[val_i: val_i + 1, :] = sat_global_val
    map_global_descriptor[val_i: val_i + 1, :] = map_global_val
    
    val_i += batch_sat.shape[0]





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

with open('./result.csv', 'w',newline='') as csvfile:
    for i in top_k_recall:
        csvfile.writelines(str(i))
        csvfile.writelines('\n')
