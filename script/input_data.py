import cv2
import random
import numpy as np
import os

class InputData:

    img_root = '../dataset'
    
    def __init__(self):

        self.test_list = self.img_root + '/test.csv'
        #NewOrleans_test.csv
        #Xian_test.csv
        #Orlando.csv
        #NewYork_test.csv



        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([data[0], data[1], pano_id])
                #print([data[0], data[1], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)




    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size: 
            batch_size = self.test_data_size - self.__cur_test_id

        batch_map = np.zeros([batch_size, 256, 256, 3], dtype = np.float32)
        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # satellite
#            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            #img[:, :, 0] -= 103.939  # Blue
            #img[:, :, 1] -= 116.779  # Green
            #img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img
            
            print(self.id_test_list[img_idx][1])
            # map
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][1])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            if img is None or img.shape[0] != 256 or img.shape[1] != 256:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_test_list[img_idx][1], i), img.shape)
                exit()
                
            img = img.astype(np.float32)
            # img -= 100.0
            #img[:, :, 0] -= 103.939  # Blue
            #img[:, :, 1] -= 116.779  # Green
            #img[:, :, 2] -= 123.6  # Red
            batch_map[i, :, :, :] = img
            
        self.__cur_test_id += batch_size

        return batch_sat, batch_map



    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0
if __name__ == "__main__":
    input_data = InputData()
    input_data.next_pair_batch(16)
    
    