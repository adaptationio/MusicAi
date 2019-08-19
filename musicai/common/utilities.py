import numpy as np
import random
from sklearn import preprocessing
import csv
import torch
import time
import cv2
import mss
import numpy
import re
import datetime


class DataMaker():
    """gets data and processes ready to use"""

    def __init__(self):
        
        self.love = 14
        


    def normalize(self, x):
        normalized = preprocessing.normalize(x)
        return normalized


    def scaled(self, x):
        scaled = preprocessing.scale(x)
        return scaled

    
    def tocsv(self, x, path):
        with open(path, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            for i in range(len(x)):
                wr.writerow(x[i])

    def totensor(self, data):
        data = torch.from_numpy(data)
        return data

    def batcher(self, x, y, batch_size):
        x_data = list()
        return x_data

    def toarray(self, x):
        x = np.array(x, dtype=np.float32)
        return x

    def process_to_normalized(self, data):
        data = self.data_converted(data)
        data = self.toarray(data)
        data = self.normalize(data)
        return data

    def process_to_array(self,data):
        data = data
        data = self.data_converted(data)
        data = self.time_to_array(data)
        data = self.toarray(data)
    
        return data


    def process_to_tensor(self, data):
        data = data
        data = self.toarray(data)
        data = self.normalize(data)
        data = self.totensor(data)
        return data

    def flatten_full(self, markeallt, user):
        market = data

        x = list()
        for i in range(len(old_data)):
            con = np.concatenate((data), axis=None)
            con = np.concatenate((con, old_data[i][1]), axis=None)
            con[0].tolist()
            x.append(con)
        return x

    def flatten(self, u, m, c):
        u = np.concatenate((u), axis=None)
        m = np.concatenate((m), axis=None)
        c = np.concatenate((c), axis=None)
        flattened = np.concatenate((m, u, c), axis=None)

        #k = self.data_grabber.flatten(market_details, player_details)
        return flattened
        
    def flatten2(self, u):
        u = np.concatenate((u), axis=None)
    
        flattened = np.concatenate(u, axis=None)

        #k = self.data_grabber.flatten(market_details, player_details)
        return flattened

    def flatten_simple(self, u):
        u = np.concatenate((u), axis=None)
        #m = np.concatenate((m), axis=None)
        #c = np.concatenate((c), axis=None)
        flattened = np.concatenate((u), axis=None)

        #k = self.data_grabber.flatten(market_details, player_details)
        return flattened

    def get_screen(self):
        with mss.mss() as sct:
            # Part of the screen to capture
            monitor = {"top": 40, "left": 0, "width": 800, "height": 640}

            while "Screen capturing":
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                img = numpy.array(sct.grab(monitor))

                # Display the picture
                #cv2.imshow("OpenCV/Numpy normal", img)

                # Display the picture in grayscale
                # cv2.imshow('OpenCV/Numpy grayscale',
                #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

                print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

                
    def difference(self, state):
        new_state = []
        r = 194
        for i in range(96):
            before = state[i][0]
            b = i+1
            after = state[b][0]
            diff = after - before
            vol = state[b][4]
            o = state[b][3]
            l = state[b][2]
            h = state[b][1]

            new_state.append([after, diff, vol, o, l, h ])
        return new_state
    
    def difference2(self, state):
        
        new_state = []
        r = 194
        for i in range(194):
            c = state[i][0]
            h = state[i][1]
            l = state[i][2]
            o = state[i][3]
            v = state[i][4]
            c = c - o
            h = h - o
            l = l - o
            

            new_state.append([c, h, l, v])
        return new_state


    

    
#dates = ["2016", "2017", "2018"]
#test = DataGrabber()
#test.process_to_array_2()
#data = test.load_state_2()
#print(len(test.full_year[0]))
#print(len(data[1]))
#candles = test.get_candles('1998-06-01T00:00:00Z', 1, "M15", "EUR_USD")
#print(candles)
#some_data = test.data_converted(candles)
#some_data = test.toarray(some_data)
#some_data = test.normalize(some_data)
#some_data = test.totensor(some_data)
#data_day = some_data[0:1440]
#print(len(data_day))
#print(len(some_data))
#print(candles)
#print(some_data)
#test.get_screen()
#state = [1,2,3,4,5,6,7,8,9,10]
#statenew = state[-4:]
#print(statenew )

