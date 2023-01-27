import matplotlib.pyplot as plt
import cv2
import os 
from args import parser 
import numpy as np
from random import sample

class Extracter():
    def __init__(self) -> None:
        pass
    
    def load_img(self):
        self.foggy_zurich=[]
        self.cityscape=[]
        self.cityscape_fog=[]
        self.data_list=[]
        
        # Foggy_Zurich
        fz_path = os.path.dirname(self.arg.fz_dir)
        for path, dirs, files in os.walk(fz_path):
            for file in files:
                    self.file_path = os.path.join(path, file)
                    _, ext = os.path.splitext(self.file_path)
                    if ext == '.png':
                        self.foggy_zurich.append(self.file_path)   
        
        print(len(self.foggy_zurich))
        self.foggy_zurich = sample(self.foggy_zurich, 2975)                  
        self.data_list.append(list(self.foggy_zurich))
        
        # Cityscape
        city_path  = os.path.dirname(self.arg.cityscape_dir)
        for path, dirs, files in os.walk(city_path):
            for file in files:
                    self.file_path = os.path.join(path, file)
                    _, ext = os.path.splitext(self.file_path)
                    if ext == '.png':
                        self.cityscape.append(self.file_path) 
        
        print(len(self.cityscape))
        self.cityscape = sample(self.cityscape, 2975)    
        self.data_list.append(list(self.cityscape))
        
        # Cityscape 0.005
        city_fog_path = os.path.dirname(self.arg.cityscape_fog_dir)   
        for path, dirs, files in os.walk(city_fog_path):
            for file in files:
                    self.file_path = os.path.join(path, file)
                    _, ext = os.path.splitext(self.file_path)
                    if ext == '.png':
                        self.cityscape_fog.append(self.file_path)            

        print(len(self.cityscape_fog))
        self.cityscape_fog = sample(self.cityscape_fog, 2975)    
        self.data_list.append(list(self.cityscape_fog))
        
    def extract_xyz(self):
        print("CIE XYZ.Rec 709 with D65 white point")
        # drawing histogram is not a good idea
    
    def extract_YCrCb(self):
        print("YCrCb JPEG (or YCC)")
    
    def extract_LUV(self):
        print("LUV")
        self.hist_ch1_sum = []
        self.hist_ch2_sum = []  
        self.hist_ch3_sum = []
                
        for i in range(3):
            dataset = self.data_list[i]    
            
            for idx, img in enumerate(dataset):
                src = cv2.imread(img).astype("float32") / 255       # check type
                conv = cv2.cvtColor(src, cv2.COLOR_BGR2LUV)         # check converter 
                c1, c2, c3 = cv2.split(conv) 
                      
                ch = []
                ch.append(c1)
                ch.append(c2)
                ch.append(c3)  
                                                
                if idx == 0:
                    print (c1, '\n', c2, '\n', c3)
                    ch1 =  cv2.calcHist([ch[0]], [0], None, [100], [0, 100])  # check range 
                    ch2 =  cv2.calcHist([ch[1]], [0], None, [200], [-100, 100])  # check range 
                    ch3 =  cv2.calcHist([ch[2]], [0], None, [200], [-100, 100])  # check range 
                                        
                else:
                    ch1 =  cv2.calcHist([ch[0]], [0], None, [100], [0, 100])  + next_ch1
                    ch2 =  cv2.calcHist([ch[1]], [0], None, [200], [-100, 100])  + next_ch2
                    ch3 =  cv2.calcHist([ch[2]], [0], None, [200], [-100, 100])  + next_ch3
                
                next_ch1 = ch1
                next_ch2 = ch2               
                next_ch3 = ch3    

            self.hist_ch1_sum.append(ch1)
            self.hist_ch2_sum.append(ch2)
            self.hist_ch3_sum.append(ch3)
            
            ch1 = 0
            ch2 = 0
            ch3 = 0    
                 
    def draw_plot(self):
        plt.plot(self.hist_ch1_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(self.hist_ch1_sum[1], color='g', label='Cityscape') 
        plt.plot(self.hist_ch1_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, self.arg.dataset+'_ch1')
        plt.savefig(save_dir)
        plt.clf()

        plt.plot(self.hist_ch2_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(self.hist_ch2_sum[1], color='g', label='Cityscape') 
        plt.plot(self.hist_ch2_sum[2], color='r', label='Cityscape foggy')  
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, self.arg.dataset+'_ch2')
        plt.savefig(save_dir)
        plt.clf()

        plt.plot(self.hist_ch3_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(self.hist_ch3_sum[1], color='g', label='Cityscape') 
        plt.plot(self.hist_ch3_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()  
        save_dir = os.path.join(self.arg.output_dir, self.arg.dataset+'_ch3')
        plt.savefig(save_dir)
        plt.clf()          

    def extract_HLS(self): # light instead of v of HSV 
        print("HLS")      
    
        hist_ch1_sum = []
        hist_ch2_sum = []  
        hist_ch3_sum = []
                
        for i in range(3):
            dataset = self.data_list[i]    
            
            for idx, img in enumerate(dataset):
                src = cv2.imread(img).astype("float32")            # check type
                conv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)         # check converter 
                c1, c2, c3 = cv2.split(conv) 
                      
                ch = []
                ch.append(c1)
                ch.append(c2)
                ch.append(c3)  
                                                
                if idx == 0:
                    print(c1, '\n', c2, '\n', c3) 
                    ch1 =  cv2.calcHist([ch[0]], [0], None, [360], [0, 360])  # check range 
                    ch2 =  cv2.calcHist([ch[1]], [0], None, [1], [0, 1])  # check range 
                    ch3 =  cv2.calcHist([ch[2]], [0], None, [256], [0, 255])  # check range 
                                        
                else:
                    ch1 =  cv2.calcHist([ch[0]], [0], None, [360], [0, 360])  + next_ch1
                    ch2 =  cv2.calcHist([ch[1]], [0], None, [1], [0, 1])  + next_ch2
                    ch3 =  cv2.calcHist([ch[2]], [0], None, [256], [0, 255])  + next_ch3
                
                next_ch1 = ch1
                next_ch2 = ch2               
                next_ch3 = ch3    

            hist_ch1_sum.append(ch1)
            hist_ch2_sum.append(ch2)
            hist_ch3_sum.append(ch3)
            
            ch1 = 0
            ch2 = 0
            ch3 = 0 

        plt.plot(hist_ch1_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_ch1_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_ch1_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'hls_h')
        plt.savefig(save_dir)
        plt.clf()

        plt.plot(hist_ch2_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_ch2_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_ch2_sum[2], color='r', label='Cityscape foggy')  
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'hls_l')
        plt.savefig(save_dir)
        plt.clf()

        plt.plot(hist_ch3_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_ch3_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_ch3_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()  
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'hls_s')
        plt.savefig(save_dir)
        plt.clf()          
    
    def extract_HSV(self):  # H: 0-179, S: 0-255, V: 0-255
        print("HSV")
        
        hist_ch1_sum = []
        hist_ch2_sum = []  
        hist_ch3_sum = []
                
        for i in range(3):
            dataset = self.data_list[i]    
            
            for idx, img in enumerate(dataset):
                src = cv2.imread(img)                               # check type
                conv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)         # check converter 
                c1, c2, c3 = cv2.split(conv) 
                      
                ch = []
                ch.append(c1)
                ch.append(c2)
                ch.append(c3)
                                
                if idx == 0:
                    ch1 =  cv2.calcHist([ch[0]], [0], None, [180], [0, 179])  # check range 
                    ch2 =  cv2.calcHist([ch[1]], [0], None, [256], [0, 255])  # check range 
                    ch3 =  cv2.calcHist([ch[2]], [0], None, [256], [0, 255])  # check range 
                                        
                else:
                    ch1 =  cv2.calcHist([ch[0]], [0], None, [180], [0, 179])  + next_ch1
                    ch2 =  cv2.calcHist([ch[1]], [0], None, [256], [0, 255])  + next_ch2
                    ch3 =  cv2.calcHist([ch[2]], [0], None, [256], [0, 255])  + next_ch3
                
                next_ch1 = ch1
                next_ch2 = ch2               
                next_ch3 = ch3    

            hist_ch1_sum.append(ch1)
            hist_ch2_sum.append(ch2)
            hist_ch3_sum.append(ch3)
            
            ch1 = 0
            ch2 = 0
            ch3 = 0 

        plt.plot(hist_ch1_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_ch1_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_ch1_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'hsv_h')
        plt.savefig(save_dir)
        plt.clf()

        plt.plot(hist_ch2_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_ch2_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_ch2_sum[2], color='r', label='Cityscape foggy')  
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'hsv_s')
        plt.savefig(save_dir)
        plt.clf()

        plt.plot(hist_ch3_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_ch3_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_ch3_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()  
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'hsv_v')
        plt.savefig(save_dir)
        plt.clf()
        
    def extract_LAB(self):
        print("LAB")
        
        hist_l_sum = []
        hist_a_sum = []  
        hist_b_sum = []
                
        for i in range(3):
            dataset = self.data_list[i]    
            
            for idx, img in enumerate(dataset):
                src = cv2.imread(img).astype("float32") / 255
                lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab) 
                
                if idx == 0:
                    hist_l =  cv2.calcHist([l], [0], None, [100], [0, 100]) 
                    hist_a =  cv2.calcHist([a], [0], None, [256], [-128, 128])  
                    hist_b =  cv2.calcHist([b], [0], None, [256], [-128, 128]) 
                    
                    next_l = hist_l
                    next_a = hist_a               
                    next_b = hist_b
                    
                else:
                    hist_l =  cv2.calcHist([l], [0], None, [100], [0, 100]) + next_l
                    hist_a =  cv2.calcHist([a], [0], None, [256], [-128, 128])  + next_a
                    hist_b =  cv2.calcHist([b], [0], None, [256], [-128, 128])  + next_b
                    
                    next_l = hist_l
                    next_a = hist_a               
                    next_b = hist_b

            # normalize since the samples are diff 
            # hist_l = hist_l / len(dataset)
            # hist_a = hist_a / len(dataset)
            # hist_b = hist_b / len(dataset)
            
            hist_l_sum.append(hist_l)
            hist_a_sum.append(hist_a)
            hist_b_sum.append(hist_b)
            
            hist_l = 0
            hist_a = 0
            hist_b = 0

        plt.plot(hist_l_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_l_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_l_sum[2], color='r', label='Cityscape 0.005') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_l')
        plt.savefig(save_dir)
        plt.clf()
        
        plt.plot(hist_a_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_a_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_a_sum[2], color='r', label='Cityscape 0.005') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_a')
        plt.savefig(save_dir)
        plt.clf()
        
        plt.plot(hist_b_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_b_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_b_sum[2], color='r', label='Cityscape 0.005') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_b')
        plt.savefig(save_dir)
        plt.clf()
    
    def extract_RGB_multiple(self):
        print("RGB_multiple")
        
        hist_l_sum = []
        hist_a_sum = []  
        hist_b_sum = []
                
        for i in range(3):
            dataset = self.data_list[i]    
            
            for idx, img in enumerate(dataset):
                src = cv2.imread(img)
                l, a, b = cv2.split(src) 
                
                if idx == 0:
                    hist_l =  cv2.calcHist([l], [0], None, [256], [0, 256]) 
                    hist_a =  cv2.calcHist([a], [0], None, [256], [0, 256])  
                    hist_b =  cv2.calcHist([b], [0], None, [256], [0, 256]) 
                    
                    next_l = hist_l
                    next_a = hist_a               
                    next_b = hist_b
                    
                else:
                    hist_l =  cv2.calcHist([l], [0], None,  [256], [0, 256]) + next_l
                    hist_a =  cv2.calcHist([a], [0], None,  [256], [0, 256])  + next_a
                    hist_b =  cv2.calcHist([b], [0], None,  [256], [0, 256])  + next_b
                    
                    next_l = hist_l
                    next_a = hist_a               
                    next_b = hist_b

            # normalize since the samples are diff 
            # hist_l = hist_l / len(dataset)
            # hist_a = hist_a / len(dataset)
            # hist_b = hist_b / len(dataset)
            
            hist_l_sum.append(hist_l)
            hist_a_sum.append(hist_a)
            hist_b_sum.append(hist_b)
            
            hist_l = 0
            hist_a = 0
            hist_b = 0

        plt.plot(hist_l_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_l_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_l_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'bgr_b')
        plt.savefig(save_dir)
        plt.clf()
        
        plt.plot(hist_a_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_a_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_a_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'bgr_g')
        plt.savefig(save_dir)
        plt.clf()
        
        plt.plot(hist_b_sum[0], color='b', label='Foggy_Zurich') 
        plt.plot(hist_b_sum[1], color='g', label='Cityscape') 
        plt.plot(hist_b_sum[2], color='r', label='Cityscape foggy') 
        plt.legend()
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'bgr_r')
        plt.savefig(save_dir)
        plt.clf()    
    
    
    def extract_RGB(self):                                      # future work: alpha, gamma 
        hist_b = np.zeros((256,1))
        hist_g = np.zeros((256,1))
        hist_r = np.zeros((256,1))
        for idx, img in enumerate(self.foggy_zurich):
            src = cv2.imread(img)
            
            colors = ['b', 'g', 'r']
            bgr_planes = cv2.split(src) # b, g, r channels 

            for (p, c) in zip(bgr_planes, colors): # 3-3 pair of a image 
                if c == 'b':
                    hist_b +=  cv2.calcHist([p], [0], None, [256], [0, 256])
                elif c == 'g':
                    hist_g +=  cv2.calcHist([p], [0], None, [256], [0, 256])
                elif c == 'r':
                    hist_r +=  cv2.calcHist([p], [0], None, [256], [0, 256])
                else : 
                    print("exception!")

        hist_b = hist_b / len(self.img_dir)
        hist_g = hist_g / len(self.img_dir)
        hist_r = hist_r / len(self.img_dir)
        
        plt.plot(hist_b, color='b') 
        plt.plot(hist_g, color='g') 
        plt.plot(hist_r, color='r') 
        
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'bgr_Cityscape_foggy') # replace with args
        plt.savefig(save_dir)
        plt.clf
    
    def run(self):
        print("running ...")
        self.arg = parser.parse_args()
        self.load_img()
    
        # self.extract_LAB()
        # self.extract_RGB_multiple()
        # self.extract_HSV()
        # self.extract_HLS()
        self.extract_LUV()
        
        self.draw_plot()
        
if __name__ == "__main__":
    ex = Extracter()
    ex.run()
    
    
