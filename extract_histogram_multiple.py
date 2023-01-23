import matplotlib.pyplot as plt
import cv2
import os 
from args import parser 
import numpy as np

####################

# sampling 3807 for Foggy Zurich / Cityscape / Citysacpe or avg
# draw in a plot with three dataset and legend 
# --arg RGB LAB
# --arg dataset 
# --output name 

####################

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
        self.data_list.append(list(self.foggy_zurich))
        
        # Cityscape
        city_path  = os.path.dirname(self.arg.cityscape_dir)
        for path, dirs, files in os.walk(city_path):
            for file in files:
                    self.file_path = os.path.join(path, file)
                    _, ext = os.path.splitext(self.file_path)
                    if ext == '.png':
                        self.cityscape.append(self.file_path)     
        self.data_list.append(list(self.cityscape))
        
        # Cityscape 0.005
        city_fog_path = os.path.dirname(self.arg.cityscape_fog_dir)   
        for path, dirs, files in os.walk(city_fog_path):
            for file in files:
                    self.file_path = os.path.join(path, file)
                    _, ext = os.path.splitext(self.file_path)
                    if ext == '.png':
                        self.cityscape_fog.append(self.file_path)            
        
        self.data_list.append(list(self.cityscape_fog))
        
    def extract_LUV(self):
        print("LUV")

    
    def extract_HLS(self):
        print("HLS")        
    
    def extract_HSV(self):  
        print("HSV")   

        for i in range(3):
            datset = self.data_list[i]    
 
            for idx, img in enumerate(datset):
                src = cv2.imread(img)
                hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv) 
    
            #     if idx == 0:
            #         hist_h =  cv2.calcHist([l], [0], None, [100], [0, 100]) 
            #         hist_s =  cv2.calcHist([a], [0], None, [256], [-128, 128])  
            #         hist_v =  cv2.calcHist([b], [0], None, [256], [-128, 128]) 
                    
            #         next_l = hist_l
            #         next_a = hist_a               
            #         next_b = hist_b
                    
            #     else:
            #         hist_l =  cv2.calcHist([l], [0], None, [100], [0, 100]) + next_l
            #         hist_a =  cv2.calcHist([a], [0], None, [256], [-128, 128])  + next_a
            #         hist_b =  cv2.calcHist([b], [0], None, [256], [-128, 128])  + next_b
                    
            #         next_l = hist_l
            #         next_a = hist_a               
            #     next_b = hist_b

            # # hist_l = hist_l / len(self.img_dir)
            # # hist_a = hist_a / len(self.img_dir)
            # # hist_b = hist_b / len(self.img_dir)

            # plt.plot(hist_l, color='b') 
            # save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_l_Cityscape')
            # plt.savefig(save_dir)
            # plt.clf()
            
            # plt.plot(hist_a, color='g') 
            # save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_a_Cityscape')
            # plt.savefig(save_dir)
            # plt.clf()
            
            # plt.plot(hist_b, color='r') 
            # save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_b_Cityscape')
            # plt.savefig(save_dir)
            # plt.clf()     

        
    def extract_YCC(self):
        print("YCC")
        
    def extract_LAB(self):
        print("LAB")
        
        for idx, img in enumerate(self.foggy_zurich):
            src = cv2.imread(img).astype("float32") / 255
            lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab) 
            
            # hist_l +=  cv2.calcHist([l], [0], None, [100], [0, 100]) 
            # hist_a +=  cv2.calcHist([a], [0], None, [256], [-128, 128])  
            # hist_b +=  cv2.calcHist([b], [0], None, [256], [-128, 128]) 
            
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

        # hist_l = hist_l / len(self.img_dir)
        # hist_a = hist_a / len(self.img_dir)
        # hist_b = hist_b / len(self.img_dir)

        plt.plot(hist_l, color='b') 
        # plt.hist(hist_l.ravel(), 100, [0, 100])
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_l_Cityscape')
        plt.savefig(save_dir)
        plt.clf()
        
        plt.plot(hist_a, color='g') 
        # plt.hist(hist_a.ravel(), 256, [-128, 128])
        # plt.xlim([-128, 128])
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_a_Cityscape')
        plt.savefig(save_dir)
        plt.clf()
        
        plt.plot(hist_b, color='r') 
        # plt.hist(hist_b.ravel(), 256, [-128, 128])
        # plt.xlim([-128, 128])
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'lab_b_Cityscape')
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
    
        self.extract_HSV()
        
if __name__ == "__main__":
    ex = Extracter()
    ex.run()
    
    
