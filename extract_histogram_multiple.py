import matplotlib.pyplot as plt
import cv2
import os 
from args import parser 
import numpy as np

class Extracter():
    def __init__(self) -> None:
        pass
    
    def extract_LAB(self):
        print("LAB")
    
    def extract_RGB(self):
        img_dir=[]
        
        png_path = os.path.dirname(self.arg.data_dir)
        
        for path, dirs, files in os.walk(png_path):
            for file in files:
                    file_path = os.path.join(path, file)
                    _, ext = os.path.splitext(file_path)
                    if ext == '.png':
                        img_dir.append(file_path)
    
        hist_b = np.zeros((256,1))
        hist_g = np.zeros((256,1))
        hist_r = np.zeros((256,1))
        for idx, img in enumerate(img_dir):
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

        hist_b = hist_b / len(img_dir)
        hist_g = hist_g / len(img_dir)
        hist_r = hist_r / len(img_dir)
        
        plt.plot(hist_b, color='b') 
        plt.plot(hist_g, color='g') 
        plt.plot(hist_r, color='r') 
        
        save_dir = os.path.join(self.arg.output_dir, str(idx)+'bgr')
        plt.savefig(save_dir)
        plt.clf
    
    def run(self):
        print("running ...")
        self.arg = parser.parse_args()
    
        self.extract_RGB()
        
if __name__ == "__main__":
    ex = Extracter()
    ex.run()
    
    
