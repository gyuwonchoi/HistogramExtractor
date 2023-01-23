import argparse

parser = argparse.ArgumentParser(
        description="Pytorch Implementation for Histogram Extraction") 
parser.add_argument('--fz-dir', type= str, default='./Foggy_Zurich/Foggy_Zurich/RGB')
parser.add_argument('--cityscape-dir', type= str, default='./Cityscape/train')
parser.add_argument('--cityscape-fog-dir', type= str, default='./Cityscape_foggy/')
parser.add_argument('--output-dir', type = str, default='./output')
