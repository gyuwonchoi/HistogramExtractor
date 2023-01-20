import argparse

parser = argparse.ArgumentParser(
        description="Pytorch Implementation for Histogram Extraction") 
parser.add_argument('--data-dir', type= str, default='./data')
parser.add_argument('--output-dir', type = str, default='./output')
