import os 
os.environ['PYTHONHASHSEED'] = '0'
from algorithm.train import train_loop

if __name__ == "__main__":
    train_loop()