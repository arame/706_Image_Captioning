import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from train import train
from config import Hyper, Constants
CUDA_LAUNCH_BLOCKING=1

# It all starts here
def main():
    print("\n" * 10)
    print("-" * 100)
    print("Start of Image Captioning")
    Hyper.display()
    print("-" * 100)
    train()

    print("\n" * 5)
    print("-" * 100)
    Hyper.display()
    print("End of Image Captioning")
    print("-" * 100)


if __name__ == "__main__":
    main()
