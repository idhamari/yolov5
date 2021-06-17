
# ~/myGitLab/DNN_ImageRegistration/IA/YOLOv5/yolov5$ clear && time python3 demoYolov5.py

## !git clone https://github.com/ultralytics/yolov5  # clone repo
# !git clone git@github.com:idhamari/yolov5.git
# %cd yolov5
# %pip install -qr requirements.txt  # install dependencies
# # matplotlib>=3.2.2  opencv-python>=4.1.2 Pillow PyYAML>=5.3.1 seaborn>=0.11.0
# # numpy>=1.18.5 scipy>=1.4.1  scikit-learn==0.19.2
# # torch>=1.7.0 torchvision>=0.8.1 tqdm>=4.41.0 tensorboard>=2.4.1 wandb
# # pandas coremltools>=4.1 onnx>=1.9.0  pycocotools>=2.0 thop

#import  wandb
import os, sys, time, torch
from IPython.display import Image, clear_output  # to display images

os.chdir("/home/ibr/myGitLab/DNN_ImageRegistration/IA/YOLOv5/yolov5")
clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# print("test detection ...................")
# os.system("python3 detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/")
# Image(filename='runs/detect/exp/zidane.jpg', width=600)

#
# print("downloading COCO 2017 ..................")
# torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip', 'tmp.zip')
# os.system("unzip -q tmp.zip -d ../ && rm tmp.zip")
#

# print("downloading COCO test-dev2017 ..................")
# torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip', 'tmp.zip')
# os.system("unzip -q tmp.zip -d ../ && rm tmp.zip # unzip labels")
# os.system('f="test2017.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f && rm $f  ') # 7GB,  41k images
# os.system("mv ./test2017 ../coco/images  ")# move to /coco

# # # Run YOLOv5s on COCO test-dev2017 using --task test
# os.system("python3 test.py --weights yolov5s.pt --data coco.yaml --task test")
#
# print("downloading COCO128 ..................")
# torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', 'tmp.zip')
# os.system("unzip -q tmp.zip -d ../ && rm tmp.zip")
#
# # Tensorboard  (optional)
#%load_ext tensorboard
# %tensorboard --logdir runs/train
#
# # Weights & Biases  (optional)
# %pip install -q wandb
#wandb.login()

train2d = int(sys.argv[1])
if train2d :
    print("train 2D ............................")
    # #Train YOLOv5s on COCO128 for 3 epochs
    os.system('python3 train.py --img 640 --batch 1 --epochs 3 --numChannels 3 --data coco128.yaml  --cfg "yolov5s.yaml" --weights yolov5s.pt --fromScratch 1 --cache --device 1 --workers 1 --numPoints 2')
else:
    print("train 3D ............................")
    # notes:
    #  69 3d volumes of 256,256,128: only 10 used
    #  69 txt files : only 10 used
    #  28 classes without background
    #  smallest approximate size: 80,70,20
    os.system('python3 train.py  --epochs 3 --batch 1 --szX 256 --szY 256 --szZ 128 --numChannels 1 --data spine3d.yaml --cfg "yolov5s3D.yaml" --fromScratch 1 --cache --device 1 --workers 1 --numPoints 3')

#
