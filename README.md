# DeepSORT_YOLOv5_Pytorch

## Prepare 
1 Create a virtual environment with Python >=3.8  
~~~
conda create -n py38 python=3.8    
conda activate py38   
~~~

2 Install pytorch >= 1.6.0, torchvision >= 0.7.0.
~~~
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
~~~


3 Install all dependencies
~~~
pip install -r requirements.txt
~~~

4 Download the yolov5 weight. 
I already put the `yolov5s.pt` inside. If you need other models, 
please go to [official site of yolov5](https://github.com/ultralytics/yolov5). 
and place the downlaoded `.pt` file under `yolov5/weights/`.   
And I also aready downloaded the deepsort weights. 
You can also download it from [here](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6), 
and place `ckpt.t7` file under `deep_sort/deep/checkpoint/`


## Run
~~~
# on video file
python main.py --input_path [VIDEO_FILE_NAME]

# on webcam 
python main.py --cam 0 --display
~~~



## Reference
1) [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)   
2) [yolov5](https://github.com/ultralytics/yolov5)  
3) [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)       
4) [deep_sort](https://github.com/nwojke/deep_sort)   

Note: please follow the [LICENCE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) of YOLOv5! 
