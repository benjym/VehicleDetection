# Vehicle Detection

Run the following code in the terminal:
```
git clone https://github.com/ultralytics/yolov5  # clone repo
cd yolov5
pip install -r requirements.txt  # install dependencies
```

Put the files you want to process in the folder `data/images/` and then run:

`python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --save-txt --source data/images/ > output.txt`

Have a look in `output.txt` to see all of the things found in each image. Boxes around each found object are saved as images/video in a subfolder in `runs/detect/`
