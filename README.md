# visionless

automated object tracking and reframing.
it sees everything so you don't have to.
capturing the beauty and emptiness of machine vision.
result is an interesting to watch video

## features
- tracks/identifies objects using yolov8x
- tracking IDs (doesnt lose track easily)
- optimized for gpu (cuda for nvidia, directml for amd)
- smart frame skipping for speed
- auto reframes to 9:16 vertical format
- background subtraction to focus on movement


https://prnt.sc/5ZdArnUDwv9k (it catching snowflakes)

# demo (quality is bad because its compressed)
https://github.com/user-attachments/assets/d5f221f3-f0e7-4a90-8b63-f0c25ab96e80




## how to use
basic usage:
`python vision.py --input video.mp4`

custom output:
`python vision.py --input video.mp4 --output result.mp4`

## install
needs python 3.10+ (tested on 3.11)

1. install torch (check pytorch.org for your specific gpu version)
2. install the rest:
`pip install ultralytics opencv-python onnxruntime-directml`

(if you have an amd card, onnxruntime-directml is the solution)

powered by YOLO and an tracking algo


https://synucu.cat
