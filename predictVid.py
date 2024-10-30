import cv2
import torch
from ultralytics import YOLOv10 as YOLO

# see if you need to change the training set
model = YOLO('./runs/detect/train14/weights/last.pt')  # or another version of YOLOv10 (e.g., yolov10s.pt for small)

# Load the video file
input_video_path = '/home/xavier/Robolab/videos/video3.mp4'
output_video_path = '/home/xavier/Robolab/videos/output.mp4'

# Open the video using OpenCV
video_capture = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

dict = {0: 'weed', 1:'lettuce'}
# Iterate over each frame
frame_count = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read a frame
    if not ret:
        break
    
    results = model(frame)[0]

    # Apply YOLOv10 object detection
    img = frame.copy()
    boxes = results[0].boxes

    # Iterate through the detections and draw bounding boxes
    for box in boxes:  # Each detection in the format [x1, y1, x2, y2, conf, class]
        coords = box.xyxy[0].cpu().numpy()
        conf = float(box.conf)
        cls = int(box.cls)
            
        if conf > .3:
            x1, y1, x2, y2 = map(int, coords)
                
                # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
            label = f'Class {dict[cls]}: {conf:.2f}'
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
    # Write the processed frame to the output video

    out_video.write(img)
    
    # Print progress
    frame_count += 1
    print(f'Processed frame {frame_count}/{total_frames}')

# Release resources
video_capture.release()
out_video.release()
cv2.destroyAllWindows()

print(f'Output video saved to {output_video_path}')