import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import time

COLOR = (255, 0, 0)
MARGIN = 5
FONT_SIZE = 1
FONT_THICKNESS = 1

def visualize(frame, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), (bbox.origin_x+bbox.width, bbox.origin_y+bbox.height), COLOR, 2)

        category = detection.categories[0]
        category_name = category.category_name
        score = round(category.score, 3) * 100
        text = f"{category_name}({str(score)}%)"
        text_location = (bbox.origin_x, bbox.origin_y - MARGIN)
        cv2.putText(frame, text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, COLOR, FONT_THICKNESS)

    return frame

ObjectDetector = vision.ObjectDetector
BaseOptions = mp.tasks.BaseOptions
ObjectDetectorOptions = vision.ObjectDetectorOptions
VisionRunningMode = vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='exported_model/model_int8.tflite'),
    score_threshold=0.35,
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture("videos/video (540p)(1).mp4")
cap_fps = cap.get(cv2.CAP_PROP_FPS)
print(cap_fps)

delay = int(1000 / cap_fps)

frame_number = 0
start_time_video = start_time_frame = time.time()

with ObjectDetector.create_from_options(options) as detector: 
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_number+=1
            # Calculate the timestamp of the current frame in milliseconds
            frame_timestamp_ms = int(1000 * frame_number / cap_fps)

            #gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Convert the frame to a mediapipe's image object
            rgb_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            #gray_mp_image = mp.Image(image_format=mp.ImageFormat.GRAY8,data=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            # Perform object detection on the video frame
            detection_result = detector.detect_for_video(rgb_mp_image, frame_timestamp_ms)
            annotated_frame = visualize(frame, detection_result)

            # Calculation of the frame rate in real time
            end_time_frame = time.time()
            fps = int(1 / (end_time_frame - start_time_frame))
            start_time_frame = end_time_frame
            cv2.putText(annotated_frame, str(fps), (7, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            #print(detection_result)

            # Display the frame on the screen
            cv2.imshow("Video", annotated_frame)
            
            # Check if the user has pressed the `q` key, if yes then close the program. 
            if cv2.waitKey(int(delay)) & 0xFF == ord('q'): # cv2.waitkey(delay)
                break
        else:
            break

end_time_video = time.time()
overall = end_time_video - start_time_video
avg_fps = frame_number / overall
print(f"Average FPS: {avg_fps}")

# Release the VideoCapture object
cap.release()

# Close all open windows
cv2.destroyAllWindows()         
