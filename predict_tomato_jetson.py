from ultralytics import YOLO
import cv2
import supervision as sv
import time
import socket
import struct
import numpy as np

from line_counter_tomatoes import LineZone, LineZoneAnnotator
from box_annotator import BoxAnnotator

class TomatoesTracking():
    """
    A class for detect, track and count tomatoes by class (Ripen, Semi-ripe and Unripe).

    Attributes:
        input_video: path of input video
        weight_path: path of weight
        save_vid: flag (True & False)
        output_video: path of output video
    """
    def __init__(self,input_video=None, weight_path="weight/tomato_3classes_2000images_21_4_2023.pt", save_vid=False, output_video='tomatoes_predicted_part1.avi'):
        self.tomatoes_unripe = 0
        self.tomatoes_semiripe = 0
        self.tomatoes_ripen = 0
        self.time_stamp = time.ctime(time.time())

        self.model = YOLO(weight_path)
        self.box_annotator = BoxAnnotator(
                                            thickness=1,
                                            text_thickness=1,
                                            text_scale=0.25
                                            )
        
        LINE_START = sv.Point(120,0)
        LINE_END = sv.Point(120,480)
        self.line_counter = LineZone(start=LINE_START, end=LINE_END)
        self.line_annotator = LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

        self.video_capture = cv2.VideoCapture(input_video)
        self.output_video = output_video
        size = (640,480)

        if save_vid==True:
            self.video_record = cv2.VideoWriter(output_video,
                                                cv2.VideoWriter_fourcc(*'MJPG'),
                                                16, size)

    def tomatoes_tracking(self):
        self.loop_time = time.time()
        while(True):
            try: 
                if self.video_capture.isOpened():
                    ret, frame = self.video_capture.read()
                    frame = cv2.resize(frame, dsize=(640,480))
                    for result in self.model.track(source=frame, tracker='bytetrack.yaml',persist=True,stream=True):
                        detections = sv.Detections.from_yolov8(result)

                        if result.boxes.id is not None:
                            detections.tracker_id  = result.boxes.id.cpu().numpy().astype(int)

                        self.box_annotator.annotate( scene=frame,
                                                detections=detections,
                                                # labels=labels
                                                )
                        self.line_counter.trigger(detections=detections)
                        self.line_annotator.annotate(frame=frame, line_counter=self.line_counter)

                        self.tomatoes_ripen = self.line_counter.out_count_ripen
                        self.tomatoes_semiripe = self.line_counter.out_count_semiripe
                        self.tomatoes_unripe = self.line_counter.out_count_unripe

                    fps_text = 1/(time.time() - self.loop_time)
                    print('FPS: {}'.format(fps_text))
                    print(f"Tomatoes unripe, semi, ripen: {self.tomatoes_unripe}, {self.tomatoes_semiripe}, {self.tomatoes_ripen}")
                    
                    self.time_stamp = time.ctime(time.time())
                    cv2.rectangle(
                                    img=frame,
                                    pt1=(390,460),
                                    pt2=(640,480),
                                    color=[255,255,255],
                                    thickness=cv2.FILLED,
                                )
                    cv2.putText(
                                    img=frame,
                                    text=f"{self.time_stamp}",
                                    org=(395,475),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5,
                                    color=[0,0,0],
                                    thickness=1,
                                    lineType=cv2.LINE_AA,
                                )
                    
                    cv2.rectangle(
                                    img=frame,
                                    pt1=(0,390),
                                    pt2=(130,480),
                                    color=[20,20,20],
                                    thickness=cv2.FILLED,
                                )
                    cv2.putText(
                                    img=frame,
                                    text=f"FPS: {fps_text:0.2f}",
                                    org=(10,410),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5,
                                    color=[255,255,255],
                                    thickness=1,
                                    lineType=cv2.LINE_AA,
                                )
                    cv2.putText(
                                    img=frame,
                                    text=f"Fully-ripe: {self.tomatoes_ripen}",
                                    org=(10,430),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5,
                                    color=[0,0,255],
                                    thickness=1,
                                    lineType=cv2.LINE_AA,
                                )
                    cv2.putText(
                                    img=frame,
                                    text=f"Semi-ripe: {self.tomatoes_semiripe}",
                                    org=(10,450),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5,
                                    color=[0,255,255],
                                    thickness=1,
                                    lineType=cv2.LINE_AA,
                                )
                    cv2.putText(
                                    img=frame,
                                    text=f"Unripe: {self.tomatoes_unripe}",
                                    org=(10,470),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5,
                                    color=[0,255,0],
                                    thickness=1,
                                    lineType=cv2.LINE_AA,
                                )
                    
                    self.video_record.write(frame)
                    cv2.imshow("Tomatoes Tracking",frame)
                    self.loop_time = time.time()
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            except Exception as e:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                print(e)
            finally:
                pass
        self.video_capture.release()
        self.video_record.release()
        cv2.destroyAllWindows()
        print(f"The video {self.output_video} was successfully saved")

if __name__ == '__main__':
    tomatoes = TomatoesTracking(input_video="tomatoes_video/test_video1.mp4",
                                weight_path="weight/tomato_3classes_2000images_21_4_2023.pt",
                                save_vid=True,
                                output_video="test_video1_predicted.avi")
    tomatoes.tomatoes_tracking()