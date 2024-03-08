#!/Users/m/miniconda3/envs/mlenv/bin/python
import cv2
import threading
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import time

class StreetVisionApp:
    def __init__(self, video_source):
        
        self.count = 0
        self.buffer = []
        self.video_source = video_source
        self.processor = PersonProcessor()
        self.media_stream = MediaStream(self.video_source)

    def start(self):
        """Start the surveillance process"""

        
        fc =0
        while self.media_stream.isReady():
            success, frame = self.media_stream.read()
            if not success or fc == 100:
                break

            self.processor.detect_persons(frame)

        self.processor.printPeople()

    def stopSurveillance(self):
        """Stop the surveillance process"""
        self.media_stream.stop()


class AppearanceExtractor:
    """ Class to store the appearance attributes of a single person"""
    def __init__(self, model='/Users/m/Desktop/SurveillanceProject/StreetVisionEdgeApp/StreamApp/models/clothing-3.pt'):

        self.model = ObjectDetection(model=model)
    
    def ext_appearance(self, croppedRegion):
        """Detect the appearance attributes of the person"""
        results = self.model.detect_objects(croppedRegion)
        det_classes = self.model.getDetObjClass()
        rois = self.model.cropDetections()

        item_class = []
        item_colour = []
        item_confidence = []

        for i, roi in enumerate(rois):
            if det_classes[i] in ['tie', 'shirt', 'jacket', 'coat', 'dress']:
                hist = HistogramColourEstimator(roi)
                item_class.append(det_classes[i])
                item_colour.append(hist.estimate_dominant_colour())
                item_confidence.append(self.model.getObjConfidence())

        appearance = {'class': item_class, 
                      'colour': item_colour, 
                      'confidence': item_confidence}

        return appearance

class PersonProcessor:
    
    def __init__(self, detectAttributes=True):
        self.softIds = set()
        self.detectAttributes = True
        self.people_count = 0
        self.model = ObjectDetection(taskType='track') #initialize here, to keep open
        self.appearance = AppearanceExtractor()
        self.peopleCollection = []
        self.results = None

    def add_person(self, person):
        # self.people.append(person)
        pass

    def detect_persons(self, frame):   
        self.results = self.model.detect_objects(frame)

        det_time = time.time()
        
        # check if already processed. Will come back to this as i may want to get best score of x inference
        det_ids = list(map(int, self.model.getIds()))
        scope = set(det_ids).difference(self.softIds) # items not already processed

        # persons not processed (get the index position, now we can fiter)
        scope_indexes = [index for index, value in enumerate(det_ids) if value in scope]

        cropped_regions = self.model.cropDetections(scope_indexes)

        for i, region in enumerate(cropped_regions):
            id = det_ids[i]
            appearance = self.appearance.ext_appearance(region)
            appearance['id'] = id
            appearance['timestamp'] = det_time
            self.peopleCollection.append(appearance)

    def printPeople(self):
        for person in self.peopleCollection:
            print(person)



class MediaStream:
    """Stream video from a camera or a video file using threading, need to revise threading approach"""
    def __init__(self, source=0, max_retries=3):
        self.capture = cv2.VideoCapture(source)
        self.frame = None
        self.ready = self.capture.isOpened()

    def open_video_source(self, source, max_retries):

        """ IMplements timer as sometimes it takes a moment to initialise the camera"""
        for attempt in range(max_retries):
            self.capture = cv2.VideoCapture(source)
            if self.capture.isOpened():
                break  # Successfully opened the device
            print(f"Failed to open video source on attempt {attempt + 1}. Retrying...")
            time.sleep(attempt + 1) # Wait before retrying
        else:  
            raise ValueError("Failed to open video capture device after maximum retries")
        
        return self.capture.isOpened()
    
    def read(self):
        """Read the next frame from the video source"""
        if self.capture.isOpened():
            success, frame = self.capture.read()
            if success:
                self.frame = frame
            else:
                print("Failed to read frame from video source")
        else:
            print("Video source is not open")
        return success, frame

    def isReady(self):
        return self.ready
    
    def stop(self):
        """Release the video source"""
        if self.capture.isOpened():
            self.capture.release()
            self.ready = False
        else:
            print("Video source is not open")   

class ObjectDetection:
    """YOLO Object Detection Class"""
    def __init__(self, model='yolov8n.pt', taskType=None):
        self.model = YOLO(model)
        self.taskType = taskType
        self.results = None
        self.device = "cpu"
        self.dectectCount = 0

    def detect_objects(self, frame):

        if self.taskType == None:
            self.results = self.model(frame)
        elif self.taskType == 'track':
            self.results = self.model.track(frame, device=self.device)
        return self.results

    def getObjCount(self):
        if self.results[0] != None:
            self.dectectCount = len(self.results[0])
        return self.dectectCount
    
    def getObjConfidence(self):
        return list(map(int, self.results[0].boxes.conf))
    
    def getDetObjClass(self, format='str'):
        """Get the class of the detected object: string mapping or integer mapping."""
        if format == 'str':
            return [self.model.names[int(cls)] 
                    for cls in self.results[0].boxes.cls]
        else:
            return list(map(int,self.results[0].boxes.cls.tolist()))

    def getIds(self):
        if self.results[0] == None:
            return None
        if (self.results[0].boxes.is_track):
            return list(map(int, self.results[0].boxes.id))
        else:
            return []
    
    def pltDetections(self, frame):
        """Display the detected objects on the frame"""
        if self.results is None or self.results[0] is None:
            return None
        return self.results[0].plot()


    def cropDetections(self, unique_indexes=None):
        """Crop the detected object from the img, returns coods of the bounding box."""

        if self.results[0] is None:
            return None
        
        croppedRegions = []
        boxCoords = self.results[0].boxes.xyxy.tolist()

        # If unique_indexes is not provided, use all indexes
        if unique_indexes is None:
            unique_indexes = range(len(boxCoords))

        for i in unique_indexes:  # Loop through the specified indexes or all indexes (avoids multiple detections of same object)
            if i < len(boxCoords):  # Ensure the index valid
                x1, y1, x2, y2 = map(int, boxCoords[i])  # Convert coordinates to integers
                croppedRegions.append(self.results[0].orig_img[y1:y2, x1:x2])
            
        return croppedRegions


class HistogramColourEstimator:
    """ Estimate colour of inoput region using histogram analysis"""
    def __init__(self, region):
        self.hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    def estimate_dominant_colour(self):
        # Compute histograms for each channel (H, S, V)
        hist_hue = cv2.calcHist([self.hsv], [0], None, [180], [0, 180])
        hist_sat = cv2.calcHist([self.hsv], [1], None, [256], [0, 256])
        hist_val = cv2.calcHist([self.hsv], [2], None, [256], [0, 256])

        # Get dom hue from peak in histogram
        dominant_hue = np.argmax(hist_hue)

        # use ave sat and val to determine to refine colour mapping
        average_sat = np.mean(self.hsv[:, :, 1])
        average_val = np.mean(self.hsv[:, :, 2])

        # Map dominant hue to colour name
        color_name = self.map_hue_to_colour(dominant_hue, average_sat, average_val)
        return color_name

    def map_hue_to_colour(self, hue, sat, val):
        # hue ranges for basic colours
        color_ranges = {
            'Red': range(0, 8),
            'Orange': range(8, 18),
            'Yellow': range(18, 30),
            'Green': range(30, 80),
            'Cyan': range(80, 100),
            'Blue': range(100, 130),
            'Violet': range(130, 160),
            'Pink': range(160, 179),
        }

        # Low sat colors
        if sat < 40:  # Low saturation
            if val < 50:
                return 'Black'
            elif val > 200:
                return 'White'
            else:
                return 'Grey'
        else:  #  sat colors by hue
            for color_name, hue_range in color_ranges.items():
                if hue in hue_range:
                    return color_name
        return "Unknown"  # Doesn't match any predefined color


import matplotlib.pyplot as plt
video_path = '/Users/m/Desktop/SurveillanceProject/peoplewalking.mp4'
print('hi')
app = StreetVisionApp(0)
app.start()

app.stopSurveillance()


# while True:
#     current_frame = app.media_stream.get_current_frame()
#     if current_frame is not None:
#         cv2.imshow('Frame', current_frame)
#         cv2.waitKey(1)  # Wait for a key press for a short period
#     else:
#         print("No Frames")
#         # break
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # time.wait(10)
# app.media_stream.stop() 


# # You can now access the current frame from the video as follows:
# # current_frame = app.get_current_frame()

# # Don't forget to call app.media_stream.stop() when you're done to clean up the resources.
