#!/Users/m/miniconda3/envs/mlenv/bin/python
import cv2
import threading
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from collections import deque
import datetime as dt
from elasticsearch import Elasticsearch
from elasticsearch import helpers

class Person:
    def __init__(self, id,  inference_score=0, img=None):
        self.id = id
        self.first_seen = dt.datetime.now()
        self.last_seen = dt.datetime.now()
        self.setScore(inference_score)
        self.img = img
        
        self.attributes = {"score": [], "class": [], "colour": []}

    def setAttributes(self, attributes):
        self.attributes = attributes

    def getAttributes(self):
        return self.attributes
    
    def getAttrConfAverage(self):
        if self.attributes['score']:  # Check if the list is not empty
            return np.mean(self.attributes['score'])
        else:
            return 0

    def setScore(self, inference_score):
        self.inference_score = inference_score

    def setFirstSeen(self):
        return dt.datetime.now()

    def setLastSeen(self, timestamp):
        self.last_seen = timestamp

    def __str__(self):
        return (f"Person(id={self.id}, attributes={self.attributes}, "
                f"score={self.inference_score}, first_seen={self.first_seen}, last_seen={self.last_seen})")

    def __repr__(self):
        return self.__str__()
    
    def setImgCrop(self, img):
        self.img = img

    def getImgCrop(self):
        return self.img

    def toJSON(self):
        return {
            "personId": self.id,
            "attributes": self.getAttributes(),
            "inferenceScore": self.inference_score,
            "firstSeen": self.first_seen.isoformat(),
            "lastSeen": self.last_seen.isoformat(),
        }
    
class ObjectDetection:
    """YOLO Object Detection Class"""
    def __init__(self, model='yolov8n.pt', taskType=None, filterClasses=None, device="mps"):
        self.model = YOLO(model)
        self.taskType = taskType
        self.results = None
        self.device = device
        self.dectectCount = 0
        self.classScope=list(self.model.names.keys()) if filterClasses is None else filterClasses
        self.currentFrame = None

    def setClasses(self, classes):
        if len(classes) > 0:
            self.classes = set(classes)

    def setCurrentFrame(self, frame):
        self.currentFrame = frame

    def detectObjects(self, frame):
        self.setCurrentFrame(frame)
    
        if self.taskType == None:
            self.results = self.model(self.currentFrame, device=self.device, classes=self.classScope, verbose=False)
        elif self.taskType == 'track':
            self.results = self.model.track(self.currentFrame, device=self.device, classes=self.classScope,persist=True, verbose=False)
        return self.results

    def getObjCount(self):
        if self.results[0] != None:
            self.dectectCount = len(self.results[0])
        return self.dectectCount
    
    def getObjConfidence(self):
        # returns confidence as a list
        return self.results[0].boxes.conf.tolist() # convert from tensor to list
    
    def getDetObjClass(self, format='str'):
        """Get the class of the detected object: string mapping or integer mapping."""
        if format == 'str':
            return [self.model.names[int(cls)] 
                    for cls in self.results[0].boxes.cls]
        else:
            return list(map(int,self.results[0].boxes.cls.tolist()))

    def getIds(self):
        # if self.results[0] == None:
        #     return None
        if (self.results[0].boxes.is_track):
            return list(map(int, self.results[0].boxes.id))
        else:
            return []
    
    def plot(self):
        """Display the detected objects on the frame"""
        # if self.results is None or self.results[0] is None or self.results[0].boxes.id is None:
        #     return self.results[0].orig_img
        return self.results[0].plot()


    def cropDetections(self):
        """Crop the detected object from the img, returns coods of the bounding box."""

        if self.results[0] is None:
            return None
        
        croppedRegions = []
        boxCoords = self.results[0].boxes.xyxy.tolist()

        for box in boxCoords:  # Loop through the specified indexes or all indexes (avoids multiple detections of same object)  # Ensure the index valid
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            croppedRegions.append(self.results[0].orig_img[y1:y2, x1:x2])
            
        return croppedRegions
    
    def isSuccess(self):
        return self.results[0] != None
    
    def getIndexById(self, id):
        if id in self.results[0].boxes.id:
            return self.results[0].boxes.id.index(id)
        return None

class ThreadedMediaStream:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.ready = self.capture.isOpened()
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        self.buffer = deque(maxlen=10) # Store last 2 frames in ring buffer

        if self.ready:
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()

    def update(self):
        print("Media thread started") 
        while self.ready and not self.stopped:
            success, frame = self.capture.read()
            if success:
                with self.lock:
                    self.frame = frame
                    self.buffer.append(frame)
            else:
                print("Failed to read frame") 
                self.ready = False

    def read(self):
        with self.lock:
            if len(self.buffer) > 0:
                return True, self.buffer[-1]  # Return latest frame of buffer
            else:
                return False, None

    def stop(self):
        self.stopped = True
        self.thread.join()
        if self.capture.isOpened():
            self.capture.release()


    def isReady(self):
        return self.ready



class PeopleDetectionStore:
    def __init__(self):
        self.people = []  
        self.attrInferenceQueue = []  # List of person IDs that need their attributes updated

    def addPerson(self, person):
        if isinstance(person, Person):
            self.people.append(person)
            if person.id not in self.attrInferenceQueue:
                self.attrInferenceQueue.append(person.id)
        else:
            raise TypeError("addPerson expects an instance of Person")

    def getPersonById(self, id):
        for person in self.people:
            if person.id == id:
                return person
        return None

    def updateBestAttributes(self, id, newAttributes):
        if (person := self.getPersonById(id)) is None:
            return False

        currAttributes = person.getAttributes()

        if not all(key in currAttributes for key in ['class', 'score', 'colour']):
            return False
        # print(f"attempting to update {newttributes}")
        for i, newClass in enumerate(newAttributes['class']):
            # Check if class exists
            if newClass in currAttributes['class']:
                index = currAttributes['class'].index(newClass)
                if newAttributes['score'][i] > currAttributes['score'][index]:
                    currAttributes['score'][index] = newAttributes['score'][i]
                    currAttributes['colour'][index] = newAttributes['colour'][i]
            else:
                currAttributes['class'].append(newClass)
                currAttributes['score'].append(newAttributes['score'][i])
                currAttributes['colour'].append(newAttributes['colour'][i])

        # Update the person's attributes
        person.setAttributes(currAttributes)
        return True

    def removePerson(self, id):
        self.people = [person for person in self.people if person.id != id]
        self.attrInferenceQueue = [person_id for person_id in self.attrInferenceQueue if person_id != id]

    def print(self):
        for person in self.people:
            print(person)

    def getPendingAttrInfList(self):
        return self.attrInferenceQueue

    def clearInactivePersons(self, activePersonIds):
        """ Handles runtime of people in the scene. If left, clear the buffer."""
        for person in self.people:
            if person.id not in activePersonIds:
                self.removePerson(person.id)
                if person.id in self.attrInferenceQueue:
                    self.attrInferenceQueue.remove(person.id)

    def personExists(self, id):
        for person in self.people:
            if person.id == id:
                return True
        return False
    
    def getInferenceScore(self, id):
        for person in self.people:
            if person.id == id:
                return person.inference_score
        return None
    
    def updateScore(self, id, score):
        for person in self.people:
            if person.id == id:     
                person.setScore(score)
                if id not in self.attrInferenceQueue:
                    self.attrInferenceQueue.append(id)
                return True
        return False
    
    def updateLastSeen(self, id, timestamp):
        for person in self.people:
            if person.id == id:
                person.setLastSeen(timestamp)
                return True
        return False
    
    def updateImgCrop(self, id, img):
        for person in self.people:
            if person.id == id:
                person.setImgCrop(img)
                if id not in self.attrInferenceQueue:
                    self.attrInferenceQueue.append(id)
                return True
        return False    
    
    def removeFromInferenceQueue(self, id):
        if id in self.attrInferenceQueue:
            self.attrInferenceQueue.remove(id)
            return True
        return False

    def collectionToJSON(self):
        docs = []
        for person in self.people:
            item = person.toJSON()
            docs.append(item)
        return docs

class ESClient:
    def __init__(self, host="localhost", port=9200, index="people_detection", scheme="http"):
        self.host = host
        self.port = port
        self.index = index
        self.es = Elasticsearch([{'host': self.host, 'port': self.port, 'scheme':scheme}])

    def write(self, data):
        self.es.index(index=self.index, document=data)


    def indexExists(self):
        return self.es.indices.exists(index=self.index)

    def bulkWrite(self, docs):
        actions = [
            {
                "_index": self.index,
                "_op_type": "index",
                "_source": doc
            }
            for doc in docs
        ]
        return helpers.bulk(self.es, actions)

class HistogramColourEstimator:
    """Estimate colour of input region using histogram analysis."""
    def __init__(self, region):
        self.hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    def estimate_dominant_colour(self):
        # Compute histogram for the hue channel
        hist_hue = cv2.calcHist([self.hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist_hue)

        # Calculate median saturation and value
        median_sat = np.median(self.hsv[:, :, 1])
        median_val = np.median(self.hsv[:, :, 2])

        # Map dominant hue to colour name
        color_name = self.__map_hue_to_colour(dominant_hue, median_sat, median_val)
        return color_name

    def __map_hue_to_colour(self, hue, sat, val):
        # Define hue ranges for basic colours using approximate midpoints
        color_ranges = {
            'Red': (0, 10),
            'Yellow': (30, 50),
            'Green': (60, 90),
            'Light_Blue': (90, 110),
            'Blue': (120, 140),
            'Magenta': (150, 170),
            # Red also appears near the end of the hue range
            'Red2': (170, 180)
        }

        # Identify black, white, gray, and silver based on saturation and value
        if sat < 25:  # Threshold for non-saturated colors
            if val < 50:
                return 'Black'
            elif val > 180:
                return 'White'
            elif val > 120:
                return 'Silver'
            else:
                return 'Gray'
        else:
            # Coloured regions
            for color_name, (lower, upper) in color_ranges.items():
                if lower <= hue <= upper:
                    return color_name.replace('2', '')  # Consolidate the Red range

        return "Unknown"

if __name__ == "__main__":
    
    # Datbase setiup
    esClient = ESClient(host="localhost", port=9200, index="people_detection")
    ES_WRITE_INTERVAL = 10
    lastWriteTime = None

    # Detection confidence threshold (currently used by both person and attribute detection models)
    MIN_CONF_THRESHOLD = 0.5

    personModel = "yolov8n.pt"
    attrModel = "/Users/m/Desktop/SurveillanceProject/WorkingModels/clothing-20.pt"
    
    fc = 0 # for frame count during debugging

    # Initialize person and attribute detection models
    personDetection = ObjectDetection(taskType='track', filterClasses=[0])  # [0] = Filter for person class
    attributeDetection = ObjectDetection(model=attrModel)  

    # Set up camera stream and store
    # cameraStream = ThreadedMediaStream('/Users/m/Desktop/SurveillanceProject/peoplewalking.mp4')  # Use camera 1, change to 0 if using laptop webcam or phone
    cameraStream = ThreadedMediaStream(1)  # Use camera 1, change to 0 if using laptop webcam or phone
    peopleDetectionStore = PeopleDetectionStore() # Store people in the scene

    print("Surveillance Running...")

    while cameraStream.isReady():
        success, frame = cameraStream.read()
        if frame is None: continue

        currentTs = dt.datetime.now()

        # Run person detection
        personDetection.detectObjects(frame)

        if personDetection.isSuccess():

            # Get the IDs of the detected people
            detIds = personDetection.getIds()
            
            """ PART1:  PERSON DETECTION LOOP:
            - Check if the person is already in the store:

            - PERSON EXISTS:
            - If the person exists, update their attributes if the new score is higher.

            - PERSON DOES NOT EXIST:
            - add them to the store.

            Note: Inference queue is used track required sub-inferences (i.e. clothing detection).
        
            """
            for idx, id in enumerate(detIds):

                if (confScore := personDetection.getObjConfidence()[idx]) < MIN_CONF_THRESHOLD: # get confidence score
                    continue
                
                imgCrop = personDetection.cropDetections()[idx] # crop the detected person from the frame
                
                if peopleDetectionStore.personExists(id):
                    # Update the person's attributes if the new score is higher
                    if confScore > peopleDetectionStore.getInferenceScore(id):
                        peopleDetectionStore.updateScore(id, confScore)
                        peopleDetectionStore.updateImgCrop(id, imgCrop)
                    peopleDetectionStore.updateLastSeen(id, dt.datetime.now()) # always update the last seen time

                else:
                    peopleDetectionStore.addPerson(Person(id, confScore, imgCrop))


        """ PART2: Get clothing descriptions for each person pending inference (in the queue)
            - For each person in the store, run clothing detection
            - Get the person's image crop
            - Run clothing detection on the person's image crop
            - Update the person's attributes, if attributes exist, update those that have higher confidence
            - remove from inference queue if average confidence is above threshold
        """
        for id in peopleDetectionStore.getPendingAttrInfList():
            # Get the person's image crop
            person = peopleDetectionStore.getPersonById(id)
            personCrop = person.getImgCrop()
            
            if personCrop is None:
                continue

            # Run clothing detection on the person's image crop
            attributeDetection.detectObjects(personCrop)
            detClasses = attributeDetection.getDetObjClass()
            
            # get array of cropped regions of interest
            rois = attributeDetection.cropDetections()
            itemClass = []
            approxColour = []
            attrConfScore = []
            

            for i, roi in enumerate(rois):                
                hist = HistogramColourEstimator(roi) # set up histogram estimator
                itemClass.append(detClasses[i])
                approxColour.append(hist.estimate_dominant_colour())
                attrConfScore.append(attributeDetection.getObjConfidence()[i])

            attributes = {'class': itemClass, 
                        'colour': approxColour, 
                        'score': attrConfScore}
            
            # Remove from inference queue if confidence is above threshold (will get re-added if next detection for same person is higher)
            peopleDetectionStore.updateBestAttributes(id, attributes)
            if person.getAttrConfAverage() > 20:
                peopleDetectionStore.removeFromInferenceQueue(id)
                
        
        """ PART3: Write to DB and clean up inactive people from collection"""
        if currentTs.second % ES_WRITE_INTERVAL == 0:
            if lastWriteTime is None or (currentTs - lastWriteTime).total_seconds() >= ES_WRITE_INTERVAL:
                batch = peopleDetectionStore.collectionToJSON()
                esClient.bulkWrite(batch)
                lastWriteTime = currentTs  # Update the last DB write time

                """ Only Clear inactive persons from the store after DB write"""
                peopleDetectionStore.clearInactivePersons(personDetection.getIds())


    cameraStream.stop()
    cv2.destroyAllWindows()