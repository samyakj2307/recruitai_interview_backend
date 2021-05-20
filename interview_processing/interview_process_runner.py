import os, shutil
import math

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import moviepy.editor as mp
import pyrebase
import speech_recognition as sr
from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings

audiopath = str(settings.BASE_DIR) + "/staticfiles/interviewprocessing_files/audiofiles"
videopath = str(settings.BASE_DIR) + "/staticfiles/interviewprocessing_files/interview_video.mp4"
imagespath = str(settings.BASE_DIR) + "/staticfiles/interviewprocessing_files/InterviewVideoImages"
modelpath = str(settings.BASE_DIR) + "/staticfiles/interviewprocessing_files/mymodelhistory1.h5"
faceCascadepath = str(settings.BASE_DIR) + "/staticfiles/interviewprocessing_files/haarcascade_frontalface_default.xml"
eyeCascadepath = str(settings.BASE_DIR) + "/staticfiles/interviewprocessing_files/haarcascade_eye.xml"

seviceAccountKeypath = str(settings.BASE_DIR) + "/staticfiles/interviewprocessing_files/serviceAccountKey.json"


def audio_extracter():
    os.mkdir(audiopath)
    clip = mp.VideoFileClip(videopath)
    duration = math.floor(clip.duration)
    clipname = "subclip1"
    subclip = clip.subclip(0, min(60, duration))
    subclip.audio.write_audiofile(audiopath + "/" + clipname + ".wav")
    duration -= subclip.duration
    duration_start = 1
    i = 2
    while duration > 1:
        duration_start += 60
        clipname = "subclip" + str(i)
        duration_end = min(duration_start + 60, duration_start + duration - 1)
        subclip = clip.subclip(duration_start, duration_end)
        subclip.audio.write_audiofile(audiopath + "/" + clipname + ".wav")
        duration -= subclip.duration
        i += 1
    clip.close()


def video_analyser():
    mmh1 = load_model(modelpath)
    faceCascade = cv2.CascadeClassifier(faceCascadepath)
    eye_cascade = cv2.CascadeClassifier(eyeCascadepath)
    vidcap = cv2.VideoCapture(videopath)
    predictions = []

    def getframe(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasframes, image = vidcap.read()
        if hasframes:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                image,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = image[y:y + h, x:x + w]
                
                eyes = eye_cascade.detectMultiScale(roi_color)
                if len(eyes)>0:
                    my_img = cv2.resize(roi_color, (64, 64))
                    x_train = np.asarray(my_img)
                    x_train = x_train.reshape((1,64,64,3))
                    prediction = 100*(mmh1.predict(x_train))
                    predictions.append(prediction)

            
        return hasframes

    sec = 0
    framerate = 5
    count = 1
    success = getframe(sec)
    while success:
        count = count + 1
        sec = sec + framerate
        sec = round(sec, 2)
        success = getframe(sec)
    
    confidence_list = []
    for prediction in predictions:
        confidence_list.append(prediction[0][0])
        
    return np.mean(confidence_list)


def speech_to_text(audio_file, audio_language):
    t = sr.Recognizer()
    z = sr.AudioFile(audio_file)
    with z as audio_file:
        audio_content = t.record(audio_file)
    return t.recognize_google(audio_content, language=audio_language)


def speech_to_text_generator(inputlanguage):
    audio_text = ''
    for filename in os.listdir(audiopath):
        audio_text += speech_to_text(audiopath + "/" + filename, inputlanguage)
    return audio_text


def clean_directory_video():
    if os.path.isdir(audiopath):
        shutil.rmtree(audiopath)
    if os.path.isfile(videopath):
        os.remove(videopath)


logger = get_task_logger(__name__)


@shared_task(name="processInterviewVideo")
def process_video(user_id, company_id):
    clean_directory_video()
    firebaseConfig = {
        "apiKey": "AIzaSyBZReE0HUqcTKagQCPU5HwDrKrBJsW787A",
        "authDomain": "recruit-ai-cb3c1.firebaseapp.com",
        "projectId": "recruit-ai-cb3c1",
        "databaseURL": "https://recruit-ai-cb3c1-default-rtdb.firebaseio.com",
        "storageBucket": "recruit-ai-cb3c1.appspot.com",
        "messagingSenderId": "1012646425167",
        "appId": "1:1012646425167:web:e9c3d8c58927d64078303a",
        "measurementId": "G-G87J196BTV",
        "serviceAccount": seviceAccountKeypath,
    }

    firebase = pyrebase.initialize_app(firebaseConfig)
    db = firebase.database()
    # language = db.child('users').child(userid).child('language').get().val()
    language = "en-IN"
    storage = firebase.storage()
    file_name = "users/" + user_id + '/interview_video.mp4'
    storage.child(file_name).download(videopath)
    confidence_score = video_analyser()
    audio_extracter()
    audio_text = speech_to_text_generator(language)
    print(audio_text)
    db.child("Jobs").child(company_id).child("Juser").child("0").child("audio_text").set(audio_text)
    db.child("Jobs").child(company_id).child("Juser").child("0").child("confidence_score").set(str(confidence_score))
    db.child("Jobs").child(company_id).child("Juser").child("0").child("status").set("Interview Processed")
    logger.info("Video Processed")
    clean_directory_video()

    return "Video Processing ended"
