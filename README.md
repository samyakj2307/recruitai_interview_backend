# RecruitAI- An AI-Based Recruitment System

An Android App that makes online recruitements easy and effective using 4 AI models. 

Consists of 2 phases: 
* RESUME ANALYSIS PHASE:

  -  The model compares the job description and resume uploaded and gives a similarity score. Both the documents are taken in pdf format and converted to docx and then to text at the backend. Then, both the documents are passed through a count vectorizer function. A count vectorizer is a sckit learn module in python which tokenizes the text along and does preprocessing such as removal of stop words and punctuations. We have used cosine similarity which is a metric used to measure how similar 2 documents are irrespective of their size. At the end the model gives us similarity score in a matrix form through a variable sparse matrix using fit_transform.
  

* ONLINE INTERVIEW PHASE:
  - For the Interview part, the video is uploaded by the candidate.

  1. Face confidence: The video is split and images are extracted every 5 seconds. These images are run through the deep learning model which classifies them as confident or non-confident. We have trained a convolutional neural network on an image dataset we found on Kaggle. We have made a keras sequential model and we have added many convolution layers and activation functions. We have taken an average of the confidence score of all the images as our final confidence score.

  2. Voice Emotions Model: The audio of the interview video is extracted and analyzed for every 0.1 second. The audio is run through a deep learning model which classifies the audio into 7 emotions: neutral, happy, surprised, sad, fearful, disgusted and angry. For reading the audio file we have used Librosa library and then converted the audio data into NumPy array which is then classified using the CNN.

  3. Speech to text: We have used speech recognition library. We have made a function which takes our audio file and sends it to Google cloud speech API which returns the transcript of the audio. We have included language support for 5 languages: English (India), English (USA), Hindi, Tamil and Telugu.


## Final analysis sample: 


![ss](https://user-images.githubusercontent.com/56762107/121242217-eddb2880-c8b9-11eb-83de-811a3d44bc8e.png)


### Android App Repository:
https://github.com/adyaagrawal/Recruit-AI

### Backend repository for Interview Analysis:
https://github.com/samyakj2307/recruitai_interview_backend

### Backend Repository for Resume Analysis:
https://github.com/samyakj2307/recruitai_resume_backend
