from boto3 import client as boto3_client
import boto3
import os
import imutils
import cv2
import json
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from shutil import rmtree
import numpy as np
import torch

# s3_stage_bucket = "1227953352-stage-1"
s3_output_bucket = "1227953352-output"
s3_resource_bucket = "awslambbda"

os.environ['TORCH_HOME'] = '/tmp/'

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

def face_recognition_function(key_path):
    # Face extraction
    img = cv2.imread(key_path, cv2.IMREAD_COLOR)
    boxes, _ = mtcnn.detect(img)

    # Face recognition
    key = os.path.splitext(os.path.basename(key_path))[0].split(".")[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face, prob = mtcnn(img, return_prob=True, save_path=None)
    saved_data = torch.load('/tmp/data.pt')  # loading data.pt file
    if face != None:
        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false
        embedding_list = saved_data[0]  # getting embedding data
        name_list = saved_data[1]  # getting list of names
        dist_list = []  # list of matched distances, minimum distance is used to identify the person
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))

        print(name_list[idx_min])
        # Save the result name in a file
        with open("/tmp/" + key + ".txt", 'w+') as f:
            f.write(name_list[idx_min])
        return name_list[idx_min]
    else:
        print(f"No face is detected")
    return


def handler(event, context):	
    
    # Process the event data
    print("Received an event:", event)

    # Read the payload data
    payload = event.get('payload', {})
    bucket_name = payload.get('bucket_name')
    image_file_name = payload.get('image_file_name')

    # Perform any necessary processings
    print(f"Payload data: key1={bucket_name}, key2={image_file_name}")
    
    # Download the image file from S3
    s3 = boto3.client('s3')
    tmp_dir = '/tmp'
    download_image_filename = os.path.join(tmp_dir, os.path.basename(image_file_name))
    s3.download_file(bucket_name, image_file_name, download_image_filename)

    print('Downloaded the Image from S3')

    # Download the data.pt file from S3
    datapt_filename = os.path.join(tmp_dir, 'data.pt')
    if os.path.exists(datapt_filename):
        print('Data.pt exists !!!')
    else:
        s3.download_file(s3_resource_bucket,'data.pt', datapt_filename)
        print('Downloaded the Data.pt from S3')

    result = face_recognition_function(download_image_filename)

    print(result)

    upload_file = os.path.splitext(os.path.basename(download_image_filename))[0].split(".")[0]
    upload_file_path = "/tmp/" + upload_file + ".txt"
    s3.upload_file(upload_file_path, s3_output_bucket, upload_file + ".txt")

    os.remove(download_image_filename)
    os.remove(upload_file_path)
    
    return {
        'statusCode': 200,
        'body': 'Result generated from the video and uploaded to S3 output Bucket'
    }