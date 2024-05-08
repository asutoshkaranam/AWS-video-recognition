from fileinput import filename
from boto3 import client as boto3_client
import boto3
import os
import json
import urllib.parse
import math
import subprocess
import shutil

s3_input_bucket = "1227953352-input"
s3_output_bucket = "1227953352-stage-1"


def video_splitting_cmdline(video_filename):
    filename = os.path.basename(video_filename)
    outfile = os.path.splitext(filename)[0] + ".jpg"

    split_cmd = '/opt/python/ffmpeg -i ' + video_filename + ' -vframes 1 ' + '/tmp/' + outfile
    try:
        subprocess.check_call(split_cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.output)

    fps_cmd = '/opt/python/ffmpeg -i ' + video_filename + ' 2>&1 | sed -n "s/.*, \\(.*\\) fp.*/\\1/p"'
    fps = subprocess.check_output(fps_cmd, shell=True).decode("utf-8").rstrip("\n")
    return outfile

def lambda_handler(event, context):
    print("hello there!!!")
    # Get the S3 object key from the event
    key = event['Records'][0]['s3']['object']['key']

    # Download the video file from S3
    s3 = boto3.client('s3')
    lambda_client = boto3.client('lambda')
    
    tmp_dir = '/tmp'
    video_filename = os.path.join(tmp_dir, os.path.basename(key))
    s3.download_file(s3_input_bucket, key, video_filename)

    # Extract frames and upload to S3
    frame_ext = video_splitting_cmdline(video_filename)
    frame_path = os.path.join(tmp_dir, frame_ext)
    s3.upload_file(frame_path, s3_output_bucket, f"{frame_ext}")

    # Clean up temporary files
    os.remove(frame_path)
    os.remove(video_filename)
    
    function_name = 'face-recognition'
    payload = {
	    'payload' : {
	        'bucket_name': '1227953352-stage-1',
	        'image_file_name': frame_ext
	        }
    }

    invoke_params = {
	    'FunctionName': 'arn:aws:lambda:us-east-1:533266969005:function:face-recognition',
	    'InvocationType': 'Event',
	    'Payload': json.dumps(payload)
    }
    
    response = lambda_client.invoke(**invoke_params)

    print(f"Invocation response: {response}")

    return {
        'statusCode': 200,
        'body': 'Video frames extracted and uploaded to S3'
    }