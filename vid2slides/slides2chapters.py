import argparse
import cv2
import glob
import json
import moviepy.editor as mpy
import os
import subprocess
import tempfile


def extract_chapters(sequence):
    rgy = slice(sequence['crop'][1], sequence['crop'][1] + sequence['crop'][3])
    rgx = slice(sequence['crop'][0], sequence['crop'][0] + sequence['crop'][2])
    
    sources = []

    chapters = ['00:00 Start']
    for _, el in enumerate(sequence['sequence']):
        if el['type'] == 'slide' and el['title']:
            if el['source'] not in sources:
                sources.append(el['source'])
                chapters.append(el['start_time'] + ' ' + el['title'])

    print('\n'.join(chapters))


if __name__ == "__main__":
    desc = """Creates a chapter listing that you can copy paste into a YouTube 
description from a list of slides. Prints to STDOUT.

Read more about this feature here: 
https://support.google.com/youtube/answer/9884579?hl=en
"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("in_path", help='Input json path')
    
    args = parser.parse_args()

    with open(args.in_path, 'r') as f:
        extract_chapters(json.load(f))