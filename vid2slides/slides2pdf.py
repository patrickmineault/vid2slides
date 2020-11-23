import argparse
import cv2
import glob
import json
import moviepy.editor as mpy
import os
import subprocess
import tempfile


def extract_pdf(sequence, output_file):
    rgy = slice(sequence['crop'][1], sequence['crop'][1] + sequence['crop'][3])
    rgx = slice(sequence['crop'][0], sequence['crop'][0] + sequence['crop'][2])
    
    sources = []
    for _, el in enumerate(sequence['sequence']):
        if el['type'] == 'slide' and el['title']:
            im = cv2.imread(el['source'])
            filename = el['source'][:-4] + '.cropped.png'
            cv2.imwrite(filename, im[rgy, rgx])
            if filename not in sources:
                sources.append(filename)


    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('\n'.join(sources))

    if output_file.endswith('.pdf'):
        output_file = output_file[:-4]

    subprocess.run([
        "tesseract",
        f.name,
        output_file,
        "pdf"], shell=False)


if __name__ == "__main__":
    desc = """Create gif from sequence of slides in json. Use extract_keyslides 
to create json first"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("in_path", help='Input json path')
    parser.add_argument("out_path", help='Output pdf path')
    parser.add_argument("--tmp_path", help='Temporary path for thumbnails')
    
    args = parser.parse_args()

    with open(args.in_path, 'r') as f:
        extract_pdf(json.load(f), args.out_path)