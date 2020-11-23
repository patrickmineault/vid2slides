import argparse
import glob
import json
import moviepy.editor as mpy
import os
import tempfile

def extract_pdf(sequence, output_file):
    sources = []
    for _, el in enumerate(sequence['sequence']):
        if el['type'] == 'slide' and el['title']:
            sources.append(el['source'])

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('\n'.join(sources))

    cmd = f'tesseract "{f.name}" "{output_file}" pdf'
    print(cmd)
    os.system(cmd)


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