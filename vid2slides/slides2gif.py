import argparse
import glob
import json
import moviepy.editor as mpy

def extract_gif(sequence, output_file, max_size):
    sources = []
    for _, el in enumerate(sequence['sequence']):
        if el['type'] == 'slide' and el['title']:
            sources.append(el['source'])

    fps = 1
    clip = mpy.ImageSequenceClip(sources, fps=fps)
    clip = clip.fx(mpy.vfx.crop, 
                   x1=sequence['crop'][0],
                   y1=sequence['crop'][1],
                   width=sequence['crop'][2],
                   height=sequence['crop'][3])
    clip = clip.fx(mpy.vfx.resize, width=max_size[0])
    clip.write_gif(output_file, fps=fps)

if __name__ == "__main__":
    desc = """Create gif from sequence of slides in json. Use extract_keyslides 
to create json first"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("in_path", help='Input json path')
    parser.add_argument("out_path", help='Output gif path')
    parser.add_argument("--tmp_path", help='Temporary path for thumbnails')
    
    args = parser.parse_args()

    with open(args.in_path, 'r') as f:
        extract_gif(json.load(f), args.out_path, (320, 180))