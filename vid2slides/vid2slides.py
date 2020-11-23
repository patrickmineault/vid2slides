import argparse
import collections
import cv2
from decord import VideoReader
from decord import cpu
import glob
import ffmpeg
import json
from matplotlib import image
import numpy as np
import os
import pickle
import pytesseract
from pytesseract import Output
import sklearn
import sklearn.cluster
import tempfile


def get_video_info(path):
    probe = ffmpeg.probe(path)

    for stream in probe["streams"]:
        if stream["codec_type"] == "video":
            return stream


def to_timestamp(ts):
    h, m, s = ts // 3600, (ts // 60) % 60, int(ts % 60)
    return f'{h:02}:{m:02}:{s:02}'


def log_viterbi(log_B, log_A, log_pi):
    """Computes the Viterbi path.

    Note that this function uses numerically stable logs of probability.

    Arguments:
    log_B: A matrix of size T X N, where N is the number of states.
        It's the log probability of the observation at time T given that the
        system was in state N.
    log_A: The state-transition probability matrix, size NxN
    log_pi: The initial state distribution vector pi = {pi_1....pi_N}
    """
    delta = log_pi + log_B[0, :]
    phi = np.zeros(log_B.shape, dtype=np.uint16)

    for t in range(1, log_B.shape[0]):
        projected = delta.reshape((-1, 1)) + log_A
        delta = np.max(projected, axis=0) + log_B[t, :]
        phi[t, :] = np.argmax(projected, axis=0)

    q = np.zeros(log_B.shape[0], dtype=np.int)
    q[-1] = np.argmax(delta)
    for t in range(log_B.shape[0] - 2, -1, -1):
        q[t] = phi[t + 1, q[t + 1]]

    return q


def heuristic_frames(sizes, ban_time=5):
    """
    Get some heuristically chosen reference frames. 
    
    Pick the images which are least compressible, and ban surrounding images.
    """
    tuples = [(sz, x) for x, sz in enumerate(sizes)]
    banned = {}

    chosen = []
    for _, x in sorted(tuples)[::-1]:
        if x in banned:
            continue

        for d in range(-ban_time, ban_time + 1):
            banned[x + d] = 1

        chosen.append(x)

    return chosen


def extract_thumbnails(video, 
                       lo_dir,                        
                       lo_size=(320, 180), 
                       thumb_interval=2):
    """
    Extract thumbnails from video and output them to output dir.

    Arguments:
        video: the video path
        lo_dir: the output directory for low-res thumbnails. Thumbnails are put 
            in this directory named thumb-%02d.jpg
        output_dir: the output directory for hi-res thumbnails. Thumbnails are 
            put in this directory named thumb-%02d.png
        lo_size: the max size of the low-res thumbnails. We will resize the thumbnails
            to fit within this bounding box.
        hi_size: max size of the high-res thumbails
        thumb_interval (optional): the time in seconds between thumbnails
    """
    info = get_video_info(video)
    w, h = info['coded_width'], info['coded_height']

    aspect_ratio = w / h
    if aspect_ratio > lo_size[0] / lo_size[1]:
        # Wide format
        wo, ho = lo_size[0], int(lo_size[0] // aspect_ratio)
    else:
        wo, ho = int(lo_size[1] * aspect_ratio), lo_size[1]

    os.system(f'ffmpeg -i {video} -s {wo}x{ho} -r 1/{thumb_interval}'
              f' -f image2 {lo_dir}/thumb-%04d.jpg')


def extract_frames(video, hi_dir, hi_size, times):
    info = get_video_info(video)
    w, h = info['coded_width'], info['coded_height']

    aspect_ratio = w / h
    if aspect_ratio > hi_size[0] / hi_size[1]:
        # Wide format
        wo, ho = hi_size[0], int(hi_size[0] // aspect_ratio)
    else:
        wo, ho = int(hi_size[1] * aspect_ratio), hi_size[1]

    framerate = int(info['nb_frames']) / float(info['duration'])
    
    nframes = []
    for time in times:
        nframes.append(int(framerate * (2 * (time + 1))))

    vr = VideoReader(video, ctx=cpu(0))
    frames = vr.get_batch(nframes).asnumpy()
    
    for i in range(len(nframes)):
        frame = frames[i, :, :, :]
        # Now clear why r and b are mixed up.
        frame = frame[:, :, np.array([2, 1, 0])]
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
        
        cv2.imwrite(
            os.path.join(hi_dir, f'thumb-{times[i]+1:04}.png'),
            cv2.resize(frame, (wo, ho))
        )


def detect_faces(the_dir):
    """
    Read faces in a directory and report on them.
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detections = []
    for im_name in sorted(glob.glob(os.path.join(the_dir, 'thumb*.jpg'))):
        im = cv2.imread(im_name)
        height = im.shape[0]
        detections.append(face_cascade.detectMultiScale(im, 1.1, 4))

    # Classify detections into full-screen faces or not.
    full_face = []
    small_faces = []

    for frame in detections:
        is_full = False
        for l, t, w, h in frame:
            # If it fills up more than 25% of the screen, it's a likely 
            # full-screen face.
            if w > height * .25:
                is_full = True
            else:
                small_faces.append((l, t, w, h))

        full_face.append(is_full)

    # Cluster small faces to find the approximate location of pip.
    if len(small_faces) > 8:
        kmeans = sklearn.cluster.KMeans()
        classes = kmeans.fit_predict(np.array(small_faces))
        biggest_class = kmeans.cluster_centers_[np.bincount(classes).argmax()]
    else:
        biggest_class = []

    return {'has_full_face': np.array(full_face),
            'pip_location': biggest_class}


def get_delta_images(the_dir, has_face):
    matching_images = glob.glob(os.path.join(the_dir, 'thumb-*.jpg'))
    nimages = len(matching_images)
    sizes = []

    for i, filename in enumerate(matching_images):
        if i == 0:
            img = image.imread(filename)
            images = np.zeros((nimages, img.shape[0], img.shape[1]))

        sizes.append(os.stat(filename).st_size)
        img = image.imread(filename)
        images[i, :, :] = img.mean(axis=2)

        if has_face[i]:
            # Remove faces out of the pool of potential matches.
            sizes[i] = 0

    candidates = sorted(heuristic_frames(sizes))

    candidate_images = np.stack([images[i, :, :] for i in candidates])
    assert candidate_images.shape[0] == len(candidates)
    assert candidate_images.ndim == 3

    delta_images = np.zeros((nimages, len(candidates)))
    for i in range(len(candidates)):
        delta_images[:, i] = ((images.reshape((images.shape[0], -1)) - 
            candidate_images[i, :, :].reshape((1, -1))) ** 2).sum(axis=1)
    
    return sizes, candidates, delta_images


def max_likelihood_sequence(nll, jump_probability=0.2):
    """
    Calculate the maximum likelihood sequence of images.

    Takes a sequence of images and attributes these images to a sequence of 
    template images. Uses a left-to-right HMM to attribute find the ML sequence
    of images.

    Arguments:
        nll: an (ntotal, ntemplate) np.array, where the (i, j)'th element 
        contains the negative log-likelihood of the j'th image as an instance of
        the i'th template. Under a Gaussian noise generative model, this would 
        be the sum-of-squares between template and instance.
        jump_probability: the probability that the sequence jumps from one 
        template to a further in the sequence.

    Returns:
        The maximum likelihood sequence of templates.
    """
    assert nll.shape[0] >= nll.shape[1]
    ncandidates = nll.shape[1]

    log_B = -nll / nll.mean()

    T = (np.arange(ncandidates).reshape((-1, 1)) < 
         np.arange(ncandidates).reshape((1, -1)))
    A = (1 - jump_probability) * np.eye(ncandidates) + jump_probability * T / T.sum(axis=1, keepdims=True)
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    A = A / A.sum(axis=1, keepdims=True)
    
    log_pi = np.log(np.ones(ncandidates) / ncandidates)
    
    seq = log_viterbi(log_B, np.log(A), log_pi)
    return seq


def extract_crop(info):
    """
    Find a reasonable crop given the information available.
    """
    ims = []
    for el in info['sequence']:
        if el['type'] == 'slide':
            im = cv2.imread(el['source'])
            ims.append(im.mean(axis=2))

    A = np.stack(ims, axis=0)
    broad_crop = (A.mean(axis=0) > .2).astype(np.uint8)

    contours, _ = cv2.findContours(broad_crop, cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE) 

    # Find the largest contour
    biggest_ar = 0
    for _, contour in enumerate(contours):
        ar = cv2.contourArea(contour)
        if ar > biggest_ar:
            biggest_ar = ar
            (x, y, w, h) = cv2.boundingRect(contour)

    return (x, y, w, h)


def get_slide_title(detection):
    texts = collections.defaultdict(list)
    
    block_heights = []
    block_tops = []
    for i, text in enumerate(detection['text']):
        block_num = detection['block_num'][i]
        if int(detection['conf'][i]) > 80 and text.strip():
            texts[block_num].append(text)

        if detection['level'][i] == 2:
            block_heights.append(detection['height'][i])
            block_tops.append(detection['top'][i])

    blocks = [' '.join(texts[i]) for i in range(1, max(detection['block_num'])+1)]
    blocks = [(t, h, x) for x, h, t in zip(blocks, block_heights, block_tops) if x]
    blocks = sorted(blocks)

    # Pick the first piece of text from the top that is more than 30 pixels
    # high. 
    for block_num, block in enumerate(blocks):
        if block[0] > 30 and len(block[2]) >= 5:
            break

    if blocks:
        return blocks[block_num][2]
    else:
        return ""


def extract_keyframes_from_video(target, output_json, thumb_dir):
    lo_size = (360, 202)
    hi_size = (1920, 1080)

    if thumb_dir is None:
        thumb_dir = tempfile.mkdtemp()

    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)

    lo_dir = os.path.join(thumb_dir, 'lo')
    if not os.path.exists(lo_dir):
        os.makedirs(lo_dir)

    hi_dir = os.path.join(thumb_dir, 'hi')
    if not os.path.exists(hi_dir):
        os.makedirs(hi_dir)

    extract_thumbnails(target, lo_dir, lo_size)
    face_information = detect_faces(lo_dir)
    _, candidates, sse = get_delta_images(lo_dir, face_information['has_full_face'])

    to_select = np.where(~face_information['has_full_face'])[0]

    # Now reconstruct a slide deck based on the candidates.
    sequence = max_likelihood_sequence(sse[to_select, :])

    full_sequence = -np.ones(sse.shape[0])
    full_sequence[to_select] = sequence

    # And dump all the information in a JSON file.
    last_num = -2

    latest_slide = {'start_index': 0}
    slides = []
    
    for i, num in enumerate(full_sequence):
        if num != last_num:
            # Write things down
            latest_slide['end_time'] = to_timestamp(2 * (i + 1))
            latest_slide['end_index'] = i

            offset = candidates[int(last_num)]

            latest_slide['offset'] = offset

            latest_slide['source'] = os.path.join(hi_dir, 
                f'thumb-{offset+1:04}.png')
            slides.append(latest_slide)

            if num == -1:
                latest_slide = {
                    'type': 'speaker',
                    'start_time': to_timestamp(2 * (i + 1)),
                    'start_index': i,
                }
            else:
                latest_slide = {
                    'type': 'slide',
                    'start_time': to_timestamp(2 * (i + 1)),
                    'start_index': i
                }
            last_num = num

    latest_slide['end_time'] = to_timestamp(2 * (i + 1))
    latest_slide['end_index'] = i

    offset = candidates[int(last_num)]
    latest_slide['offset'] = offset
    latest_slide['source'] = os.path.join(hi_dir, 
        f'thumb-{offset+1:04}.png')

    slides.append(latest_slide)
    slides = slides[1:]

    offsets = []
    for slide in slides:
        if slide['type'] == 'slide':
            offsets.append(slide['offset'])

    extract_frames(target, hi_dir, hi_size, offsets)
    
    info = {'pip_location': face_information['pip_location'],
            'sequence': slides}

    info['crop'] = extract_crop(info)

    print(f"Found {len(slides)} canonical slides")

    for el in info['sequence']:
        if el['type'] == 'slide':
            im = cv2.imread(el['source'])
            d = pytesseract.image_to_data(im, output_type=Output.DICT)
            el['title'] = get_slide_title(d)

    with open(output_json, 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    desc = """Extract key slides from video.
    
Extract key slides from video presenting a slide deck. The video could be a 
recording from Zoom or Google Meet, for example. The script extracts thumbnails
from the video and outputs a JSON file with timings for key slides. Use the JSON
file with the make_gif and make_pdf utils to extract usable slides from
a video.
"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("in_path", help='Input video path')
    parser.add_argument("out_path", help='Output json path')
    parser.add_argument("--tmp_path", help='Temporary path for thumbnails')
    
    args = parser.parse_args()

    extract_keyframes_from_video(args.in_path, args.out_path, args.tmp_path)
