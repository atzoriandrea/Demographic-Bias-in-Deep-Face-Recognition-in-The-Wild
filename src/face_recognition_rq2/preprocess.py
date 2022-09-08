import gc
import multiprocessing
import os.path
import sys
import time
import cv2
from deepface import DeepFace


def get_files_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


def resume(imgs, cropped):
    import os
    basepath = "/".join(imgs[0][::-1].split("/")[2:])[::-1]
    relatives, images = [], []
    for crp in cropped:
        relatives.append("/".join(crp.split("/")[-2::]))
    for img in imgs:
        images.append("/".join(img.split("/")[-2::]))

    diff = list(set(images).difference(relatives))

    for i, d in enumerate(diff):
        diff[i] = os.path.join(basepath, d)
    return diff


def preprocess(images):
    for image in images:
        start_time = time.time()
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn']
        try:
            cropped_image = DeepFace.detectFace(image, detector_backend=backends[2], target_size=(112, 112))

            outfilepath = image.split("/")
            outfilepath[-3] = "cropped"
            sub_root_folder = "/".join(outfilepath[:-1])
            outfilepath = "/".join(outfilepath)

            if not os.path.exists(sub_root_folder):
                os.makedirs(sub_root_folder)
            cv2.imwrite(outfilepath, cv2.normalize(cropped_image[:, :, ::-1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
            gc.collect()
            print("Processing time: " + str(time.time() - start_time))
        except:
            print("Invalid Image: " + image)


if __name__ == '__main__':
    import argparse

    try:
        parser = argparse.ArgumentParser(description='Image aligner and cropper')
        parser.add_argument('--imgs', metavar='path', required=True,
                            help='path to images')
        parser.add_argument('--processes', required=True, default=6,
                            help='number of cpu processes')
        args = parser.parse_args()
        if args.imgs == "":
            print("Invalid image path")
    except Exception as e:
        print(e)
        sys.exit(1)
    images = get_files_full_path(args.imgs)
    outfilepath = images[0].split("/")
    outfilepath[-3] = "cropped"
    outfilepath = "/".join(outfilepath[:-2])
    done = get_files_full_path(outfilepath)
    if len(done) > 0:
        images = resume(images, done)
    num_processes = int(args.processes)
    chunk_length = int(len(images) / num_processes)
    image_chunks = []
    for rank in range(num_processes):
        if rank != num_processes - 1:
            image_chunks.append(images[rank * chunk_length:(rank + 1) * chunk_length])
        else:
            image_chunks.append(images[rank * chunk_length:])
    processes = []
    for i, c in enumerate(image_chunks):
        processes.append(
            multiprocessing.Process(target=preprocess, args=(c,)))

    for t in processes:
        t.start()
    for t in processes:
        t.join()
