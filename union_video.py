import cv2
from pathlib import Path


def union():
    fps = 45
    video_name = 'output.mp4'
    images = sorted(p for p in Path('./video_frames').glob('*.jpeg'))
    frame = cv2.imread(images[0])
    h, w, l = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for p in images:
        video.write(cv2.imread(p))
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    union()