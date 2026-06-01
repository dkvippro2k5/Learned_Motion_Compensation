"""
prepare_real_dataset.py — Build a real-world training/test set from raw videos.

Unlike extract_custom_videos.py (which only grabs 7 frames/video in Vimeo
layout), this extracts many consecutive frame PAIRS (t, t+1) from each clip
into the ref/cur layout consumed by FramePairDataset / get_dataloaders.

  data/real/ref/00000.png   data/real/cur/00000.png   <- (frame_t, frame_{t+1})
  data/real/ref/00001.png   data/real/cur/00001.png
  ...

One video is held out entirely for testing (real-world R-D evaluation), so the
test clip is never seen during training.

Usage:
    python prepare_real_dataset.py
    python prepare_real_dataset.py --height 256 --width 448 --stride 2 --max_pairs 2500
"""

import os, glob, argparse
import cv2


def extract_pairs(video_path, ref_dir, cur_dir, start_idx, height, width,
                  stride=1, max_from_clip=None):
    """Write consecutive (t, t+stride) pairs from one video. Returns next index."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(f, (width, height)))
    cap.release()

    idx = start_idx
    written = 0
    for t in range(0, len(frames) - stride, stride):
        cv2.imwrite(os.path.join(ref_dir, f"{idx:05d}.png"), frames[t])
        cv2.imwrite(os.path.join(cur_dir, f"{idx:05d}.png"), frames[t + stride])
        idx += 1
        written += 1
        if max_from_clip and written >= max_from_clip:
            break
    print(f"  {os.path.basename(video_path):45s} -> {written:4d} pairs")
    return idx


def build_split(videos, out_root, height, width, stride, max_pairs):
    ref_dir = os.path.join(out_root, 'ref')
    cur_dir = os.path.join(out_root, 'cur')
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(cur_dir, exist_ok=True)

    per_clip = None
    if max_pairs and len(videos):
        per_clip = max(1, max_pairs // len(videos))

    idx = 0
    for v in videos:
        idx = extract_pairs(v, ref_dir, cur_dir, idx, height, width,
                            stride=stride, max_from_clip=per_clip)
    print(f"==> {out_root}: {idx} total pairs ({height}x{width}, stride={stride})\n")
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir',   type=str, default='data/raw_videos')
    p.add_argument('--train_out', type=str, default='data/real')
    p.add_argument('--test_out',  type=str, default='data/real_test')
    p.add_argument('--height',    type=int, default=256)
    p.add_argument('--width',     type=int, default=448)
    p.add_argument('--stride',    type=int, default=1,
                   help='Frame gap for a pair. stride=2 -> larger motion, fewer pairs.')
    p.add_argument('--max_pairs', type=int, default=2500,
                   help='Cap on total training pairs (0 = no cap).')
    p.add_argument('--test_clip', type=str, default=None,
                   help='Filename of the held-out test video (default: last one).')
    args = p.parse_args()

    videos = sorted(glob.glob(os.path.join(args.raw_dir, '*.mp4')))
    if not videos:
        raise FileNotFoundError(f"No .mp4 in {args.raw_dir}")

    if args.test_clip:
        test_v = [v for v in videos if os.path.basename(v) == args.test_clip]
        if not test_v:
            raise FileNotFoundError(f"{args.test_clip} not in {args.raw_dir}")
        test_video = test_v[0]
    else:
        test_video = videos[-1]

    train_videos = [v for v in videos if v != test_video]

    print(f"Train clips: {len(train_videos)} | Test clip: {os.path.basename(test_video)}\n")
    print("Building TRAIN set:")
    build_split(train_videos, args.train_out, args.height, args.width,
                args.stride, args.max_pairs)
    print("Building TEST set:")
    build_split([test_video], args.test_out, args.height, args.width,
                args.stride, max_pairs=0)


if __name__ == '__main__':
    main()
