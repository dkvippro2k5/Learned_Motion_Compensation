import os
import cv2
import glob

raw_dir = "data/raw_videos"
out_dir = "data/vimeo90k_subset/sequences/00001"
os.makedirs(out_dir, exist_ok=True)

video_files = sorted(glob.glob(os.path.join(raw_dir, "*.mp4")))
valid_seqs = []

for idx, video_path in enumerate(video_files):
    seq_id = f"{idx + 1:04d}"
    seq_dir = os.path.join(out_dir, seq_id)
    os.makedirs(seq_dir, exist_ok=True)
    
    print(f"Extracting 7 frames from {os.path.basename(video_path)} to {seq_dir}...")
    
    cap = cv2.VideoCapture(video_path)
    # Start extracting slightly into the video to avoid black screens
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    
    success = True
    for i in range(1, 8):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read from {video_path}")
                success = False
                break
                
        # Giữ nguyên độ phân giải native (960x540) để RandomCrop xử lý
        cv2.imwrite(os.path.join(seq_dir, f"im{i}.png"), frame)
    
    cap.release()
    if success:
        valid_seqs.append(f"00001/{seq_id}")

# Tạo danh sách train/test
with open("data/vimeo90k_subset/sep_trainlist.txt", "w") as f:
    for seq in valid_seqs[:-1]: # Giữ lại 1 video cuối để test/validation
        f.write(f"{seq}\n")

with open("data/vimeo90k_subset/sep_testlist.txt", "w") as f:
    f.write(f"{valid_seqs[-1]}\n")

print(f"Hoan thanh! Da tao xong dataset tu {len(valid_seqs)} video cua ban (Chuan 7 frames/video).")
