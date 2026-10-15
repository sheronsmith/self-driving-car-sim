"""
Generate a practice track with synthetic driving data.

This creates fake camera images (solid color with lane lines drawn on them)
and a driving_log.csv so you can test the whole pipeline without needing
the simulator right away. The images are simple but they're enough to
make sure your model trains and the pipeline doesn't crash.

    python track.py --output data/ --samples 3000
"""

import argparse, os, csv, math
import numpy as np
import cv2


TRACK_WIDTH = 320
TRACK_HEIGHT = 160

# colors (BGR)
ROAD_COLOR   = (60, 60, 60)
GRASS_COLOR  = (50, 130, 50)
LANE_COLOR   = (200, 200, 200)
YELLOW_LINE  = (0, 200, 255)


def make_frame(steering, frame_idx):
    """Draw a simple top-down-ish road scene based on the steering angle."""
    img = np.full((TRACK_HEIGHT, TRACK_WIDTH, 3), GRASS_COLOR, dtype=np.uint8)

    # road is a trapezoid that shifts based on steering
    offset = int(steering * 80)
    top_left  = TRACK_WIDTH // 2 - 40 + offset
    top_right = TRACK_WIDTH // 2 + 40 + offset
    bot_left  = TRACK_WIDTH // 2 - 120
    bot_right = TRACK_WIDTH // 2 + 120

    road_pts = np.array([
        [bot_left, TRACK_HEIGHT],
        [top_left, 40],
        [top_right, 40],
        [bot_right, TRACK_HEIGHT]
    ])
    cv2.fillPoly(img, [road_pts], ROAD_COLOR)

    # center yellow dashed line
    center_top = TRACK_WIDTH // 2 + offset
    center_bot = TRACK_WIDTH // 2
    for y in range(40, TRACK_HEIGHT, 20):
        if (y // 20) % 2 == 0:
            frac_top = (y - 40) / (TRACK_HEIGHT - 40)
            frac_bot = (min(y + 10, TRACK_HEIGHT) - 40) / (TRACK_HEIGHT - 40)
            x1 = int(center_top + (center_bot - center_top) * frac_top)
            x2 = int(center_top + (center_bot - center_top) * frac_bot)
            cv2.line(img, (x1, y), (x2, min(y + 10, TRACK_HEIGHT)), YELLOW_LINE, 2)

    # white edge lines
    for side in ['left', 'right']:
        if side == 'left':
            x_top, x_bot = top_left, bot_left
        else:
            x_top, x_bot = top_right, bot_right
        cv2.line(img, (x_top, 40), (x_bot, TRACK_HEIGHT), LANE_COLOR, 2)

    # add some noise so it's not perfectly clean
    noise = np.random.randint(-5, 6, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_course(num_samples):
    """Generate a sequence of steering angles that simulate driving a course.
    Mix of straights, gentle curves, and sharp turns."""
    angles = []
    i = 0
    while i < num_samples:
        # pick a random segment type
        seg_type = np.random.choice(['straight', 'gentle_left', 'gentle_right',
                                      'sharp_left', 'sharp_right'],
                                     p=[0.3, 0.2, 0.2, 0.15, 0.15])
        seg_len = np.random.randint(20, 80)

        if seg_type == 'straight':
            seg_angles = np.random.normal(0, 0.02, seg_len)
        elif seg_type == 'gentle_left':
            seg_angles = np.random.normal(-0.15, 0.03, seg_len)
        elif seg_type == 'gentle_right':
            seg_angles = np.random.normal(0.15, 0.03, seg_len)
        elif seg_type == 'sharp_left':
            seg_angles = np.random.normal(-0.4, 0.05, seg_len)
        else:
            seg_angles = np.random.normal(0.4, 0.05, seg_len)

        angles.extend(seg_angles.tolist())
        i += seg_len

    # clip to [-1, 1] and trim to exact count
    angles = np.clip(angles[:num_samples], -1, 1)
    return angles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',  default='data', help='output directory')
    ap.add_argument('--samples', type=int, default=3000, help='number of frames')
    args = ap.parse_args()

    img_dir = os.path.join(args.output, 'IMG')
    os.makedirs(img_dir, exist_ok=True)

    print(f'generating {args.samples} frames ...')
    angles = generate_course(args.samples)

    rows = []
    for i, angle in enumerate(angles):
        # generate center, left, right camera views
        # left/right are just slightly offset versions
        center_img = make_frame(angle, i)
        left_img   = make_frame(angle + 0.08, i)   # shifted a bit
        right_img  = make_frame(angle - 0.08, i)

        center_name = f'IMG/center_{i:05d}.jpg'
        left_name   = f'IMG/left_{i:05d}.jpg'
        right_name  = f'IMG/right_{i:05d}.jpg'

        cv2.imwrite(os.path.join(args.output, center_name), center_img)
        cv2.imwrite(os.path.join(args.output, left_name), left_img)
        cv2.imwrite(os.path.join(args.output, right_name), right_img)

        # throttle, brake, speed are just filler values
        rows.append([center_name, left_name, right_name,
                     f'{angle:.6f}', '0.3', '0.0', '20.0'])

        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{args.samples}')

    csv_path = os.path.join(args.output, 'driving_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'done — {len(rows)} frames saved to {args.output}/')
    print(f'  images: {img_dir}/')
    print(f'  log:    {csv_path}')
    print()
    print('now you can train with:')
    print(f'  python train.py --data_dir {args.output} --epochs 10')


if __name__ == '__main__':
    main()
