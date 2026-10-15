import os, csv
import numpy as np
import cv2


IMG_H, IMG_W = 66, 200
INPUT_SHAPE = (IMG_H, IMG_W, 3)

# offset applied to left/right camera steering to simulate recovery
STEERING_OFFSET = 0.2


def load_log(data_dir):
    """Parse driving_log.csv — returns parallel arrays of image paths and steering angles.
    Uses all 3 cameras (centre, left, right) with a steering correction on left/right.
    Handles both header and no-header CSV formats."""
    imgs = []
    angles = []
    img_dir = os.path.join(data_dir, 'IMG')

    csv_path = os.path.join(data_dir, 'driving_log.csv')
    with open(csv_path) as f:
        reader = csv.reader(f)
        first_row = next(reader)
        # check if first row is a header
        try:
            float(first_row[3])
            rows = [first_row]
        except ValueError:
            rows = []

        for row in reader:
            rows.append(row)

        for row in rows:
            if len(row) < 4:
                continue
            angle = float(row[3])
            centre = row[0].strip()
            left   = row[1].strip()
            right  = row[2].strip()

            # handle both full paths and just filenames
            if not os.path.isabs(centre) and not os.path.exists(os.path.join(data_dir, centre)):
                centre = os.path.join('IMG', centre)
                left   = os.path.join('IMG', left)
                right  = os.path.join('IMG', right)

            imgs   += [os.path.join(data_dir, centre),
                       os.path.join(data_dir, left),
                       os.path.join(data_dir, right)]
            angles += [angle, angle + STEERING_OFFSET, angle - STEERING_OFFSET]

    return np.array(imgs), np.array(angles)


def preprocess(img):
    """Crop the sky and hood, resize to 66x200, convert to YUV (what nvidia used)."""
    img = img[60:135, :, :]
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'image not found: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess(img)


def random_flip(img, angle):
    if np.random.random() < 0.5:
        img = cv2.flip(img, 1)
        angle = -angle
    return img, angle


def random_brightness(img):
    # go YUV -> RGB -> HSV, tweak V channel, then back
    tmp = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV).astype(np.float32)
    tmp[:, :, 2] *= np.random.uniform(0.4, 1.2)
    tmp = np.clip(tmp, 0, 255).astype(np.uint8)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2RGB)
    return cv2.cvtColor(tmp, cv2.COLOR_RGB2YUV)


def augment(img, angle):
    img, angle = random_flip(img, angle)
    img = random_brightness(img)
    return img, angle


def generator(paths, angles, batch_size=64, training=True):
    """Infinite generator that yields (batch_images, batch_angles)."""
    n = len(paths)
    while True:
        idx = np.random.permutation(n) if training else np.arange(n)
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            X, y = [], []
            for i in batch:
                img = load_img(paths[i])
                a   = angles[i]
                if training:
                    img, a = augment(img, a)
                X.append(img)
                y.append(a)
            yield np.array(X), np.array(y)
