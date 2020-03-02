import numpy as np
import cv2


#data aug like resize (0.5 to 2), translation/crop,hflip,color channels noise/brightness
def random_crop(img, lbl):
    cols, rows = img.shape[:2]
    a = np.random.randint(0, rows // 2)
    b = np.random.randint(0, cols // 2)
    new_img = cv2.resize(img[b:b + (cols // 2), a:a + (rows // 2)], (rows, cols))
    new_lbl = cv2.resize(lbl[b:b + (cols // 2), a:a + (rows // 2)], (rows, cols))
    return new_img, new_lbl


def h_flip(img, lbl):
    h_img = img[:, ::-1, :]  # vflip is img[::-1,:,:]
    l_img = lbl[:, ::-1, :]
    return h_img, l_img


def translation(img, lbl):
    cols, rows = img.shape[:2]
    a = np.random.randint(0, cols // 4)
    b = np.random.randint(0, rows // 4)
    if np.random.random() < 0.5:
        t_mat = np.float32([[1, 0, -b], [0, 1, -a]])
    else:
        t_mat = np.float32([[1, 0, b], [0, 1, a]])
    img_trans = cv2.warpAffine(img, t_mat, (rows, cols))
    lbl_trans = cv2.warpAffine(lbl, t_mat, (rows, cols))
    return img_trans, lbl_trans


def brightness(img, lbl):
    factor = np.random.choice([-0.5, -0.2, 0.2, 0.5])
    img = img + (factor * img).astype(img.dtype)
    return img, lbl


def channel_noise(img, lbl):
    a, b, c = img[..., 0], img[..., 1], img[..., 2]
    noise = np.random.normal(0, 0.5, size=a.shape)
    value = np.random.random()
    if value < 0.3:
        a += (a * noise).astype(a.dtype)
    elif 0.3 < value < 0.6:
        b += (b * noise).astype(b.dtype)
    else:
        c += (c * noise).astype(c.dtype)
    new_img = np.array([a, b, c]).transpose((1, 2, 0))
    return new_img, lbl


def augmentation(argument, a, b):
    switcher = {
        1: random_crop,
        2: h_flip,
        3: translation,
        4: brightness,
        5: channel_noise,
    }
    func = switcher.get(argument, 'Invalid')
    return func(a, b)