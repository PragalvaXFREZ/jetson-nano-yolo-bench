"""
Everything around the model that is NOT the model.

The network itself is a black box: a float tensor goes in, a float tensor
comes out. This file is the two translations on either side of it:

    image  --preprocess-->  [1, 3, S, S]  --MODEL-->  [1, 84, N]  --postprocess-->  boxes

Pure numpy + OpenCV, so the same file runs on the laptop (onnxruntime) and on
the Nano (TensorRT). It has to stay compatible with the Nano's Python 3.6 and
numpy 1.13, which is why there is nothing fancy in here.

Reading tip: every line that changes a tensor has its shape in a trailing
comment. If you get lost, read only the comments top to bottom.
    S = model input size (416)      N = number of candidate boxes (3549)
    K = boxes that survive a filter
"""
import numpy as np
import cv2

# The 80 things the pretrained model knows. The index in this list IS the
# class id the model outputs.
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


# ---------------------------------------------------------------------------
# 1. PREPROCESS: camera frame -> model input
# ---------------------------------------------------------------------------
def preprocess(img_bgr, size):
    """
    img_bgr: [H, W, 3] uint8, as OpenCV gives it (blue, green, red order).
    returns: tensor [1, 3, size, size] float32, plus (scale, pad_x, pad_y)
             which postprocess needs to map boxes back onto the original image.

    The model wants a SQUARE image, but cameras are 4:3 or 16:9. Stretching
    would distort objects, so we "letterbox": shrink the image keeping its
    proportions, then fill the leftover strip with grey.
    """
    h, w = img_bgr.shape[:2]
    scale = min(size / float(h), size / float(w))            # one factor for both axes
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # [new_h, new_w, 3]

    # Centre it on a grey canvas. 114 is the grey the model saw during training.
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)                  # [S, S, 3]
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    x = canvas[:, :, ::-1]                    # [S, S, 3]  BGR -> RGB (reverse the last axis)
    x = x.astype(np.float32) / 255.0          # [S, S, 3]  0..255 -> 0..1
    x = x.transpose(2, 0, 1)                  # [3, S, S]  HWC -> CHW: channels first
    x = x[np.newaxis]                         # [1, 3, S, S] add the batch axis
    # transpose only changes how numpy *walks* the memory. The GPU needs the
    # bytes physically in the new order, so force a real copy:
    x = np.ascontiguousarray(x)
    return x, (scale, pad_x, pad_y)


# ---------------------------------------------------------------------------
# 2. IoU: how much do two boxes overlap?  (the heart of NMS)
# ---------------------------------------------------------------------------
def iou_one_to_many(box, boxes):
    """
    box:   [4]     one box  (x1, y1, x2, y2)
    boxes: [K, 4]  many boxes
    returns [K]    IoU of `box` against each of them, 0 = apart, 1 = identical.

    IoU = area of overlap / area of union.

    Broadcasting is doing the loop for us: `box[0]` is a single number and
    `boxes[:, 0]` is K numbers, so np.maximum compares the one against all K
    at once. No for-loop, same result.
    """
    ix1 = np.maximum(box[0], boxes[:, 0])     # [K] left edge of the overlap
    iy1 = np.maximum(box[1], boxes[:, 1])     # [K] top
    ix2 = np.minimum(box[2], boxes[:, 2])     # [K] right
    iy2 = np.minimum(box[3], boxes[:, 3])     # [K] bottom

    # If the boxes do not overlap, right < left and the width goes negative.
    # Clip at 0 so "no overlap" becomes area 0 instead of a nonsense number.
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)      # [K]

    area_a = (box[2] - box[0]) * (box[3] - box[1])                         # scalar
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])     # [K]
    return inter / (area_a + area_b - inter + 1e-9)                        # [K]


# ---------------------------------------------------------------------------
# 3. NMS: many overlapping guesses -> one box per object
# ---------------------------------------------------------------------------
def nms(boxes, scores, iou_thres):
    """
    boxes [K, 4], scores [K]  ->  list of indices to keep.

    The model fires on the same bus from dozens of neighbouring grid cells.
    Greedy rule: take the most confident box, delete everything that overlaps
    it too much (same object), repeat with what is left.
    """
    order = np.argsort(-scores)               # [K] indices, best score first
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]                                   # [K-1]
        overlap = iou_one_to_many(boxes[best], boxes[rest])  # [K-1]
        order = rest[overlap <= iou_thres]                 # boolean mask: drop the duplicates
    return keep


# ---------------------------------------------------------------------------
# 4. POSTPROCESS: raw model output -> final detections
# ---------------------------------------------------------------------------
def postprocess(output, meta, conf_thres=0.25, iou_thres=0.7, max_det=300):
    """
    output: [1, 84, N] raw model output
    meta:   (scale, pad_x, pad_y) from preprocess
    returns boxes [M, 4] in ORIGINAL image pixels (x1, y1, x2, y2),
            scores [M], class_ids [M]

    What the 84 rows are, for each of the N candidates:
        rows 0..3   cx, cy, w, h   box centre and size, in model-input pixels
        rows 4..83  one score per COCO class, already between 0 and 1
    (Older YOLOs had an extra "objectness" row and needed anchors. v8 does not:
    the decoding is already inside the exported graph.)
    """
    scale, pad_x, pad_y = meta

    pred = output[0].T                        # [84, N] -> [N, 84]  one ROW per candidate now
    xywh = pred[:, :4]                        # [N, 4]
    cls_scores = pred[:, 4:]                  # [N, 80]

    # For each candidate: which class does it believe in most, and how much?
    # axis=1 means "collapse the 80 class columns", leaving one value per row.
    class_ids = cls_scores.argmax(axis=1)     # [N]  index of the best class
    scores = cls_scores.max(axis=1)           # [N]  its score

    # Throw away the ~3500 candidates that are not confident about anything.
    mask = scores > conf_thres                # [N] of True/False
    xywh, scores, class_ids = xywh[mask], scores[mask], class_ids[mask]    # [K, ...]
    if scores.size == 0:
        return np.zeros((0, 4), np.float32), scores, class_ids

    # centre+size -> two corners, which is what IoU and drawing want.
    half = xywh[:, 2:4] / 2.0                                              # [K, 2]
    boxes = np.concatenate([xywh[:, 0:2] - half, xywh[:, 0:2] + half], 1)  # [K, 4] x1,y1,x2,y2

    # Class-aware NMS with one trick. A person standing in front of a bus
    # overlaps the bus heavily, and plain NMS would delete one of them.
    # Shift every box by (class_id * big_number): boxes of different classes
    # now live far apart and can never overlap, so one NMS call handles all
    # 80 classes separately. The shift is only used for NMS, not for output.
    offset = class_ids[:, np.newaxis].astype(np.float32) * 7680.0          # [K, 1] broadcasts over the 4 coords
    keep = nms(boxes + offset, scores, iou_thres)[:max_det]
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]  # [M, ...]

    # Undo the letterbox: remove the grey padding, then undo the shrink.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    return boxes, scores, class_ids


# ---------------------------------------------------------------------------
# 5. DRAW
# ---------------------------------------------------------------------------
def draw(img_bgr, boxes, scores, class_ids):
    """Draws in place on img_bgr and returns it."""
    h, w = img_bgr.shape[:2]
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, class_ids):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        # A fixed pseudo-random colour per class, so "person" is always the same colour.
        color = (int(37 * c) % 256, int(17 * c + 80) % 256, int(29 * c + 160) % 256)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = "%s %.2f" % (COCO[int(c)], s)
        cv2.putText(img_bgr, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_bgr
