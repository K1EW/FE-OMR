import os
import cv2
import numpy as np
import pickle
import sys
import cvzone

def render(marks, preview, section, img):
    height, width = img.shape[:2]
    scale = 0.8 if section == "top" else 0.5
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    height, width = img.shape[:2]
    for mark in marks:
        (x, y), rect_size = mark
        cv2.rectangle(
            img,
            (
                int(x * width),
                int(y * height)
            ),
            (
                int(x * width + rect_size * height),
                int(y * height + rect_size * height)
            ),
            (0, 255, 0),
            1
        )
    preview_rect_size = 0.07 if section == "top" else 0.023
    cv2.rectangle(
        img,
        (
            int(preview[0] * width),
            int(preview[1] * height)
        ),
        (
            int(preview[0] * width + preview_rect_size * height),
            int(preview[1] * height + preview_rect_size * height)
        ),
        (0, 0, 255),
        1
    )
    cvzone.putTextRect(img, f"Number of marks: {len(marks)}",(10, 30), 1, 1, (0, 255, 0), 2)
    return img

def mouse_event(event, x, y, flags, param):
    marks, img, section, preview = param
    height, width = img.shape[:2]
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_size = 0.07 if section == "top" else 0.023
        marks.append(((x / width, y / height), rect_size))
    elif event == cv2.EVENT_MOUSEMOVE:
        preview[0], preview[1] = x / width, y / height

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python label.py <venue> <subject>")
        sys.exit(1)
    venue = sys.argv[1]
    subject = sys.argv[2]
    if venue not in ["FEST", "FE"]:
        print("Venue must be either FEST or FE")
        sys.exit(1)
    if venue == "FEST" and subject not in ["alevel", "tpat3"]:
        print("Subject must be either math, physics or tpat3")
        sys.exit(1)
    if venue == "FE" and subject not in ["pretest", "posttest"]:
        print("Subject must be either pretest or posttest")
        sys.exit(1)

    sections = {
        "top": cv2.imread(f"./label_templates/{venue}_top.jpg"),
        "answer": cv2.imread(f"./label_templates/{venue}_{subject}_answer.jpg"),
    }
    for section in sections:
        if sections[section] is None:
            print(f"Could not read {section} section")
            sys.exit(1)
        marks = []
        pickle_path = f"./pickles/{venue}_{section}.pkl"
        if section == "answer":
            pickle_path = f"./pickles/{venue}_{subject}_{section}.pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                marks = pickle.load(f)
        preview = [0, 0]
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("u") and len(marks) > 0:
                marks.pop()
            img = render(marks, preview, section, sections[section].copy())
            cv2.imshow("Label", img)
            cv2.setMouseCallback("Label", mouse_event, (marks, img, section, preview))
        with open(pickle_path, "wb") as f:
            pickle.dump(marks, f)
