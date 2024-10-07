import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Load YOLOv3 network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set parameters
input_size = 416
confidence_threshold = 0.5
nms_threshold = 0.4
pseudo_label_threshold = 0.7  # Confidence threshold for generating pseudo-labels

pseudo_label_dir = "pseudo_labels"
os.makedirs(pseudo_label_dir, exist_ok=True)

def draw_bounding_boxes(img, boxes, confidences, class_ids):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def generate_pseudo_label(image, boxes, confidences, class_ids, save_dir, filename):
    with open(os.path.join(save_dir, filename.replace('.jpg', '.txt')), 'w') as f:
        for i in range(len(boxes)):
            if confidences[i] >= pseudo_label_threshold:
                x, y, w, h = boxes[i]
                label = f"{class_ids[i]} {(x + w / 2) / input_size} {(y + h / 2) / input_size} {w / input_size} {h / input_size}\n"
                f.write(label)
        cv2.imwrite(os.path.join(save_dir, filename), image)

def process_frame(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()

    final_boxes = [boxes[i] for i in indices]
    final_confidences = [confidences[i] for i in indices]
    final_class_ids = [class_ids[i] for i in indices]

    return final_boxes, final_confidences, final_class_ids

def run_real_time_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not found or could not be opened.")
        return
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, confidences, class_ids = process_frame(frame)
        draw_bounding_boxes(frame, boxes, confidences, class_ids)
        generate_pseudo_label(frame, boxes, confidences, class_ids, pseudo_label_dir, f"frame_{frame_id}.jpg")
        cv2.imshow('YOLOv3 Real-Time Detection', frame)
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Video file could not be opened.")
        return
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, confidences, class_ids = process_frame(frame)
        draw_bounding_boxes(frame, boxes, confidences, class_ids)
        generate_pseudo_label(frame, boxes, confidences, class_ids, pseudo_label_dir, f"frame_{frame_id}.jpg")
        cv2.imshow('YOLOv3 Video Detection', frame)
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def select_option():
    root = tk.Tk()
    root.title("YOLOv3 Detection Mode")
    
    def on_real_time():
        root.destroy()
        run_real_time_detection()
    
    def on_video_upload():
        root.destroy()
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
        if video_path:
            run_video_detection(video_path)
        else:
            messagebox.showerror("Error", "No file selected!")

    label = tk.Label(root, text="Choose Detection Mode:", font=('Helvetica', 14))
    label.pack(pady=20)

    real_time_button = tk.Button(root, text="Real-Time Detection", command=on_real_time, font=('Helvetica', 12), width=20)
    real_time_button.pack(pady=10)

    upload_button = tk.Button(root, text="Upload Video", command=on_video_upload, font=('Helvetica', 12), width=20)
    upload_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    select_option()
