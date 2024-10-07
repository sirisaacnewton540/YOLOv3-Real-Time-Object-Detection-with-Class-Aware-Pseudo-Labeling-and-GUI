# YOLOv3 Real-Time Object Detection with Class-Aware Pseudo-Labeling and GUI

## Overview

This repository provides an implementation of YOLOv3 (You Only Look Once, version 3) for real-time object detection, with a focus on addressing significant challenges like class imbalance and ease of use through a graphical user interface (GUI). The project incorporates a semi-supervised learning technique known as **Class-Aware Pseudo-Labeling** to mitigate the adverse effects of class imbalance, a common issue in object detection tasks, while also offering a user-friendly interface to choose between real-time detection via a webcam or processing a video file.

## Background and Motivation

### Object Detection with YOLOv3

YOLOv3 is a state-of-the-art object detection model known for its balance between speed and accuracy. Unlike traditional object detectors that require multiple passes over the image, YOLOv3 predicts bounding boxes and class probabilities in a single evaluation of the network. This makes it particularly well-suited for real-time applications.

However, while YOLOv3 is powerful, its performance can be significantly affected by various challenges inherent in object detection tasks, such as class imbalance, variations in object scale, occlusions, and differences in lighting conditions. Among these, class imbalance is one of the most pervasive issues, leading to biased models that perform well on frequent classes but poorly on rare ones.

### The Class Imbalance Problem

![YOLOv3 Video Detection 2024-08-12 05-20-06](https://github.com/user-attachments/assets/ddd90aff-e7a5-42f7-9f4e-959161ae91b7)


Class imbalance occurs when certain classes in a dataset are overrepresented compared to others. For instance, in a dataset like COCO, common objects like "person" and "car" are abundantly represented, while objects like "truck," "sink," "cat," and "dog" are less frequent. This imbalance can skew the learning process of a neural network, causing it to become biased towards majority classes, leading to poor generalization on minority classes.

Mathematically, class imbalance can be described by the probability distribution of classes:

#### $\[P(C_j) = \frac{|C_j|}{\sum_{k=1}^{m} |C_k|}\]$

Where:
- $\( |C_j| \)$ is the number of instances of class $\( C_j \)$.
- $\( m \)$ is the total number of classes.

When the distribution $\( P(C_j) \)$ is heavily skewed, the loss function tends to prioritize majority classes, which leads to underfitting on minority classes. This results in models that are less effective in real-world scenarios, particularly when the detection of rare objects is crucial.

### Addressing Class Imbalance with Class-Aware Pseudo-Labeling

To counter the effects of class imbalance, this project implements a technique known as **Class-Aware Pseudo-Labeling**. This approach generates additional training data (pseudo-labels) for underrepresented classes during the detection process, effectively "balancing" the dataset by augmenting it with more samples of minority classes. This helps in improving the model's generalization ability across all classes.

## Features

- **Real-Time Object Detection**: Perform object detection using your webcam, processing each frame in real-time.
- **Video File Detection**: Upload a video file and run object detection on each frame of the video.
- **Class-Aware Pseudo-Labeling**: Automatically generate pseudo-labels for high-confidence detections, particularly for underrepresented classes, to improve model training and mitigate class imbalance.
- **User-Friendly GUI**: A simple and intuitive graphical interface that allows users to choose between real-time detection and video file processing.

## Installation

### Prerequisites

- Python 3.6+
- OpenCV
- NumPy
- Tkinter (for the GUI)

### Steps to Install

 **Install Dependencies**:
   ```bash
   pip install opencv-python-headless==4.5.1.48 numpy tk
   ```

 **Download YOLOv3 Weights**:
   - Download the pre-trained YOLOv3 weights.
   - Place the `yolov3.weights` file in the project directory.

 **Download COCO Class Names**:
   - Download the `coco.names` file.
   - Place the `coco.names` file in the project directory.

## Usage

### Using the GUI

- **Real-Time Detection**: Click on the "Real-Time Detection" button to start detecting objects using your webcam. The program will process each frame captured by the webcam and display the detected objects in real-time.
- **Upload Video**: Click on the "Upload Video" button to select a video file from your system. The program will process each frame of the selected video file, applying object detection and displaying the results.

### Exiting the Detection

Press the `q` key at any time during detection to quit the application.

## Implementation Details

### YOLOv3 Setup

The YOLOv3 network is loaded using OpenCV’s DNN module, which enables efficient execution of the network on both CPUs and GPUs. The network configuration (`yolov3.cfg`) and pre-trained weights (`yolov3.weights`) are used to initialize the model.

```python
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```

### Real-Time and Video Detection

The functions `run_real_time_detection` and `run_video_detection` handle real-time and video file detection, respectively. They capture frames, process them through YOLOv3, and display the results. The process is optimized to ensure smooth performance, even on standard hardware.

```python
def run_real_time_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not found or could not be opened.")
        return
    # ... [Detection Loop]
```

### GUI Interface

The GUI is built using Tkinter, providing a simple interface for users to select between real-time detection and video processing. The GUI ensures a seamless user experience and makes the application accessible even to those without technical expertise.

```python
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
```

### Class-Aware Pseudo-Labeling

To address class imbalance, pseudo-labels are generated during the detection process. This involves identifying high-confidence detections and using them to create additional training data, particularly for underrepresented classes. The pseudo-labels are stored and can be used to fine-tune the model, helping it learn to detect minority classes more effectively.

```python
def generate_pseudo_label(image, boxes, confidences, class_ids, save_dir, filename):
    with open(os.path.join(save_dir, filename.replace('.jpg', '.txt')), 'w') as f:
        for i in range(len(boxes)):
            if confidences[i] >= pseudo_label_threshold:
                x, y, w, h = boxes[i]
                label = f"{class_ids[i]} {(x + w / 2) / input_size} {(y + h / 2) / input_size} {w / input_size} {h / input_size}\n"
                f.write(label)
        cv2.imwrite(os.path.join(save_dir, filename), image)
```

## Challenges and Considerations

### Class Imbalance

As discussed, class imbalance is a critical issue in object detection that can lead to biased models. The use of Class-Aware Pseudo-Labeling in this project is a step towards addressing this issue. However, it is important to note that pseudo-labeling alone may not completely solve the problem, and other techniques such as re-sampling, re-weighting, or advanced data augmentation might be necessary for further improvement.

### Performance and Efficiency

While YOLOv3 is designed for real-time performance, processing speed can vary based on the hardware used. The implementation in this project is optimized for standard CPUs, but further enhancements can be made for GPU acceleration or for deployment in resource-constrained environments.

## Future Work

- **Advanced Data Augmentation**: Implementing techniques like mosaic augmentation or mixup to further address class imbalance.
- **Cost-Sensitive Learning**: Exploring weighted loss functions to penalize errors on minority classes more heavily.
- - **Enhanced Model Training**: Integrating the pseudo-labels generated through this project into a continuous training pipeline, allowing the model to learn from new data over time and improve its accuracy, particularly for underrepresented classes.
- **Multi-Scale Detection**: Exploring the integration of multi-scale detection techniques to handle objects of varying sizes more effectively, especially in challenging scenarios with significant scale variation.

## Contributing

We welcome contributions to improve the functionality and performance of this project. If you would like to contribute, please follow these steps:

1. **Fork the Repository**: Create a personal fork of this repository.
2. **Create a Feature Branch**: Create a branch for your feature or bug fix.
3. **Submit a Pull Request**: Once your feature or fix is ready, submit a pull request for review.

Please ensure that your code adheres to the existing style and conventions used in the project. Contributions should also include appropriate tests to verify the functionality of new features or fixes.

## Acknowledgments

- **YOLOv3**: This project builds on the incredible work done by Joseph Redmon and Ali Farhadi in developing YOLOv3, which is available [here](https://pjreddie.com/darknet/yolo/).
- **COCO Dataset**: The COCO dataset, used in this project, is an invaluable resource for object detection research. Learn more about COCO [here](https://cocodataset.org/).
- **Tkinter**: Tkinter is a powerful and easy-to-use library for creating GUIs in Python. Documentation can be found [here](https://docs.python.org/3/library/tkinter.html).

## References

- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767. Available [here](https://arxiv.org/abs/1804.02767).
- Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. In *European Conference on Computer Vision* (pp. 740-755). Springer, Cham.
- Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning Transferable Architectures for Scalable Image Recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 8697-8710).
- Johnson, J. M., & Khoshgoftaar, T. M. (2019). Survey on Deep Learning with Class Imbalance. *Journal of Big Data, 6*(1), 1-54.

