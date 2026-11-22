# PPE Detection System

A computer vision application for detecting people and personal protective equipment (PPE) in construction-site images and videos using YOLO11 Nano.

**Live Application:** https://ppeguard.streamlit.app/  
**GitHub Repository:** https://github.com/davidmba819/PPE

## 1. Project Overview

The PPE Detection System uses a fine-tuned YOLO11 Nano object detection model to identify seven classes in construction-site images and videos.

The application allows users to:
- Upload construction-site images and videos.
- Adjust the detection confidence threshold.
- Run object detection using YOLO11.
- View annotated images and videos.
- Inspect detected classes and confidence scores.
- Download detection results.

The project covers dataset preparation, model experimentation, evaluation, application development and deployment.

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.14 | Programming language |
| PyTorch | Deep learning framework |
| Ultralytics | YOLO model training and inference |
| OpenCV | Image and video processing |
| FFmpeg | H.264 video encoding |
| Streamlit | Interactive application |
| Pandas | Detection result presentation |
| Pillow | Image processing |
| uv | Dependency and environment management |
| Git and GitHub | Version control |
| Google Colab | GPU-based model experimentation |
| Docker | Application containerization |

## 3. Dataset

**Source:** Roboflow Construction Hazard Detection, Version 97.

The original dataset contained 11 classes. Seven classes were selected for this project.

### Selected classes

| Class ID | Class name |
|---:|---|
| 0 | Hardhat |
| 1 | Mask |
| 2 | NO-Hardhat |
| 3 | NO-Mask |
| 4 | NO-Safety Vest |
| 5 | Person |
| 6 | Safety Vest |

Safety Cone, Machinery, Utility Pole and Vehicle were excluded.

### Dataset preparation

| Description | Count |
|---|---:|
| Original images | 6,689 |
| Original annotation files | 6,689 |
| Empty annotation files identified | 156 |
| Prepared images | 4,280 |
| Retained bounding boxes | 34,176 |
| Images excluded | 2,409 |

### Dataset split

| Subset | Images |
|---|---:|
| Training | 2,996 |
| Validation | 641 |
| Testing | 643 |
| **Total** | **4,280** |

Multilabel stratification was used for the dataset split. Dataset validation included image-label correspondence, class IDs, annotation structure and bounding-box coordinates.

The original dataset was preserved separately from the prepared dataset.

**Dataset license:** CC BY 4.0. The original dataset source must be appropriately attributed when redistributing or publishing derived work.

## 4. Model Experimentation

Three pretrained Nano object detection models were fine-tuned using the same prepared dataset and comparable training configurations.

### Models evaluated

1. YOLOv8 Nano
2. YOLO11 Nano
3. YOLO26 Nano

### Training configuration

| Parameter | Configuration |
|---|---|
| Image size | 640 |
| Maximum epochs | 50 |
| Batch size | 16 |
| Optimizer | Auto |
| Early stopping patience | 10 |
| Random seed | 42 |
| GPU | NVIDIA Tesla T4 |
| Workers | 2 |
| Initialization | Pretrained weights |

### Validation results

| Metric | YOLOv8 | YOLO11 | YOLO26 |
|---|---:|---:|---:|
| Precision | 0.827 | 0.844 | 0.856 |
| Recall | 0.611 | 0.606 | 0.608 |
| mAP@50 | 0.674 | 0.672 | 0.679 |
| mAP@50–95 | 0.389 | 0.390 | 0.396 |

### Test results

| Metric | YOLOv8 | YOLO11 | YOLO26 |
|---|---:|---:|---:|
| Precision | 0.817 | 0.835 | 0.840 |
| Recall | 0.605 | 0.596 | 0.596 |
| mAP@50 | 0.660 | 0.660 | 0.657 |
| mAP@50–95 | 0.386 | 0.391 | 0.383 |

### Model selection

YOLO11 Nano was selected as the application's detection model.

It achieved the highest test mAP@50–95 score of 0.391 among the three evaluated models.

The differences between the models were relatively small. YOLO26 achieved higher precision, while YOLOv8 achieved slightly higher recall on the test set.

**Evaluation limitation:** All three models were evaluated on the test set, and those results were used in the final model selection. Therefore, the test set was not preserved as a completely independent final evaluation set. This limitation should be considered when interpreting the reported results.

### Selected model artifact

`models/yolo11n_ppe_best.pt`

## 5. Application Architecture

The application uses a simple Streamlit architecture.

Streamlit handles the user interface and invokes the trained YOLO11 model directly. There is no separate API backend.

### Image detection workflow

1. User uploads an image.
2. The image is loaded and prepared.
3. YOLO11 performs inference.
4. Bounding boxes and class labels are generated.
5. Detection results are displayed.
6. The annotated image can be downloaded.

### Video detection workflow

1. User uploads a video.
2. OpenCV reads the video frames sequentially.
3. YOLO11 performs inference on each frame.
4. Annotated frames are generated.
5. OpenCV writes the intermediate video.
6. FFmpeg converts the video to H.264.
7. Streamlit displays the processed video.
8. The annotated video can be downloaded.

The application does not preserve the original video audio.

## 6. Application Features

### Image Detection
- JPG, JPEG and PNG uploads.
- Adjustable confidence threshold.
- YOLO11 inference.
- Annotated image visualization.
- Detection class and confidence display.
- Annotated image download.

### Video Detection
- MP4, AVI, MOV and MKV uploads.
- Sequential frame processing.
- YOLO11 inference on individual frames.
- Bounding-box visualization.
- Processing progress display.
- H.264 video conversion using FFmpeg.
- Annotated video playback.
- Annotated video download.

## 7. Project Structure

```text
PPE/
│
├── .devcontainer/
│
├── models/
│   └── yolo11n_ppe_best.pt
│
├── notebooks/
│   └── model_training.ipynb
│
├── src/
│   └── app.py
│
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── packages.txt
├── pyproject.toml
├── uv.lock
└── README.md
```

The dataset and training outputs are stored separately and are not part of the application source repository.

## 8. Local Setup

### Prerequisites

- Python 3.14
- uv
- FFmpeg
- Git

### Clone the repository

```bash
git clone https://github.com/davidmba819/PPE.git
cd PPE
```

### Install dependencies

```bash
uv sync
```

### Run the application

```bash
uv run streamlit run src/app.py
```

Streamlit will provide a local URL through which the application can be accessed.

## 9. Deployment

The application is deployed using Streamlit Community Cloud.

**Live application:** https://ppeguard.streamlit.app/

The deployment includes:
- The trained YOLO11 model.
- Python application dependencies.
- OpenCV video processing.
- FFmpeg for H.264 conversion.
- Streamlit application interface.

Image and video detection have been tested in the deployed application.

## 10. Model and Application Limitations

The current system performs object detection, not complete PPE compliance assessment.

Although it detects classes such as `Hardhat`, `NO-Hardhat`, `Safety Vest` and `NO-Safety Vest`, it does not currently implement:

- Person-to-PPE association.
- Individual worker compliance tracking.
- Persistent object tracking across video frames.
- Automated safety violation alerts.
- Real-time surveillance and monitoring.

Additional limitations include:
- Some objects may be missed because of relatively low recall.
- Small and distant objects are more difficult to detect.
- PPE classification can be inconsistent in certain scenes.
- Overlapping objects can produce challenging detection cases.
- The model has not been validated for safety-critical operational use.

The application should be treated as a computer vision prototype rather than a production-certified construction-site safety monitoring system.

## 11. Future Improvements

Potential future improvements include:

- Improving detection performance through additional experimentation.
- Testing inference performance on longer videos.
- Adding object tracking across video frames.
- Improving detection-result summaries.
- Adding person-to-PPE association.
- Implementing PPE compliance assessment.
- Improving application monitoring and error handling.
- Further Docker-based deployment and testing.

## 12. Conclusion

The PPE Detection System demonstrates an end-to-end computer vision workflow, from dataset preparation and comparative YOLO experimentation to model selection, application development and deployment.

YOLO11 Nano was selected following comparative evaluation of YOLOv8, YOLO11 and YOLO26.

The deployed Streamlit application currently supports construction-site image and video detection, annotated result visualization and downloads.
