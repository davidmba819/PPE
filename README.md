# PPE Detection System

## 1. Project Overview

The PPE Detection System is a computer vision application that uses a fine-tuned YOLO11 Nano object detection model to identify people and personal protective equipment (PPE) in construction-site images and videos.

The system detects seven object classes and provides an interface for users to upload images or videos, perform object detection, visualize annotated results and download the outputs.

The project covers the complete workflow from dataset preparation and model experimentation to local application development.

### Project Objectives

- Develop and evaluate YOLO object detection models for construction-site imagery.
- Compare YOLOv8 Nano, YOLO11 Nano and YOLO26 Nano.
- Select a suitable trained model based on evaluation results.
- Develop an interactive application using Streamlit.
- Support image and video object detection.
- Package and deploy the application.

## 2. Technology Stack

### Development Environment

| Technology | Purpose |
|---|---|
| Python 3.14.6 | Main programming language |
| uv | Python project and dependency management |
| VS Code | Local development environment |
| Git and GitHub | Version control and source-code hosting |
| Google Colab | Model experimentation and GPU training |
| Google Drive | Dataset and model artifact storage |

### Installed Python Dependencies

| Library | Purpose |
|---|---|
| Ultralytics | YOLO model loading and inference |
| PyTorch | Deep learning framework |
| TorchVision | Computer vision utilities |
| NumPy | Numerical operations and array manipulation |
| OpenCV (opencv-python-headless) | Image and video processing |
| Pillow | Image loading and manipulation |
| Pandas | Detection results and tabular presentation |
| Streamlit | Interactive web application |
| Pytest | Application testing |

The main dependencies were installed using:

```bash
uv add ultralytics opencv-python-headless pandas pillow streamlit
```

Development dependency:

```bash
uv add --dev pytest
```

PyTorch, TorchVision and NumPy were resolved as dependencies of the Ultralytics installation.

## 3. Dataset Preparation

### Dataset Source

The project uses the Roboflow Construction Hazard Detection dataset, version 97.

The original dataset contained 11 classes. Seven classes were selected for this project.

### Selected Classes

| Class ID | Class Name |
|---|---|
| 0 | Hardhat |
| 1 | Mask |
| 2 | NO-Hardhat |
| 3 | NO-Mask |
| 4 | NO-Safety Vest |
| 5 | Person |
| 6 | Safety Vest |

The following classes were excluded:

- Safety Cone
- Machinery
- Utility Pole
- Vehicle

### Dataset Preparation Results

| Description | Result |
|---|---:|
| Original images | 6,689 |
| Original annotation files | 6,689 |
| Empty annotation files identified | 156 |
| Images in prepared dataset | 4,280 |
| Retained bounding boxes | 34,176 |
| Images excluded during preparation | 2,409 |

The prepared dataset was divided into training, validation and testing subsets.

| Subset | Images |
|---|---:|
| Training | 2,996 |
| Validation | 641 |
| Testing | 643 |
| Total | 4,280 |

Multilabel stratification was used for the split.

Dataset validation checked image-label correspondence, class IDs, annotation structure and bounding-box coordinates.

The original dataset was preserved separately from the prepared dataset.

**Dataset License:** CC BY 4.0. The original dataset source and attribution should be acknowledged when redistributing or publishing the project.

## 4. Model Experimentation

Three Nano object detection models were trained using the same prepared dataset and comparable experimental configurations.

### Models

1. YOLOv8 Nano
2. YOLO11 Nano
3. YOLO26 Nano

### Training Configuration

| Parameter | Configuration |
|---|---|
| Image size | 640 |
| Maximum epochs | 50 |
| Batch size | 16 |
| Optimizer | Auto |
| Early stopping patience | 10 |
| Random seed | 42 |
| Device | NVIDIA Tesla T4 |
| Workers | 2 |
| Initialization | Pretrained weights |

### Training Results

| Model | Training Time | Parameters |
|---|---:|---:|
| YOLOv8 Nano | 0.847 hours | 3.13M |
| YOLO11 Nano | 1.079 hours | 2.60M |
| YOLO26 Nano | 1.166 hours | 2.39M |

### Validation Results

| Metric | YOLOv8 | YOLO11 | YOLO26 |
|---|---:|---:|---:|
| Precision | 0.827 | 0.844 | 0.856 |
| Recall | 0.611 | 0.606 | 0.608 |
| mAP@50 | 0.674 | 0.672 | 0.679 |
| mAP@50–95 | 0.389 | 0.390 | 0.396 |

### Test Results

| Metric | YOLOv8 | YOLO11 | YOLO26 |
|---|---:|---:|---:|
| Precision | 0.817 | 0.835 | 0.840 |
| Recall | 0.605 | 0.596 | 0.596 |
| mAP@50 | 0.660 | 0.660 | 0.657 |
| mAP@50–95 | 0.386 | 0.391 | 0.383 |

### Model Selection

YOLO11 Nano was selected as the final model.

The primary selection criterion was test-set mAP@50–95.

YOLO11 achieved 0.391, the highest result among the three models on the test set.

Although YOLO26 achieved the highest validation precision and mAP scores, its test mAP@50–95 was lower than YOLO11.

YOLOv8 achieved slightly higher recall on the test set, but its overall localization metric was lower.

The differences between the models were relatively small. Therefore, YOLO11 was selected as the most suitable model based on the comparative results, not because it decisively outperformed the alternatives.

**Evaluation limitation:** All three models were evaluated on the test set, and the test results were used in the final selection. Consequently, the test set was not preserved as a completely independent final evaluation set. This limitation should be acknowledged when interpreting the reported performance.

### Selected Model Artifact

```text
models/yolo11n_ppe_best.pt
```

The trained model was copied from the YOLO11 experiment's best checkpoint and committed to the GitHub repository.

## 5. Model Performance and Observations

The models achieved approximately:

- 83–86% validation precision.
- 60–61% validation recall.
- 0.67 validation mAP@50.
- 0.39 validation mAP@50–95.

The results indicate that the models can detect construction-site objects, but detection performance is not yet sufficiently strong for safety-critical deployment.

### Observed Limitations

1. Recall is relatively low, meaning some objects are missed.
2. Small and distant objects are more difficult to detect.
3. PPE classification is inconsistent in some scenes.
4. Overlapping objects can produce challenging detection cases.
5. Confidence scores vary across objects and image conditions.
6. The validation and test mAP@50–95 values indicate room for improvement in localization quality.

### Local Inference Performance

During local CPU inference, the selected YOLO11 model produced the following approximate timings:

| Operation | Time |
|---|---:|
| Preprocessing | 7.2 ms |
| Inference | 191.1 ms |
| Postprocessing | 7.5 ms |
| Total | 205.8 ms |

This corresponds to approximately 4.9 images per second under the measured test conditions.

This is an initial local performance measurement, not a comprehensive application benchmark.

## 6. Application Development

### Architecture

The application uses a simple Streamlit architecture.

Streamlit handles the user interface and calls the trained YOLO11 model directly.

No separate FastAPI backend is used.

### Current Project Structure

```text
PPE/
│
├── models/
│   └── yolo11n_ppe_best.pt
│
├── assets/
│   └── sample_images/
│
├── notebooks/
│   └── model_training.ipynb
│
├── src/
│   └── app.py
│
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

### Application Features

#### Image Detection

Implemented functionality:

- Upload JPG, JPEG and PNG images.
- Configure the confidence threshold.
- Run YOLO11 inference.
- Generate bounding boxes and class labels.
- Display annotated images.
- Display detected classes and confidence scores.
- Download annotated images.

#### Video Detection

Implemented functionality:

- Upload MP4, AVI, MOV and MKV videos.
- Read uploaded videos using OpenCV.
- Extract video frames sequentially.
- Run YOLO11 inference on individual frames.
- Generate annotated frames.
- Write processed frames into an output video.
- Display the processed-frame count.
- Provide an annotated video download.

The video inference process has successfully completed during local testing.

**Outstanding issue:** The generated MP4 uses OpenCV's `mp4v` codec. The browser currently reports that it cannot find a supported video format and MIME type. H.264 encoding using FFmpeg is the planned fix. Browser playback has not yet been confirmed.

The basic video-processing implementation does not preserve original audio.

## 7. Development Challenges and Setbacks

Several technical issues were encountered during development.

| Issue | Cause | Resolution or Current Status |
|---|---|---|
| PyTorch installation timeout | Large package download and unstable network connection | Retried dependency installation with a longer timeout |
| Model checkpoint not found | Incorrect relative path from the `src` directory | Corrected project-root resolution using `pathlib` |
| Streamlit slider error | Zero was supplied as the slider step | Changed step to 0.05 |
| Deprecated Streamlit image argument | `use_column_width` is no longer supported | Replaced with `width="stretch"` |
| YOLO prediction argument error | Used `img_size` instead of `imgsz` | Corrected the argument |
| Dataset inference took too long | The test script processed thousands of images | Limited inference testing to five images |
| Existing image annotations | Some uploaded images already contained embedded bounding boxes | Switched to clean images for functional testing |
| Video browser playback failure | Generated MP4 codec is not reliably supported by browsers | H.264 conversion with FFmpeg remains to be implemented |

These challenges helped validate the environment, model-loading process, inference pipeline and application workflow.

## 8. Current Development Status

| Component | Status |
|---|---|
| Dataset preparation | Completed |
| YOLOv8 experimentation | Completed |
| YOLO11 experimentation | Completed |
| YOLO26 experimentation | Completed |
| Model comparison | Completed |
| Final model selection | Completed |
| Model artifact copied locally | Completed |
| Python environment setup | Completed |
| Dependency installation | Completed |
| Local model loading | Completed |
| Local image inference | Completed |
| Streamlit image detection | Working |
| Image result download | Implemented |
| Video frame inference | Working |
| Annotated video generation | Working |
| Browser video playback | Requires codec fix |
| Docker packaging | Not started |
| Deployment | Not started |

## 9. Remaining Development Roadmap

### Phase 1: Complete the MVP

- Fix browser video playback through H.264 encoding.
- Verify downloaded video playback.
- Test image and video uploads with different files.
- Handle invalid uploads and video-processing errors.
- Test confidence-threshold adjustments.

### Phase 2: Application Refinement

- Improve the Streamlit interface.
- Improve detection-result presentation.
- Add useful detection summaries.
- Improve processing feedback for longer videos.

### Phase 3: Containerization

- Create a Dockerfile.
- Build and test the Docker image.
- Verify model loading and inference inside the container.

### Phase 4: Deployment

- Prepare the application for Hugging Face Spaces.
- Configure model artifact availability.
- Deploy the application.
- Test the deployed image and video workflows.

## 10. Project Scope and Limitations

This project currently performs object detection, not complete PPE compliance assessment.

Although the model detects classes such as `Hardhat`, `NO-Hardhat`, `Safety Vest` and `NO-Safety Vest`, it does not implement:

- Person-to-PPE association.
- Individual worker compliance tracking.
- Persistent object tracking across video frames.
- Automated safety violation alerts.
- Real-time surveillance or monitoring.

The current application should therefore be treated as a prototype object detection system, not a production-certified construction-site safety monitoring solution.

## 11. Conclusion

The project has progressed from dataset preparation and comparative YOLO experimentation to a locally working Streamlit application.

YOLO11 Nano was selected based on its test-set mAP@50–95 result of 0.391.

The trained model successfully loads and performs local image and video inference. Image detection and annotated-image downloading are working. Video processing and annotated-video generation also work, but browser playback still requires a codec correction.

The next priority is to complete video playback, validate the full application and proceed to Docker packaging and deployment.
