
from pathlib import Path
from io import BytesIO
import tempfile
import subprocess

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n_ppe_best.pt"

st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide"
)


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


model = load_model()


# ==========================================
# 3. SIDEBAR SETTINGS
# ==========================================

st.sidebar.title("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

st.sidebar.caption(
    "Adjust the minimum confidence required for detections."
)


# ==========================================
# 4. MAIN APPLICATION
# ==========================================

st.title("Construction Site PPE Detection")

st.write(
    "Detect construction-site objects using our trained YOLO11 model."
)

detection_mode = st.radio(
    "Select Detection Mode",
    ["Image Detection", "Video Detection"],
    horizontal=True
)


# ==========================================
# 5. IMAGE DETECTION
# ==========================================

if detection_mode == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")
        st.image(image, width="stretch")

        if st.button("Run Image Detection", type="primary"):

            with st.spinner("Running YOLO11 inference..."):

                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    verbose=False
                )

                result = results[0]
                annotated_image = result.plot()[:, :, ::-1]

            st.subheader("Detection Results")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    annotated_image,
                    caption="Annotated Image",
                    width="stretch"
                )

            with col2:

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:

                    detections = []

                    for box in boxes:

                        class_id = int(box.cls.item())
                        score = float(box.conf.item())

                        detections.append({
                            "Class": model.names[class_id],
                            "Confidence": round(score, 4)
                        })

                    df = pd.DataFrame(detections)

                    st.metric("Total Detections", len(detections))

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                else:
                    st.info("No objects detected in this image.")

            output_image = Image.fromarray(annotated_image)

            image_buffer = BytesIO()
            output_image.save(image_buffer, format="JPEG")

            st.download_button(
                "Download Annotated Image",
                data=image_buffer.getvalue(),
                file_name="ppe_detection_result.jpg",
                mime="image/jpeg"
            )



# ==========================================
# 6. VIDEO DETECTION
# ==========================================

elif detection_mode == "Video Detection":

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.subheader("Original Video")
        st.video(uploaded_video)

        if st.button("Run Video Detection", type="primary"):

            with st.spinner("Processing video..."):

                try:
                    with tempfile.TemporaryDirectory() as temp_dir:

                        temp_dir = Path(temp_dir)

                        input_path = temp_dir / (
                            "input" + Path(uploaded_video.name).suffix.lower()
                        )

                        raw_output_path = temp_dir / "annotated_video.mp4"
                        output_path = temp_dir / "annotated_video_h264.mp4"

                        # Save uploaded video
                        input_path.write_bytes(uploaded_video.getvalue())

                        cap = cv2.VideoCapture(str(input_path))

                        if not cap.isOpened():
                            raise RuntimeError(
                                "Unable to open the uploaded video."
                            )

                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        if fps <= 0:
                            fps = 25.0

                        if width <= 0 or height <= 0:
                            cap.release()
                            raise RuntimeError("Invalid video dimensions.")

                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                        writer = cv2.VideoWriter(
                            str(raw_output_path),
                            fourcc,
                            fps,
                            (width, height)
                        )

                        if not writer.isOpened():
                            cap.release()
                            raise RuntimeError(
                                "Unable to create the intermediate video."
                            )

                        progress = st.progress(0)
                        status = st.empty()

                        frame_count = 0

                        try:
                            while True:

                                success, frame = cap.read()

                                if not success:
                                    break

                                result = model.predict(
                                    source=frame,
                                    conf=confidence,
                                    imgsz=640,
                                    verbose=False
                                )[0]

                                annotated_frame = result.plot()

                                writer.write(annotated_frame)

                                frame_count += 1

                                if total_frames > 0:
                                    progress.progress(
                                        min(frame_count / total_frames, 1.0)
                                    )

                                status.write(
                                    f"Processed {frame_count} / "
                                    f"{total_frames} frames"
                                )

                        finally:
                            cap.release()
                            writer.release()

                        if frame_count == 0:
                            raise RuntimeError(
                                "No video frames could be processed."
                            )

                        if (
                            not raw_output_path.exists()
                            or raw_output_path.stat().st_size == 0
                        ):
                            raise RuntimeError(
                                "OpenCV failed to generate the intermediate video."
                            )

                        # ==================================
                        # FFMPEG H.264 CONVERSION
                        # ==================================

                        conversion = subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i", str(raw_output_path),
                                "-c:v", "libx264",
                                "-preset", "ultrafast",
                                "-crf", "23",
                                "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart",
                                str(output_path)
                            ],
                            check=True,
                            capture_output=True,
                            text=True
                        )

                        if (
                            not output_path.exists()
                            or output_path.stat().st_size == 0
                        ):
                            raise RuntimeError(
                                "FFmpeg did not generate the final video."
                            )

                        # Read output while temporary directory exists
                        video_bytes = output_path.read_bytes()

                    # ==================================
                    # DISPLAY RESULTS
                    # ==================================

                    st.success("Video detection completed.")

                    st.subheader("Annotated Video")

                    st.video(video_bytes, format="video/mp4")

                    st.metric("Total Frames Processed", frame_count)

                    st.download_button(
                        "Download Annotated Video",
                        data=video_bytes,
                        file_name="ppe_detection_result.mp4",
                        mime="video/mp4"
                    )

                    st.caption(
                        "The processed video does not retain the original audio."
                    )

                except subprocess.CalledProcessError as error:
                    st.error("FFmpeg conversion failed.")
                    st.code(error.stderr[-4000:])

                except Exception as error:
                    st.error("Video detection failed.")
                    st.exception(error)
