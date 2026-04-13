import gradio as gr
import numpy as np
import cv2
import io
import random
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from datasets import load_dataset

# Load model & dataset
model = YOLO("Model/sived_yolov8m_obb_best.pt")
dataset = load_dataset("AIOmarRehan/SIVED", split="test")


# Helper: pick a random test image
def get_random_image():
    idx = random.randint(0, len(dataset) - 1)
    return dataset[idx]["image"]


# Main detection pipeline
def run_detection(image, confidence_threshold):
    if image is None:
        empty_df = pd.DataFrame(
            columns=["ID", "Confidence", "Angle (°)", "Area (px²)",
                     "Center", "Width", "Height"]
        )
        return None, None, [], None, empty_df

    # Inference
    img_np = np.array(image)
    results = model.predict(source=img_np, imgsz=640,
                            conf=confidence_threshold, verbose=False)
    result = results[0]

    orig = result.orig_img.copy()
    if len(orig.shape) == 2:
        img_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    annotated = img_rgb.copy()
    heatmap_base = np.zeros((h, w), dtype=np.float32)

    crops = []
    detections = []
    confidences = []

    if result.obb is not None and len(result.obb) > 0:
        obbs = result.obb.xyxyxyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()

        for i, (obb, conf) in enumerate(zip(obbs, confs)):
            pts = obb.astype(np.int32)
            confidences.append(float(conf))

            # Color by confidence
            if conf >= 0.8:
                color = (0, 230, 118)       # green
            elif conf >= 0.5:
                color = (255, 213, 79)      # amber
            else:
                color = (255, 82, 82)       # red

            # Draw OBB polygon
            cv2.polylines(annotated, [pts], True, color, 2)

            # Label
            cx, cy = pts.mean(axis=0).astype(int)
            label = f"Vehicle {conf:.2f}"
            (tw, th_t), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                annotated,
                (cx - tw // 2 - 3, cy - th_t - 6),
                (cx + tw // 2 + 3, cy + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                annotated, label, (cx - tw // 2, cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

            # Heatmap - fast circle + blur approach
            radius = max(20, int(min(h, w) * 0.04))
            cv2.circle(heatmap_base, (cx, cy), radius, float(conf), -1)

            # Crop individual detection
            rect = cv2.minAreaRect(pts)
            box_w, box_h = rect[1]
            angle = rect[2]

            pad = 15
            x_min = max(0, pts[:, 0].min() - pad)
            y_min = max(0, pts[:, 1].min() - pad)
            x_max = min(w, pts[:, 0].max() + pad)
            y_max = min(h, pts[:, 1].max() + pad)

            crop = img_rgb[y_min:y_max, x_min:x_max].copy()
            if crop.size > 0:
                shifted = pts.copy()
                shifted[:, 0] -= x_min
                shifted[:, 1] -= y_min
                cv2.polylines(crop, [shifted], True, color, 2)
                crops.append((
                    Image.fromarray(crop),
                    f"#{i+1}  conf {conf:.3f}"
                ))

            # Detection row
            area = cv2.contourArea(pts)
            detections.append({
                "ID": i + 1,
                "Confidence": f"{conf:.4f}",
                "Angle (°)": f"{angle:.1f}",
                "Area (px²)": f"{area:.0f}",
                "Center": f"({cx}, {cy})",
                "Width": f"{box_w:.1f}",
                "Height": f"{box_h:.1f}",
            })

    n_det = len(detections)

    # Tab 1: Annotated image
    annotated_pil = Image.fromarray(annotated)

    # Tab 2: Confidence heatmap overlay
    if heatmap_base.max() > 0:
        sigma = max(25, int(min(h, w) * 0.06))
        heatmap_blur = cv2.GaussianBlur(heatmap_base, (0, 0), sigma)
        hm_norm = (heatmap_blur / heatmap_blur.max() * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.55, hm_color, 0.45, 0)
        cv2.putText(
            overlay, f"{n_det} vehicles detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        heatmap_pil = Image.fromarray(overlay)
    else:
        heatmap_pil = Image.fromarray(img_rgb)

    # Tab 4: Statistics
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    if confidences:
        axes[0].hist(
            confidences, bins=20, range=(0, 1),
            color="#42A5F5", edgecolor="black", alpha=0.85,
        )
        axes[0].axvline(
            np.mean(confidences), color="#E53935", linestyle="--", linewidth=1.5,
            label=f"Mean: {np.mean(confidences):.3f}",
        )
        axes[0].set_title("Confidence Distribution", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Confidence")
        axes[0].set_ylabel("Count")
        axes[0].legend(fontsize=10)
        axes[0].set_xlim(0, 1)

        stats_text = (
            f"DETECTION SUMMARY\n"
            f"{'━' * 32}\n"
            f"Total Detections:   {n_det}\n"
            f"Avg Confidence:     {np.mean(confidences):.4f}\n"
            f"Min Confidence:     {np.min(confidences):.4f}\n"
            f"Max Confidence:     {np.max(confidences):.4f}\n"
            f"Std Confidence:     {np.std(confidences):.4f}\n"
            f"{'━' * 32}\n"
            f"High  (≥ 0.80):     {sum(1 for c in confidences if c >= 0.8)}\n"
            f"Med   (0.50–0.80):  {sum(1 for c in confidences if 0.5 <= c < 0.8)}\n"
            f"Low   (< 0.50):     {sum(1 for c in confidences if c < 0.5)}\n"
        )
    else:
        axes[0].text(
            0.5, 0.5, "No detections at this\nconfidence threshold.",
            ha="center", va="center", fontsize=13, color="gray",
        )
        axes[0].set_title("Confidence Distribution", fontsize=13, fontweight="bold")
        axes[0].axis("off")
        stats_text = (
            f"DETECTION SUMMARY\n"
            f"{'━' * 32}\n"
            f"No vehicles detected.\n"
            f"Try lowering the threshold.\n"
        )

    axes[1].axis("off")
    axes[1].text(
        0.08, 0.5, stats_text, fontsize=12, fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", alpha=0.9),
    )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    stats_pil = Image.open(buf)
    plt.close(fig)

    # Tab 5: Detection table
    if detections:
        df = pd.DataFrame(detections)
    else:
        df = pd.DataFrame(
            columns=["ID", "Confidence", "Angle (°)", "Area (px²)",
                     "Center", "Width", "Height"]
        )

    return annotated_pil, heatmap_pil, crops, stats_pil, df


# Video detection pipeline
def annotate_frame(frame, result):
    """Draw OBB detections on a single frame (in-place)."""
    if result.obb is not None and len(result.obb) > 0:
        obbs = result.obb.xyxyxyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()

        for obb, conf in zip(obbs, confs):
            pts = obb.astype(np.int32)

            # Color by confidence
            if conf >= 0.8:
                color = (0, 230, 118)
            elif conf >= 0.5:
                color = (255, 213, 79)
            else:
                color = (255, 82, 82)

            cv2.polylines(frame, [pts], True, color, 2)

            cx, cy = pts.mean(axis=0).astype(int)
            label = f"Vehicle {conf:.2f}"
            (tw, th_t), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                frame,
                (cx - tw // 2 - 3, cy - th_t - 6),
                (cx + tw // 2 + 3, cy + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                frame, label, (cx - tw // 2, cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

    # Detection count overlay
    n = len(result.obb) if result.obb is not None else 0
    cv2.putText(
        frame, f"{n} vehicles", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


def run_video_detection(video_path, confidence_threshold, process_every_n):
    if video_path is None:
        return None

    process_every_n = int(process_every_n)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))

    frame_idx = 0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only run inference every Nth frame; reuse last result otherwise
        if frame_idx % process_every_n == 0:
            results = model.predict(source=frame, imgsz=640,
                                    conf=confidence_threshold, verbose=False)
            last_result = results[0]

        if last_result is not None:
            frame = annotate_frame(frame, last_result)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return tmp.name


# Gradio UI
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
    title="SIVED — SAR Vehicle Detection",
    css="""
        .gr-button { min-width: 140px; }
        footer { display: none !important; }
    """,
) as demo:

    gr.Markdown(
        """
        # SIVED - SAR Image Vehicle Detection
        **YOLOv8m-OBB** fine-tuned on the
        [SIVED](https://huggingface.co/datasets/AIOmarRehan/SIVED) dataset
        for oriented bounding-box detection of vehicles in SAR imagery.

        Upload a SAR image or click **Random Test Image** to try a sample
        from the dataset. You can also upload a **video** for real-time
        frame-by-frame detection.
        """
    )

    with gr.Row():
        # Left column: inputs
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input SAR Image")
            confidence_slider = gr.Slider(
                0.05, 0.95, value=0.25, step=0.05,
                label="Confidence Threshold",
            )
            with gr.Row():
                random_btn = gr.Button(
                    "Random Test Image", variant="secondary",
                )
                detect_btn = gr.Button(
                    "Detect Vehicles", variant="primary",
                )

        # Right column: outputs
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Annotated Image"):
                    output_annotated = gr.Image(
                        type="pil", label="Detections",
                    )
                with gr.Tab("Confidence Heatmap"):
                    output_heatmap = gr.Image(
                        type="pil", label="Detection Density Overlay",
                    )
                with gr.Tab("Cropped Detections"):
                    output_gallery = gr.Gallery(
                        label="Individual Detections",
                        columns=5, object_fit="contain", height="auto",
                    )
                with gr.Tab("Statistics"):
                    output_stats = gr.Image(
                        type="pil", label="Detection Statistics",
                    )

    output_table = gr.Dataframe(
        label="Detection Table", interactive=False,
    )

    # Outputs list (reused across handlers)
    outputs = [
        output_annotated,
        output_heatmap,
        output_gallery,
        output_stats,
        output_table,
    ]

    # Event wiring
    random_btn.click(get_random_image, outputs=input_image).then(
        run_detection,
        inputs=[input_image, confidence_slider],
        outputs=outputs,
    )

    detect_btn.click(
        run_detection,
        inputs=[input_image, confidence_slider],
        outputs=outputs,
    )

    confidence_slider.release(
        run_detection,
        inputs=[input_image, confidence_slider],
        outputs=outputs,
    )

    # Video detection section
    gr.Markdown("---")
    gr.Markdown("### Video Detection")

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(
                label="Upload SAR Video",
                sources=["upload"],
            )
            video_conf_slider = gr.Slider(
                0.05, 0.95, value=0.25, step=0.05,
                label="Video Confidence Threshold",
            )
            video_skip_slider = gr.Slider(
                1, 30, value=5, step=1,
                label="Process Every N Frames (higher = faster)",
            )
            video_btn = gr.Button(
                "Detect in Video", variant="primary",
            )
        with gr.Column(scale=2):
            output_video = gr.Video(label="Detection Output")

    video_btn.click(
        run_video_detection,
        inputs=[input_video, video_conf_slider, video_skip_slider],
        outputs=output_video,
    )

demo.launch()
