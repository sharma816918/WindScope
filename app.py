import os
import uuid
import logging

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from scipy.signal import (
    detrend,
    find_peaks,
    savgol_filter
)
from scipy.fft import rfft, irfft, next_fast_len


# ============================================================
# WINDSCOPE - APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR, "static", "uploads"
)

app.config["RESULT_FOLDER"] = os.path.join(
    BASE_DIR, "static", "results"
)

ALLOWED_EXTENSIONS = {
    "mp4", "avi", "mov", "mkv", "webm"
}

MAX_ANALYSIS_FRAMES = 900
MAX_DETECTION_FRAMES = 120
MAX_ANALYSIS_DIMENSION = 640

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

logging.basicConfig(level=logging.INFO)


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):
    """Validate supported video file extensions."""

    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def video_info(path):
    """Read and validate video metadata."""

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError(
            "Unable to open this video. "
            "Try a valid MP4 or AVI video."
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    finally:
        cap.release()

    if (
        not np.isfinite(fps)
        or fps < 1
        or fps > 240
        or total_frames < 30
        or width < 32
        or height < 32
    ):
        raise ValueError(
            "Invalid video metadata. Use a readable video "
            "with at least 30 frames and a valid frame rate."
        )

    duration = total_frames / fps

    if duration < 1.0:
        raise ValueError(
            "The video is too short. Upload a longer video."
        )

    return fps, total_frames, width, height


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_frame(frame):
    """
    Convert a BGR frame into a denoised grayscale image.

    CLAHE improves local contrast.
    Gaussian blur reduces high-frequency noise.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    return enhanced


# ============================================================
# AUTOMATIC REGION OF INTEREST DETECTION
# ============================================================

def detect_roi(path):
    """
    Detect a candidate moving region using accumulated
    frame differences.

    This is a motion-based heuristic, not a guaranteed
    semantic wind-turbine detector.
    """

    fps, total, width, height = video_info(path)

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError(
            "Unable to open the video for motion detection."
        )

    scale = min(
        1.0,
        MAX_ANALYSIS_DIMENSION / max(width, height)
    )

    small_width = max(
        32,
        int(round(width * scale))
    )

    small_height = max(
        32,
        int(round(height * scale))
    )

    motion_map = np.zeros(
        (small_height, small_width),
        dtype=np.float32
    )

    previous = None
    valid_samples = 0

    sample_step = max(
        1,
        int(np.ceil(total / MAX_DETECTION_FRAMES))
    )

    try:

        frame_index = 0

        while frame_index < total:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index
            )

            ok, frame = cap.read()

            if not ok:
                frame_index += sample_step
                continue

            frame = cv2.resize(
                frame,
                (small_width, small_height),
                interpolation=cv2.INTER_AREA
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            if previous is not None:

                difference = cv2.absdiff(
                    gray,
                    previous
                ).astype(np.float32)

                noise_floor = max(
                    4.0,
                    float(np.percentile(difference, 55))
                )

                difference = np.maximum(
                    difference - noise_floor,
                    0
                )

                motion_map += difference

                valid_samples += 1

            previous = gray

            frame_index += sample_step

    finally:
        cap.release()

    if valid_samples < 5:
        raise ValueError(
            "Not enough readable frames for automatic "
            "motion detection."
        )

    motion_map /= valid_samples

    motion_map = cv2.GaussianBlur(
        motion_map,
        (0, 0),
        sigmaX=3
    )

    positive = motion_map[
        motion_map > 0
    ]

    if positive.size < 30:
        raise ValueError(
            "Insufficient motion detected. Upload clearer "
            "footage containing visible rotor movement."
        )

    max_motion = float(motion_map.max())

    if max_motion < 1.0:
        raise ValueError(
            "The video contains insufficient measurable motion."
        )

    threshold = max(
        float(np.percentile(positive, 65)),
        max_motion * 0.12
    )

    mask = (
        motion_map >= threshold
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = small_width * small_height

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        if area < image_area * 0.001:
            continue

        if area > image_area * 0.75:
            continue

        local_motion = motion_map[
            y:y + h,
            x:x + w
        ]

        score = float(
            np.sum(local_motion)
        )

        if score <= 0:
            continue

        candidates.append(
            (score, x, y, w, h)
        )

    if not candidates:
        raise ValueError(
            "Unable to identify a suitable moving region. "
            "Try a video with a clearer rotor and less "
            "background movement."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # Start with the strongest moving component.
    _, x, y, w, h = candidates[0]

    center_x = x + w / 2
    center_y = y + h / 2

    # Merge strong nearby components.
    selected = []

    for score, cx, cy, cw, ch in candidates:

        if score < candidates[0][0] * 0.12:
            continue

        component_center_x = cx + cw / 2
        component_center_y = cy + ch / 2

        nearby_x = (
            abs(component_center_x - center_x)
            <= max(small_width * 0.25, w)
        )

        nearby_y = (
            abs(component_center_y - center_y)
            <= max(small_height * 0.25, h)
        )

        if nearby_x and nearby_y:
            selected.append(
                (cx, cy, cw, ch)
            )

    if selected:

        x1 = min(
            item[0] for item in selected
        )

        y1 = min(
            item[1] for item in selected
        )

        x2 = max(
            item[0] + item[2]
            for item in selected
        )

        y2 = max(
            item[1] + item[3]
            for item in selected
        )

    else:

        x1, y1 = x, y
        x2, y2 = x + w, y + h

    # Add padding around the detected region.
    pad_x = max(
        6,
        int((x2 - x1) * 0.18)
    )

    pad_y = max(
        6,
        int((y2 - y1) * 0.18)
    )

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(
        small_width,
        x2 + pad_x
    )

    y2 = min(
        small_height,
        y2 + pad_y
    )

    # Convert coordinates to original resolution.
    x1 = int(x1 / scale)
    y1 = int(y1 / scale)

    x2 = int(np.ceil(x2 / scale))
    y2 = int(np.ceil(y2 / scale))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))

    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    roi_area = (x2 - x1) * (y2 - y1)

    if roi_area > width * height * 0.80:
        raise ValueError(
            "The detected region is too large. "
            "The rotor could not be isolated reliably."
        )

    return x1, y1, x2, y2


# ============================================================
# MOTION SIGNAL EXTRACTION
# ============================================================

def motion_signal(path, roi):

    fps, total, width, height = video_info(path)

    x1, y1, x2, y2 = roi

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError(
            "Unable to reopen the uploaded video."
        )

    # Sample uniformly when videos contain many frames.
    step = max(
        1,
        int(np.ceil(total / MAX_ANALYSIS_FRAMES))
    )

    effective_fps = fps / step

    times = []
    signal = []

    preview = None
    previous = None

    frame_index = 0

    try:

        while frame_index < total:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index
            )

            ok, frame = cap.read()

            if not ok:
                frame_index += step
                continue

            if preview is None:
                preview = frame.copy()

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                raise ValueError(
                    "The detected ROI is empty."
                )

            # Resize large ROI regions for faster processing.
            crop_height, crop_width = crop.shape[:2]

            crop_scale = min(
                1.0,
                400.0 / max(
                    crop_width,
                    crop_height
                )
            )

            if crop_scale < 1.0:

                crop = cv2.resize(
                    crop,
                    (
                        max(16, int(crop_width * crop_scale)),
                        max(16, int(crop_height * crop_scale))
                    ),
                    interpolation=cv2.INTER_AREA
                )

            processed = preprocess_frame(crop)

            if previous is None:

                value = 0.0

            else:

                difference = cv2.absdiff(
                    processed,
                    previous
                ).astype(np.float32)

                # Limit outlier influence from abrupt pixel changes.
                difference = np.minimum(
                    difference,
                    80.0
                )

                value = float(
                    np.mean(difference)
                )

            signal.append(value)

            times.append(
                frame_index / fps
            )

            previous = processed

            frame_index += step

    finally:
        cap.release()

    times = np.asarray(
        times,
        dtype=float
    )

    signal = np.asarray(
        signal,
        dtype=float
    )

    if len(signal) < 30:
        raise ValueError(
            "Not enough readable frames for signal analysis."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "The extracted motion signal contains invalid values."
        )

    if np.std(signal) < 1e-4:
        raise ValueError(
            "No meaningful motion variation was detected. "
            "Try clearer footage."
        )

    return (
        fps,
        effective_fps,
        total,
        width,
        height,
        times,
        signal,
        preview
    )


# ============================================================
# SIGNAL PROCESSING AND ROTATION ESTIMATION
# ============================================================

def estimate_rotation(raw, sample_fps, blade_count):

    raw = np.asarray(
        raw,
        dtype=float
    )

    n = len(raw)

    empty_result = {
        "period": None,
        "omega": None,
        "rpm": None,
        "interval": None,
        "peaks": [],
        "signal": raw,
        "quality": 0.0,
        "status": "Insufficient evidence for a reliable estimate."
    }

    if (
        n < 30
        or sample_fps <= 0
        or not np.all(np.isfinite(raw))
    ):
        return empty_result

    # Remove slow illumination and camera-intensity trends.
    signal = detrend(
        raw,
        type="linear"
    )

    signal -= np.mean(signal)

    signal_std = float(
        np.std(signal)
    )

    if signal_std < 1e-4:
        return {
            **empty_result,
            "status": "Motion signal is too weak."
        }

    signal /= signal_std

    # Adaptive smoothing.
    window = max(
        5,
        int(round(sample_fps * 0.08))
    )

    if window % 2 == 0:
        window += 1

    # Savitzky-Golay window must be smaller than signal length.
    window = min(
        window,
        n if n % 2 else n - 1
    )

    if window >= 5:

        signal = savgol_filter(
            signal,
            window_length=window,
            polyorder=2,
            mode="interp"
        )

    signal -= np.mean(signal)

    # --------------------------------------------------------
    # NORMALIZED AUTOCORRELATION
    # --------------------------------------------------------

    fft_length = next_fast_len(
        2 * n - 1
    )

    spectrum = rfft(
        signal,
        n=fft_length
    )

    autocorrelation = irfft(
        spectrum * np.conjugate(spectrum),
        n=fft_length
    )[:n]

    # Correct for the decreasing number of overlapping samples.
    autocorrelation /= np.arange(
        n,
        0,
        -1,
        dtype=float
    )

    if autocorrelation[0] <= 1e-12:
        return {
            **empty_result,
            "status": "Autocorrelation could not identify a pattern."
        }

    autocorrelation /= autocorrelation[0]

    # Search only periods that are observable in the sampled signal.
    # The minimum interval is approximately 0.15 seconds.
    min_lag = max(
        2,
        int(np.ceil(sample_fps * 0.15))
    )

    # A period must repeat at least twice in the available signal.
    max_lag = min(
        n // 2,
        int(np.floor(sample_fps * 10.0))
    )

    if max_lag <= min_lag:
        return {
            **empty_result,
            "status": "The clip is too short for stable periodicity analysis."
        }

    region = autocorrelation[
        min_lag:max_lag + 1
    ]

    candidate_indices, properties = find_peaks(
        region,
        prominence=0.04,
        distance=max(
            1,
            int(sample_fps * 0.08)
        )
    )

    candidate_lags = (
        candidate_indices + min_lag
    )

    if len(candidate_lags) == 0:
        return {
            **empty_result,
            "status": (
                "No clear repeating motion pattern was found. "
                "Use a longer video with a stationary camera."
            )
        }

    prominences = properties.get(
        "prominences",
        np.zeros(len(candidate_lags))
    )

    # Rank by correlation strength and prominence.
    scores = (
        autocorrelation[candidate_lags]
        + 0.25 * prominences
    )

    order = np.argsort(scores)[::-1]

    chosen_lag = None
    chosen_quality = 0.0
    chosen_prominence = 0.0

    for index in order:

        lag = int(
            candidate_lags[index]
        )

        repeat_count = (
            (n - 1) / lag
        )

        if repeat_count < 2:
            continue

        correlation_strength = float(
            autocorrelation[lag]
        )

        prominence = float(
            prominences[index]
        )

        # Reject weak or poorly defined periodicity.
        if (
            correlation_strength < 0.15
            or prominence < 0.04
        ):
            continue

        chosen_lag = lag
        chosen_quality = correlation_strength
        chosen_prominence = prominence

        break

    if chosen_lag is None:
        return {
            **empty_result,
            "status": (
                "Repeating motion was not sufficiently strong. "
                "RPM has been withheld."
            )
        }

    passage_interval = (
        chosen_lag / sample_fps
    )

    # --------------------------------------------------------
    # ROTOR PERIOD ESTIMATION
    # --------------------------------------------------------

    # IMPORTANT:
    # This conversion assumes the selected interval corresponds
    # to one blade-passage cycle and that the blades produce
    # distinguishable, approximately periodic motion.
    #
    # If autocorrelation instead selects a full revolution or
    # a harmonic, multiplying by blade_count will be incorrect.
    #
    # This is therefore an estimate, not a guaranteed exact RPM.

    period = (
        passage_interval * blade_count
    )

    if (
        not np.isfinite(period)
        or period <= 0
    ):
        return {
            **empty_result,
            "status": "The estimated period is invalid."
        }

    # The signal should contain at least two estimated rotor cycles.
    duration = (n - 1) / sample_fps

    if duration / period < 2:
        return {
            **empty_result,
            "interval": passage_interval,
            "status": (
                "Insufficient complete rotor cycles for a "
                "defensible RPM estimate."
            )
        }

    # A rough signal-quality indicator, not a statistical
    # probability that the result is correct.
    quality = float(
        np.clip(
            0.7 * chosen_quality
            + 0.3 * min(chosen_prominence, 1.0),
            0.0,
            1.0
        )
    )

    graph_peaks, _ = find_peaks(
        signal,
        prominence=max(
            0.20,
            float(np.std(signal) * 0.20)
        ),
        distance=max(
            1,
            int(chosen_lag * 0.45)
        )
    )

    graph_peaks = graph_peaks[
        (graph_peaks > 0)
        & (graph_peaks < n - 1)
    ]

    if quality < 0.20:
        return {
            **empty_result,
            "interval": passage_interval,
            "peaks": graph_peaks.tolist(),
            "signal": signal,
            "quality": quality,
            "status": (
                "Low confidence. RPM has been withheld "
                "because the periodicity is weak."
            )
        }

    omega = (
        2.0 * np.pi / period
    )

    rpm = (
        60.0 / period
    )

    return {
        "period": float(period),
        "omega": float(omega),
        "rpm": float(rpm),
        "interval": float(passage_interval),
        "peaks": graph_peaks.tolist(),
        "signal": signal,
        "quality": quality,
        "status": (
            "Candidate rotational estimate calculated. "
            "Verify blade-passage interpretation and compare "
            "against a known reference speed."
        )
    }


# ============================================================
# SAVE MOTION GRAPH
# ============================================================

def save_motion_graph(
    result_id,
    times,
    raw,
    processed,
    peaks
):

    filename = (
        f"{result_id}_graph.png"
    )

    path = os.path.join(
        app.config["RESULT_FOLDER"],
        filename
    )

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    try:

        ax.plot(
            times,
            raw,
            alpha=0.55,
            linewidth=1.0,
            label="Raw motion signal"
        )

        ax.plot(
            times,
            processed,
            linewidth=1.5,
            label="Processed signal"
        )

        if len(peaks) > 0:

            ax.scatter(
                times[peaks],
                processed[peaks],
                s=20,
                label="Detected peaks"
            )

        ax.set_title(
            "WindScope - Rotor Motion Analysis"
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Normalized motion intensity"
        )

        ax.grid(
            True,
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight"
        )

    finally:
        plt.close(fig)

    return filename


# ============================================================
# SAVE ROI PREVIEW
# ============================================================

def save_roi_preview(
    result_id,
    preview,
    roi
):

    filename = (
        f"{result_id}_roi.jpg"
    )

    path = os.path.join(
        app.config["RESULT_FOLDER"],
        filename
    )

    x1, y1, x2, y2 = roi

    annotated = preview.copy()

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (0, 220, 160),
        max(
            2,
            int(annotated.shape[1] / 400)
        )
    )

    success = cv2.imwrite(
        path,
        annotated,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not success:
        raise RuntimeError(
            "Could not save the ROI preview."
        )

    return filename


# ============================================================
# COMPLETE VIDEO ANALYSIS
# ============================================================

def analyze_video(
    path,
    result_id,
    blade_count
):

    fps, total, width, height = video_info(
        path
    )

    # 1. Automatically detect the moving region.
    roi = detect_roi(
        path
    )

    x1, y1, x2, y2 = roi

    # 2. Extract the motion signal.
    (
        fps,
        sample_fps,
        total,
        width,
        height,
        times,
        raw_signal,
        preview
    ) = motion_signal(
        path,
        roi
    )

    if preview is None:
        raise ValueError(
            "Could not generate the video preview."
        )

    # 3. Estimate periodic motion.
    estimate = estimate_rotation(
        raw_signal,
        sample_fps,
        blade_count
    )

    processed_signal = estimate["signal"]

    # 4. Generate graph and ROI preview.
    graph_filename = save_motion_graph(
        result_id,
        times,
        raw_signal,
        processed_signal,
        estimate["peaks"]
    )

    roi_filename = save_roi_preview(
        result_id,
        preview,
        roi
    )

    # 5. Prepare template-compatible results.
    period = estimate["period"]
    omega = estimate["omega"]
    rpm = estimate["rpm"]
    interval = estimate["interval"]

    return {
        "fps": round(float(fps), 2),

        "sample_fps": round(
            float(sample_fps),
            2
        ),

        "frames": len(times),

        "duration": round(
            float(times[-1] - times[0]),
            2
        ),

        "original_frames": int(total),

        "width": int(width),

        "height": int(height),

        "peak_count": len(
            estimate["peaks"]
        ),

        "interval": (
            round(float(interval), 4)
            if interval is not None
            else None
        ),

        "period": (
            round(float(period), 4)
            if period is not None
            else None
        ),

        "omega": (
            round(float(omega), 4)
            if omega is not None
            else None
        ),

        "rpm": (
            round(float(rpm), 3)
            if rpm is not None
            else None
        ),

        "blade_count": int(
            blade_count
        ),

        "status": estimate["status"],

        "roi_method": (
            "Automatic motion-based detection"
        ),

        "roi_pixels": (
            f"x={x1}:{x2}, y={y1}:{y2}"
        ),

        "quality": round(
            float(estimate["quality"]),
            3
        ),

        "graph": url_for(
            "static",
            filename=f"results/{graph_filename}"
        ),

        "roi_preview": url_for(
            "static",
            filename=f"results/{roi_filename}"
        ),
    }


# ============================================================
# MAIN FLASK ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        video_path = None
        result_id = uuid.uuid4().hex[:12]

        try:

            uploaded = request.files.get(
                "video"
            )

            if (
                uploaded is None
                or not uploaded.filename
            ):
                raise ValueError(
                    "Please select a wind-turbine video first."
                )

            if not allowed_file(
                uploaded.filename
            ):
                raise ValueError(
                    "Unsupported video format. "
                    "Use MP4, AVI, MOV, MKV or WEBM."
                )

            # UI se manual form input hata diya gaya hai.
            # Modern wind turbines me automatically 3 blades assume karna sabse accurate hota hai.
            blade_count = 3

            safe_filename = secure_filename(
                uploaded.filename
            )

            if not safe_filename:
                raise ValueError(
                    "The uploaded filename is invalid."
                )

            video_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                f"{result_id}_{safe_filename}"
            )

            uploaded.save(
                video_path
            )

            logging.info(
                "Starting WindScope analysis: %s",
                result_id
            )

            result = analyze_video(
                video_path,
                result_id,
                blade_count
            )

            logging.info(
                "WindScope analysis completed: %s",
                result_id
            )

        except ValueError as exc:

            error = str(exc)

        except Exception:

            app.logger.exception(
                "Unexpected WindScope processing error."
            )

            error = (
                "Analysis could not be completed. "
                "Try a clearer video or check the server terminal."
            )

        finally:

            # Uploaded source videos are temporary.
            if (
                video_path
                and os.path.isfile(video_path)
            ):

                try:

                    os.remove(
                        video_path
                    )

                except OSError:

                    app.logger.warning(
                        "Unable to remove temporary video: %s",
                        video_path
                    )

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ============================================================
# APPLICATION STARTUP
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
