#!/usr/bin/env python3
import os

# Jetson Nano OpenBLAS compatibility.
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

import csv
import time
import ctypes
import ctypes.util
import hashlib
import threading

import cv2
import numpy as np
import tensorrt as trt

try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except Exception:
    JTOP_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

ENGINE_PATH = (
    "swinsmall_mask2former_pma_"
    "256x512_CONSTPOS_fp32.engine"
)

EXPECTED_ENGINE_SHA256 = (
    "b788e354be4c6dbc00f006a80cde2eb"
    "215481732a94e0fda603b8ce0aae9312d"
)

PLUGIN_PATH = (
    "/home/jet/jetson_setup/mmdeploy/"
    "build_trt_ops/lib/"
    "libmmdeploy_tensorrt_ops.so"
)

# Exact validated model resolution.
INPUT_HEIGHT = 256
INPUT_WIDTH = 512

# USB camera.
CAMERA_INDEX = 0
CAMERA_DEVICE = f"/dev/video{CAMERA_INDEX}"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_READ_FAILURE_LIMIT = 5
CAMERA_REOPEN_DELAY_SEC = 1.0

# One HDMI stream containing two synchronized panels.
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
DISPLAY_FPS = 30

# Each panel keeps the USB camera 4:3 aspect ratio.
PANEL_WIDTH = 880
PANEL_HEIGHT = 660
LEFT_PANEL_X = 40
RIGHT_PANEL_X = 1000
PANEL_Y = 180

CSV_PATH = "metrics_swinsmall_mask2former_pma_dualview.csv"

# Segmentation transparency.
OVERLAY_CAMERA_WEIGHT = 0.55
OVERLAY_MASK_WEIGHT = 0.45


# ============================================================
# HIGH-CONTRAST 19-CLASS PALETTE
# ============================================================

HIGH_CONTRAST_PALETTE = np.array([
    [255, 255, 255],   # 0  Road
    [240, 30, 200],    # 1  Sidewalk
    [60, 80, 120],     # 2  Building
    [130, 90, 60],     # 3  Wall
    [180, 130, 50],    # 4  Fence
    [200, 200, 200],   # 5  Pole
    [255, 50, 50],     # 6  Traffic Light
    [50, 255, 255],    # 7  Traffic Sign
    [0, 180, 80],      # 8  Vegetation
    [120, 160, 40],    # 9  Terrain
    [0, 0, 0],         # 10 Sky
    [255, 50, 255],    # 11 Person
    [255, 160, 0],     # 12 Rider
    [0, 0, 255],       # 13 Car
    [0, 0, 130],       # 14 Truck
    [0, 120, 255],     # 15 Bus
    [100, 0, 100],     # 16 Train
    [0, 150, 150],     # 17 Motorcycle
    [180, 0, 0],       # 18 Bicycle
], dtype=np.uint8)


# ============================================================
# MODEL PRE/POST-PROCESSING
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def preprocess(frame):
    resized = cv2.resize(
        frame,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    tensor = rgb.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    tensor = (tensor - mean) / std
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(
        tensor,
        dtype=np.float32
    )


def stable_softmax(x, axis=-1):
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)

    exp_x = np.exp(x)

    return (
        exp_x
        /
        np.sum(exp_x, axis=axis, keepdims=True)
    )


def stable_sigmoid(x):
    x = x.astype(np.float32, copy=False)

    output = np.empty_like(
        x,
        dtype=np.float32
    )

    positive = x >= 0

    output[positive] = (
        1.0
        /
        (1.0 + np.exp(-x[positive]))
    )

    negative = ~positive

    exp_x = np.exp(x[negative])

    output[negative] = (
        exp_x
        /
        (1.0 + exp_x)
    )

    return output


def mask2former_postprocess(
    class_logits,
    mask_logits
):
    """
    class_logits: [1,100,20]
    mask_logits : [1,100,64,128]

    Returns:
        prediction [256,512] uint8
    """

    class_logits = class_logits[0]
    mask_logits = mask_logits[0]

    class_prob = stable_softmax(
        class_logits,
        axis=-1
    )[:, :-1]

    # Resize all 100 mask queries together.
    mask_hwc = np.transpose(
        mask_logits,
        (1, 2, 0)
    )

    mask_up_hwc = cv2.resize(
        mask_hwc,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    mask_up = np.transpose(
        mask_up_hwc,
        (2, 0, 1)
    )

    mask_prob = stable_sigmoid(
        mask_up
    )

    semantic_flat = (
        class_prob.T
        @
        mask_prob.reshape(100, -1)
    )

    semantic = semantic_flat.reshape(
        19,
        INPUT_HEIGHT,
        INPUT_WIDTH
    )

    prediction = np.argmax(
        semantic,
        axis=0
    ).astype(np.uint8)

    return prediction


# ============================================================
# ROBUST GSTREAMER USB CAMERA
# ============================================================

def camera_pipeline_candidates():
    """
    Prefer MJPEG because it reduces USB bandwidth.
    appsink drops stale frames while inference is slow.
    """

    return [
        (
            "MJPEG",
            (
                f"v4l2src device={CAMERA_DEVICE} io-mode=2 ! "
                f"image/jpeg,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},"
                f"framerate={CAMERA_FPS}/1 ! "
                "jpegdec ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false"
            )
        ),

        (
            "YUY2",
            (
                f"v4l2src device={CAMERA_DEVICE} io-mode=2 ! "
                f"video/x-raw,format=YUY2,width={CAMERA_WIDTH},"
                f"height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false"
            )
        ),

        (
            "RAW_AUTO",
            (
                f"v4l2src device={CAMERA_DEVICE} io-mode=2 ! "
                f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT} ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false"
            )
        ),
    ]


def safe_release_capture(cap, timeout_sec=3.0):
    """
    Avoid an indefinite shutdown hang if the camera backend
    becomes unhealthy.
    """

    if cap is None:
        return

    done = threading.Event()

    def _release():
        try:
            cap.release()
        except Exception:
            pass
        finally:
            done.set()

    thread = threading.Thread(
        target=_release,
        daemon=True
    )

    thread.start()
    thread.join(timeout_sec)

    if done.is_set():
        print("[PASS] Camera capture released")
    else:
        print(
            "[WARN] Camera release exceeded "
            f"{timeout_sec:.1f}s; continuing shutdown."
        )


def open_camera_gstreamer():
    print()
    print("[INFO] Opening USB camera through GStreamer:")
    print("       device =", CAMERA_DEVICE)

    for pipeline_name, pipeline in camera_pipeline_candidates():

        print()
        print(
            f"[INFO] Trying camera pipeline: {pipeline_name}"
        )

        cap = cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER
        )

        if not cap.isOpened():

            print(
                f"[WARN] {pipeline_name} did not open."
            )

            safe_release_capture(
                cap,
                timeout_sec=1.0
            )

            continue

        successful_reads = 0
        last_frame = None

        for _ in range(5):

            ok, frame = cap.read()

            if (
                ok
                and frame is not None
                and frame.size > 0
            ):
                successful_reads += 1
                last_frame = frame
            else:
                break

        if successful_reads >= 3:

            print(
                f"[PASS] Camera pipeline selected: "
                f"{pipeline_name}"
            )

            print(
                "       initial successful reads =",
                successful_reads
            )

            print(
                "       frame shape =",
                last_frame.shape
            )

            return cap, pipeline_name

        print(
            f"[WARN] {pipeline_name} opened but "
            f"was not stable "
            f"({successful_reads}/5 reads)."
        )

        safe_release_capture(
            cap,
            timeout_sec=1.0
        )

    raise RuntimeError(
        "No stable GStreamer camera pipeline "
        f"could be opened for {CAMERA_DEVICE}."
    )


# ============================================================
# HDMI DUAL-VIEW COMPOSITOR
# ============================================================

def make_dual_view(
    raw_frame,
    segmented_frame,
    fps_smoothed,
    inference_ms,
    preprocess_ms,
    postprocess_ms,
    total_ms,
    gpu,
    ram,
    power,
    temp
):
    """
    Create one 1920x1080 HDMI frame containing:

        LEFT  = synchronized raw USB-camera frame
        RIGHT = segmentation result from the SAME frame

    This is ideal for research comparison and can be cropped
    into two independent sources inside OBS if desired.
    """

    canvas = np.zeros(
        (
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    raw_panel = cv2.resize(
        raw_frame,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    segmented_panel = cv2.resize(
        segmented_frame,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    canvas[
        PANEL_Y:PANEL_Y + PANEL_HEIGHT,
        LEFT_PANEL_X:LEFT_PANEL_X + PANEL_WIDTH
    ] = raw_panel

    canvas[
        PANEL_Y:PANEL_Y + PANEL_HEIGHT,
        RIGHT_PANEL_X:RIGHT_PANEL_X + PANEL_WIDTH
    ] = segmented_panel

    # Main title.
    cv2.putText(
        canvas,
        "Jetson Nano 2GB Developer Kit",
        (665, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Panel labels.
    cv2.putText(
        canvas,
        "USB CAMERA INPUT",
        (LEFT_PANEL_X + 245, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "SEGMENTED OUTPUT",
        (RIGHT_PANEL_X + 210, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Panel borders.
    cv2.rectangle(
        canvas,
        (LEFT_PANEL_X - 2, PANEL_Y - 2),
        (
            LEFT_PANEL_X + PANEL_WIDTH + 2,
            PANEL_Y + PANEL_HEIGHT + 2
        ),
        (0, 255, 255),
        3
    )

    cv2.rectangle(
        canvas,
        (RIGHT_PANEL_X - 2, PANEL_Y - 2),
        (
            RIGHT_PANEL_X + PANEL_WIDTH + 2,
            PANEL_Y + PANEL_HEIGHT + 2
        ),
        (0, 255, 0),
        3
    )

    # Vertical separator.
    cv2.line(
        canvas,
        (960, 105),
        (960, 865),
        (90, 90, 90),
        2
    )

    # Bottom telemetry panel.
    cv2.rectangle(
        canvas,
        (40, 885),
        (1880, 1045),
        (15, 15, 15),
        -1
    )

    line1 = (
        f"FPS {fps_smoothed:.2f}   |   "
        f"TRT {inference_ms:.0f} ms   |   "
        f"Pre {preprocess_ms:.0f} ms   |   "
        f"Post {postprocess_ms:.0f} ms   |   "
        f"Total {total_ms:.0f} ms"
    )

    line2 = (
        f"GPU {gpu}   |   RAM {ram}   |   "
        f"Power {power}   |   Temp {temp}"
    )

    cv2.putText(
        canvas,
        line1,
        (85, 945),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        line2,
        (85, 1005),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# JETSON TELEMETRY
# ============================================================

def get_jetson_stats(jetson):
    gpu = 0
    ram = 0
    power = 0
    temp = 0

    if jetson is None:
        return gpu, ram, power, temp

    try:
        if not jetson.ok():
            return gpu, ram, power, temp

        stats = jetson.stats

        gpu = stats.get("GPU", 0)
        ram = stats.get("RAM", 0)

        power = stats.get(
            "power cur",
            stats.get("Power TOT", 0)
        )

        temp = stats.get(
            "Temp GPU",
            stats.get("Temp CPU", 0)
        )

    except Exception:
        pass

    return gpu, ram, power, temp


# ============================================================
# CUDA RUNTIME
# ============================================================

def load_cuda_runtime():
    candidates = [
        ctypes.util.find_library("cudart"),
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda-10.2/lib64/libcudart.so",
    ]

    cudart = None

    for candidate in candidates:

        if not candidate:
            continue

        try:
            cudart = ctypes.CDLL(candidate)

            print(
                "[INFO] CUDA runtime:",
                candidate
            )

            break

        except OSError:
            pass

    if cudart is None:
        raise RuntimeError(
            "Could not load CUDA runtime."
        )

    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]

    cudart.cudaSetDevice.restype = ctypes.c_int
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]

    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t
    ]

    cudart.cudaFree.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [
        ctypes.c_void_p
    ]

    cudart.cudaMemcpy.restype = ctypes.c_int
    cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int
    ]

    cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    return cudart


def cuda_check(
    cudart,
    code,
    operation
):
    if code != 0:

        message = cudart.cudaGetErrorString(
            code
        )

        if message:

            message = message.decode(
                "utf-8",
                errors="replace"
            )

        raise RuntimeError(
            operation
            + " failed: "
            + str(code)
            + " "
            + str(message)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "[INFO] Swin-Small + Mask2Former + PMA"
    )

    print(
        "[INFO] Dual-view Jetson Nano 2GB deployment"
    )

    print(
        "[INFO] Left: raw USB input | "
        "Right: synchronized segmentation"
    )

    # --------------------------------------------------------
    # Validate exact engine
    # --------------------------------------------------------

    if not os.path.exists(ENGINE_PATH):
        raise FileNotFoundError(ENGINE_PATH)

    engine_sha = sha256_file(
        ENGINE_PATH
    )

    print()
    print("[INFO] Engine SHA256:")
    print(engine_sha)

    if engine_sha != EXPECTED_ENGINE_SHA256:
        raise RuntimeError(
            "Engine SHA256 mismatch."
        )

    print(
        "[PASS] Validated TensorRT engine identity"
    )

    # --------------------------------------------------------
    # CUDA + TensorRT
    # --------------------------------------------------------

    cudart = load_cuda_runtime()

    cuda_check(
        cudart,
        cudart.cudaSetDevice(0),
        "cudaSetDevice"
    )

    logger = trt.Logger(
        trt.Logger.WARNING
    )

    trt.init_libnvinfer_plugins(
        logger,
        ""
    )

    if not os.path.exists(PLUGIN_PATH):
        raise FileNotFoundError(
            PLUGIN_PATH
        )

    ctypes.CDLL(
        PLUGIN_PATH,
        mode=ctypes.RTLD_GLOBAL
    )

    print(
        "[PASS] MMDeploy grid_sampler plugin loaded"
    )

    with open(ENGINE_PATH, "rb") as f:
        engine_bytes = f.read()

    runtime = trt.Runtime(logger)

    engine = runtime.deserialize_cuda_engine(
        engine_bytes
    )

    if engine is None:
        raise RuntimeError(
            "Engine deserialization failed."
        )

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "Could not create execution context."
        )

    print(
        "[PASS] TensorRT engine loaded"
    )

    if engine.num_bindings != 3:
        raise RuntimeError(
            "Expected exactly 3 bindings."
        )

    input_index = engine.get_binding_index(
        "pixel_values"
    )

    mask_index = engine.get_binding_index(
        "masks_queries_logits"
    )

    class_index = engine.get_binding_index(
        "class_queries_logits"
    )

    if min(
        input_index,
        mask_index,
        class_index
    ) < 0:
        raise RuntimeError(
            "Required TensorRT binding missing."
        )

    print()
    print(
        "[INFO] Input:",
        tuple(
            engine.get_binding_shape(
                input_index
            )
        )
    )

    print(
        "[INFO] Masks:",
        tuple(
            engine.get_binding_shape(
                mask_index
            )
        )
    )

    print(
        "[INFO] Class:",
        tuple(
            engine.get_binding_shape(
                class_index
            )
        )
    )

    # --------------------------------------------------------
    # Persistent CUDA buffers
    # --------------------------------------------------------

    CUDA_H2D = 1
    CUDA_D2H = 2

    bindings = [
        0
    ] * engine.num_bindings

    host_mask = np.empty(
        (1, 100, 64, 128),
        dtype=np.float32
    )

    host_class = np.empty(
        (1, 100, 20),
        dtype=np.float32
    )

    input_nbytes = (
        1
        * 3
        * INPUT_HEIGHT
        * INPUT_WIDTH
        * 4
    )

    d_input = ctypes.c_void_p()
    d_mask = ctypes.c_void_p()
    d_class = ctypes.c_void_p()

    device_ptrs = []

    cuda_check(
        cudart,
        cudart.cudaMalloc(
            ctypes.byref(d_input),
            input_nbytes
        ),
        "cudaMalloc input"
    )

    cuda_check(
        cudart,
        cudart.cudaMalloc(
            ctypes.byref(d_mask),
            host_mask.nbytes
        ),
        "cudaMalloc masks"
    )

    cuda_check(
        cudart,
        cudart.cudaMalloc(
            ctypes.byref(d_class),
            host_class.nbytes
        ),
        "cudaMalloc class"
    )

    device_ptrs.extend([
        d_input,
        d_mask,
        d_class
    ])

    bindings[input_index] = int(
        d_input.value
    )

    bindings[mask_index] = int(
        d_mask.value
    )

    bindings[class_index] = int(
        d_class.value
    )

    print(
        "[PASS] Persistent CUDA buffers allocated"
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        "[INFO] Warming up TensorRT engine..."
    )

    dummy_input = np.zeros(
        (
            1,
            3,
            INPUT_HEIGHT,
            INPUT_WIDTH
        ),
        dtype=np.float32
    )

    cuda_check(
        cudart,
        cudart.cudaMemcpy(
            d_input,
            ctypes.c_void_p(
                dummy_input.ctypes.data
            ),
            dummy_input.nbytes,
            CUDA_H2D
        ),
        "Warmup H2D"
    )

    if not context.execute_v2(
        bindings
    ):
        raise RuntimeError(
            "Warmup inference failed."
        )

    cuda_check(
        cudart,
        cudart.cudaDeviceSynchronize(),
        "Warmup synchronize"
    )

    print(
        "[PASS] Warmup complete"
    )

    # --------------------------------------------------------
    # Robust USB camera
    # --------------------------------------------------------

    cap, camera_pipeline_name = (
        open_camera_gstreamer()
    )

    consecutive_camera_failures = 0

    # --------------------------------------------------------
    # HDMI output
    # --------------------------------------------------------

    print()
    print(
        "[INFO] Opening 1920x1080 HDMI dual-view output..."
    )

    gst_out = (
        "appsrc ! "
        "video/x-raw, "
        "format=BGR, "
        f"width={DISPLAY_WIDTH}, "
        f"height={DISPLAY_HEIGHT}, "
        f"framerate={DISPLAY_FPS}/1 ! "
        "queue max-size-buffers=1 "
        "leaky=downstream ! "
        "videoconvert ! "
        "video/x-raw,format=BGRx ! "
        "nvvidconv ! "
        "video/x-raw(memory:NVMM),format=NV12 ! "
        "nvoverlaysink "
        "display-id=0 "
        "overlay-x=0 "
        "overlay-y=0 "
        "overlay-w=1920 "
        "overlay-h=1080 "
        "sync=false"
    )

    out = cv2.VideoWriter(
        gst_out,
        cv2.CAP_GSTREAMER,
        0,
        float(DISPLAY_FPS),
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )
    )

    if not out.isOpened():
        raise RuntimeError(
            "Could not open HDMI GStreamer output."
        )

    print(
        "[PASS] HDMI dual-view output ready"
    )

    # --------------------------------------------------------
    # CSV metrics
    # --------------------------------------------------------

    csv_file = open(
        CSV_PATH,
        mode="w",
        newline=""
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow([
        "Timestamp",
        "Frame",
        "FPS",
        "Preprocess_ms",
        "Inference_ms",
        "Postprocess_ms",
        "Total_ms",
        "GPU_Usage",
        "RAM",
        "Power",
        "Temperature"
    ])

    fps_smoothed = 0.0
    frame_number = 0

    # --------------------------------------------------------
    # jtop telemetry
    # --------------------------------------------------------

    jetson_context = None

    if JTOP_AVAILABLE:

        try:

            jetson_context = jtop()
            jetson_context.start()

            print(
                "[PASS] jtop telemetry started"
            )

        except Exception as exc:

            print(
                "[WARN] jtop unavailable:",
                exc
            )

            jetson_context = None

    print()
    print(
        "[SUCCESS] DUAL VIEW LIVE"
    )

    print(
        "[INFO] HDMI -> capture card -> OBS"
    )

    print(
        "[INFO] Camera pipeline ->",
        camera_pipeline_name
    )

    print(
        "[INFO] LEFT  = raw input frame"
    )

    print(
        "[INFO] RIGHT = segmented SAME frame"
    )

    print(
        "[INFO] Metrics ->",
        CSV_PATH
    )

    print(
        "[INFO] Press CTRL+C to stop."
    )

    print()

    try:

        while True:

            frame_start = time.perf_counter()

            # ------------------------------------------------
            # Camera
            # ------------------------------------------------

            ret, frame = cap.read()

            if (
                not ret
                or frame is None
                or frame.size == 0
            ):

                consecutive_camera_failures += 1

                print()
                print(
                    "[WARN] Camera frame read failed "
                    f"({consecutive_camera_failures}/"
                    f"{CAMERA_READ_FAILURE_LIMIT})."
                )

                if (
                    consecutive_camera_failures
                    >= CAMERA_READ_FAILURE_LIMIT
                ):

                    print(
                        "[INFO] Reopening GStreamer camera..."
                    )

                    safe_release_capture(
                        cap
                    )

                    time.sleep(
                        CAMERA_REOPEN_DELAY_SEC
                    )

                    cap, camera_pipeline_name = (
                        open_camera_gstreamer()
                    )

                    consecutive_camera_failures = 0

                continue

            consecutive_camera_failures = 0
            frame_number += 1

            # Keep an untouched copy for the LEFT research panel.
            raw_display_frame = frame.copy()

            # ------------------------------------------------
            # Preprocess
            # ------------------------------------------------

            t0 = time.perf_counter()

            input_tensor = preprocess(
                frame
            )

            preprocess_ms = (
                (
                    time.perf_counter()
                    - t0
                )
                * 1000.0
            )

            # ------------------------------------------------
            # TensorRT inference
            # ------------------------------------------------

            t1 = time.perf_counter()

            cuda_check(
                cudart,
                cudart.cudaMemcpy(
                    d_input,
                    ctypes.c_void_p(
                        input_tensor.ctypes.data
                    ),
                    input_tensor.nbytes,
                    CUDA_H2D
                ),
                "Frame H2D"
            )

            inference_ok = context.execute_v2(
                bindings
            )

            if not inference_ok:
                raise RuntimeError(
                    "TensorRT frame inference failed."
                )

            cuda_check(
                cudart,
                cudart.cudaDeviceSynchronize(),
                "Frame synchronize"
            )

            cuda_check(
                cudart,
                cudart.cudaMemcpy(
                    ctypes.c_void_p(
                        host_mask.ctypes.data
                    ),
                    d_mask,
                    host_mask.nbytes,
                    CUDA_D2H
                ),
                "Mask D2H"
            )

            cuda_check(
                cudart,
                cudart.cudaMemcpy(
                    ctypes.c_void_p(
                        host_class.ctypes.data
                    ),
                    d_class,
                    host_class.nbytes,
                    CUDA_D2H
                ),
                "Class D2H"
            )

            inference_ms = (
                (
                    time.perf_counter()
                    - t1
                )
                * 1000.0
            )

            if not np.isfinite(
                host_class
            ).all():
                raise RuntimeError(
                    "Class logits became NaN/Inf."
                )

            if not np.isfinite(
                host_mask
            ).all():
                raise RuntimeError(
                    "Mask logits became NaN/Inf."
                )

            # ------------------------------------------------
            # Segmentation
            # ------------------------------------------------

            t2 = time.perf_counter()

            prediction = (
                mask2former_postprocess(
                    host_class,
                    host_mask
                )
            )

            color_mask_rgb = (
                HIGH_CONTRAST_PALETTE[
                    prediction
                ]
            )

            color_mask_bgr = cv2.cvtColor(
                color_mask_rgb,
                cv2.COLOR_RGB2BGR
            )

            # Resize segmentation back to ORIGINAL camera size,
            # so both left/right panels have the same geometry.
            color_mask_camera = cv2.resize(
                color_mask_bgr,
                (
                    frame.shape[1],
                    frame.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

            segmented_frame = cv2.addWeighted(
                frame,
                OVERLAY_CAMERA_WEIGHT,
                color_mask_camera,
                OVERLAY_MASK_WEIGHT,
                0
            )

            postprocess_ms = (
                (
                    time.perf_counter()
                    - t2
                )
                * 1000.0
            )

            # ------------------------------------------------
            # Timing + telemetry
            # ------------------------------------------------

            total_ms = (
                (
                    time.perf_counter()
                    - frame_start
                )
                * 1000.0
            )

            current_fps = (
                1000.0 / total_ms
                if total_ms > 0
                else 0.0
            )

            if fps_smoothed == 0.0:
                fps_smoothed = current_fps
            else:
                fps_smoothed = (
                    0.85 * fps_smoothed
                    +
                    0.15 * current_fps
                )

            gpu, ram, power, temp = (
                get_jetson_stats(
                    jetson_context
                )
            )

            # ------------------------------------------------
            # Synchronized side-by-side HDMI frame
            # ------------------------------------------------

            display_frame = make_dual_view(
                raw_frame=raw_display_frame,
                segmented_frame=segmented_frame,
                fps_smoothed=fps_smoothed,
                inference_ms=inference_ms,
                preprocess_ms=preprocess_ms,
                postprocess_ms=postprocess_ms,
                total_ms=total_ms,
                gpu=gpu,
                ram=ram,
                power=power,
                temp=temp
            )

            out.write(
                display_frame
            )

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            csv_writer.writerow([
                timestamp,
                frame_number,
                round(fps_smoothed, 4),
                round(preprocess_ms, 3),
                round(inference_ms, 3),
                round(postprocess_ms, 3),
                round(total_ms, 3),
                gpu,
                ram,
                power,
                temp
            ])

            if frame_number % 10 == 0:
                csv_file.flush()

            print(
                f"\rFrame {frame_number} | "
                f"FPS {fps_smoothed:.2f} | "
                f"TRT {inference_ms:.1f} ms | "
                f"Post {postprocess_ms:.1f} ms | "
                f"Total {total_ms:.1f} ms",
                end="",
                flush=True
            )

    except KeyboardInterrupt:

        print()
        print(
            "[INFO] CTRL+C received."
        )

    finally:

        print()
        print(
            "[INFO] Releasing hardware..."
        )

        safe_release_capture(
            cap
        )

        try:
            out.release()
        except Exception:
            pass

        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass

        if jetson_context is not None:

            try:
                jetson_context.close()
            except Exception:
                pass

        for ptr in device_ptrs:

            try:
                if ptr and ptr.value:
                    cudart.cudaFree(ptr)
            except Exception:
                pass

        print(
            "[PASS] Camera shutdown requested"
        )

        print(
            "[PASS] HDMI output released"
        )

        print(
            "[PASS] Metrics saved:"
        )

        print(
            CSV_PATH
        )


if __name__ == "__main__":
    main()
