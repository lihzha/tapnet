import os
from glob import glob

import jax
import mediapy as media
import numpy as np

from tapnet.models import tapir_model
from tapnet.utils import model_utils, transforms, viz_utils

MODEL_TYPE = "bootstapir"  # 'tapir' or 'bootstapir'
CKPT_DIR = (
    "/home/irom-lab/projects/lang_annotation/utils/point_trackers/tapnet/checkpoints"
)
if MODEL_TYPE == "tapir":
    checkpoint_path = f"{CKPT_DIR}/tapir_checkpoint_panning.npy"
else:
    checkpoint_path = f"{CKPT_DIR}/bootstapir_checkpoint_v2.npy"
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state["params"], ckpt_state["state"]

kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
if MODEL_TYPE == "bootstapir":
    kwargs.update(dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0))
tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)

def load_video_from_images(image_dir):
    frames = []
    num_frames = len(glob(f"{image_dir}/*.png"))
    for i in range(num_frames):
        img_path = f"{image_dir}/{i:04d}.png"
        img = media.read_image(img_path)
        frames.append(img)
    return np.array(frames)


def get_keypoint_tracking(
    image_dir, query_points: np.ndarray, visualize=False, image_dims=(224, 224)
):

    video = load_video_from_images(image_dir)

    def inference(frames, query_points):
        """Inference on one video.

        Args:
        frames: [num_frames, height, width, 3], [0, 255], np.uint8
        query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

        Returns:
        tracks: [num_points, 3], [-1, 1], [t, y, x]
        visibles: [num_points, num_frames], bool
        """
        # Preprocess video to match model inputs format
        frames = model_utils.preprocess_frames(frames)
        query_points = query_points.astype(np.float32)
        frames, query_points = frames[None], query_points[None]  # Add batch dimension

        outputs = tapir(
            video=frames,
            query_points=query_points,
            is_training=False,
            query_chunk_size=32,
        )
        tracks, occlusions, expected_dist = (
            outputs["tracks"],
            outputs["occlusion"],
            outputs["expected_dist"],
        )

        # Binarize occlusions
        visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
        return tracks[0], visibles[0]

    inference = jax.jit(inference)

    def sample_random_points(frame_max_idx, height, width, num_points):
        """Sample random points with (time, height, width) order."""
        y = np.random.randint(0, height, (num_points, 1))
        x = np.random.randint(0, width, (num_points, 1))
        t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
        points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
        return points

    resize_height = image_dims[0]  # @param {type: "integer"}
    resize_width = image_dims[1]  # @param {type: "integer"}
    # num_points = 20  # @param {type: "integer"}

    frames = media.resize_video(video, (resize_height, resize_width))
    # query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)

    if len(query_points.shape) == 1:
        # If query_points is a single point, expand it to a batch of points
        query_points = query_points[None, :]
    # query_points = np.array([[0, 115, 140]])

    tracks, visibles = inference(frames, query_points)
    tracks = np.array(tracks)
    visibles = np.array(visibles)

    # Visualize sparse point tracks
    height, width = video.shape[1:3]
    tracks = transforms.convert_grid_coordinates(
        tracks, (resize_width, resize_height), (width, height)
    )

    if visualize:
        # media.show_video(video_viz, fps=10)
        # save the video with point tracks
        video_viz = viz_utils.paint_point_track(video, tracks, visibles)
        save_dir = os.path.join(image_dir, "point_tracks_video.mp4")
        media.write_video(save_dir, video_viz, fps=10)
        print(f"Saved video with point tracks to {save_dir}")

    return tracks
