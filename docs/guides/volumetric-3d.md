# Volumetric 3D on multi-camera rigs (`centered_volume`)

On a calibrated multi-camera rig, sleap-nn's top-down models find each animal in each
camera (centroid model) and its body parts in a crop around it (centered-instance
model). Triangulating those 2D peaks gives 3D. The **volumetric model** replaces that
last step with a small learned 3D network:

1. A cube (96 voxels a side by default) is placed around each animal, centred on a
   triangulated body point.
2. Every voxel is projected into every camera, the centered-instance model's
   confidence maps are read there, and the values are averaged over the cameras that
   see the voxel. Only the calibration is needed.
3. A 3D UNet turns the averaged maps into one 3D probability volume per body part. The
   3D point is the volume's mean; its spread gives an uncertainty (a covariance) for
   each point.

It reads sleap-nn's own 2D maps, not raw pixels, so it rides on the 2D models a lab
already has. It is the DANNCE / learnable-triangulation idea.

Model type / head-config key: **`centered_volume`**, backbone **`unet3d`** (each only
with the other). Sample config:
[`config_centered_volume_unet3d.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_centered_volume_unet3d.yaml).

## When it helps

From the 2026-11 volumetric study (same 2D models for both methods, a DLT that drops
outlier cameras as the baseline, predicted cube centres):

| rig | test data | triangulation | volume |
|---|---|---|---|
| 8 cameras (SLAP) | 5 held-out sessions | 5.59 | 5.03 (-10 %) |
| 5 cameras | 5 later sessions | 3.93 mm | 3.27 mm (-17 %) |
| 8 cameras, strong production 2D | 125 hand-labelled mice, 2D retrained without them | 3.29 mm | 3.05 mm (-7 %) |

Large errors drop more than the median (the volume's answer cannot leave its cube). In
every number above, animals were matched across cameras from the reference, for both
methods.

Limits:

- **One model per rig.** A volume run on another rig's 2D model lost 26-117 % against
  triangulation. Inference refuses a centered-instance model other than the training
  one unless you pass `--allow_other_ci` (meant for a retrained 2D model of the same rig).
- **A GPU.** About 45-50 ms per animal per frame per pass on an A100.
- **Two cameras are unreliable; many cameras leave little to gain** (-5 % with 17).
- **Better 2D training data is the bigger lever.** On one rig, better 2D models cut the
  triangulation error far more than the volume did.

## Training data: recording sessions with instance groups

Training reads `.slp` files with **recording sessions** (sleap-io's `RecordingSession`:
calibrated `Camera`s with OpenCV intrinsics, 5-term distortion and extrinsics, each
linked to its video). Each **instance group** on a **frame group** (one instance per
camera, matched across cameras) is one training animal. LUC3D projects export exactly
this.

Its 3D target (`data_config.multiview.target_source`):

- `auto` (default): the group's `instance_3d` when set (LUC3D's 3D, hand-placed 3D),
  else the robust DLT of its 2D instances (a user instance made from a prediction wins
  over it);
- `instance_3d`: only groups with one;
- `triangulate`: always the robust DLT. A point trains with at least `min_views`
  cameras and a mean reprojection error below `max_reproj_px`.

The robust DLT triangulates the undistorted rays of every camera that sees a point and
drops the worst camera while its reprojection error exceeds 12 px and at least 3
cameras remain.

**Validation** is `val_labels_path` if given, else the last `validation_fraction` of
each session's frames as one time block (never a random frame split: neighbouring
frames would leak). The split is written to `split.json` in the run folder.

## The cube

- **Centre.** The mean of `center_nodes`. It must be where the 2D model centres its
  crops: the default is the 2D model's `anchor_part`. When that anchor has no 3D targets
  (an added midpoint node), name the nodes it is the midpoint of (SLAP:
  `[Neck, TTI]`).
- **Channels** are exactly `part_names`, in that order, each read from the 2D maps by
  name. The default is the 2D model's parts that have 3D targets.
- **Size.** `cube_size` is required, in the calibration's world units. The study used 384
  on SLAP (its calibration unit is about 0.62 mm, so about 236 mm) and 249.6 mm on
  DoublePoke, and compared only 320 and 384 (on SLAP). The cube should hold the whole
  animal around its centre, tail tip included, at any yaw. `cube_size: auto` (twice
  the largest per-node 99th-percentile distance from the centre, written into the saved
  config) is an untested convenience: on the study's data it gives larger cubes (SLAP
  412.8, DoublePoke 268.8).
- **Floor.** The cube's up axis is the floor normal: `multiview.up` if given, else the
  session's metadata (`up` or `floor_normal`), else a RANSAC plane through
  `up_fit_nodes` (floor-touching nodes such as tail tips), else through each
  animal-frame's lowest reliable node. A poor fit (fewer than 60 % inliers, or more than
  3° from the rough fit) warns: set `up` on ramps and uneven floors.

## Training

```bash
sleap-nn train --config-dir docs/sample_configs --config-name config_centered_volume_unet3d
```

The 2D model stays frozen and is referenced by folder and sha256; `best.ckpt` holds only
the 3D network (about 23 MB). Each step (on the GPU, bf16 autocast): augment (cube-centre
jitter, random yaw, camera dropout, per-camera brightness/contrast), crop, 2D maps,
unproject, 3D UNet, loss (L1 on the soft-argmax point + a volumetric cross-entropy to a
Gaussian around the target). The scheduler (`cosine_annealing_warmup`) steps every step:
16 epochs of 500 steps are the study's 8,000. Validation logs the median, mean and 90th
percentile error of reliable points (world units); `best.ckpt` keeps the lowest median.

At the end, `validation_cov_scale` is measured on the validation set with cubes centred
as at inference (a fallback; inference measures it on each run, see below).

`image_decode: full` (default) crops full-resolution frames with sleap-nn's own top-down
code, so the 2D maps are exactly what `sleap-nn predict` would give. `reduced` decodes
JPEG frames at the 2D model's input scale and crops there, as the study did.

## Inference

```bash
sleap-nn predict -i session.slp -m models/centered_volume -o out.slp
```

- The input is an `.slp` with a recording session. Instance groups already there are
  used as they are. For frames without them, add the 2D models:
  `-m models/centroid -m models/centered_instance` (each camera's video goes through
  them, and detections are matched across cameras by the study's rule: every camera
  pair proposes animals, other cameras support them within 60 px at 1280 px image
  width, at least 3 cameras per animal, at most `--max_instances`).
- `--volume_passes auto|1|2`: by default 2 passes (the second re-centred on the first's
  `center_nodes`) when the cube centre is not the 2D model's anchor, else 1.
- Outputs:
    - `out.slp`: a `PredictedInstance3D` on every instance group (point scores are the
      peak voxel probability). A user-placed `Instance3D` is kept.
    - `out.vol3d.h5` (or `--vol3d PATH`; one file per session when there are several):
      the file LUC3D's File ▸ Load Volume 3D reads (`lucid-volume3d/1`: `/frame_indices`,
      `/means`, `/covs`, `/node_names`, attributes `cov_scale`, `source`, optional
      `skeleton_chains` from `--chains FILE.json`).
- `cov_scale` is measured on the run, as LUC3D's exporter does: the median squared
  Mahalanobis distance of (volume - robust DLT) on points seen by at least 4 cameras
  below 3 px, over the chi-square(3) median. With fewer than 20 such points the model's
  `validation_cov_scale` is used; `--cov_scale X` overrides both.

Python API:

```python
import sleap_io as sio
from sleap_nn.inference.volume import predict_volume3d, write_session_volumes

labels = sio.load_slp("session.slp")
pred = predict_volume3d(labels, "models/centered_volume")
pred.labels.save("out.slp")
write_session_volumes(pred, "out.vol3d.h5", source="my run")
```

## Loading the study's checkpoints

`sleap_nn.training.volume.load_study_checkpoint("last.pt")` loads a 2026-11 study
checkpoint into the packaged `UNet3D` unchanged (same module names and skip order).
Its `part_names` and `center_nodes` must be those of its training rig.
