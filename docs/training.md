# Training

## Which Models Need Training

| Model | Pretrained? | Needs Fine-Tuning? | Priority |
|-------|------------|-------------------|----------|
| YOLO11 (players) | Yes (COCO) | No | — |
| YOLO11 (ball) | Yes (COCO) | **Recommended** | High |
| MobileNetV3 (rally) | Yes (ImageNet) | **Recommended** (heuristic fallback always works) | High |
| R3D-18 (actions) | Partially (Kinetics-400) | **Required** if you want hit type classification | Medium |

The pipeline works without any fine-tuning — hits are detected rule-based from ball trajectory, rally state is classified by heuristic. Model training improves quality.

## 1. Ball Detection Fine-Tuning

### Option A: Fine-Tune YOLO

```bash
# Dataset structure (YOLO format):
# dataset/
#   train/images/  train/labels/
#   val/images/    val/labels/
# Label format: class_id center_x center_y width height (normalized)
```

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=1280, batch=16, name="padel_ball")
```

Update config:
```yaml
ball:
  detection:
    model: "runs/detect/padel_ball/weights/best.pt"
    sports_ball_class_id: 0
```

**Tips**: Label the ball even when blurred. Include hard negatives. Aim for 2000+ frames.

### Option B: Switch to TrackNet

[TrackNet v2](https://nol.cs.nctu.edu.tw/nien20/TrackNet/) takes 3 consecutive frames, outputs a heatmap. Detection rate 85%+ vs 30-60% for generic YOLO. Requires custom integration into `src/ball/`.

## 2. Action Recognition Training

### Action Classes

| Class | Description |
|-------|-------------|
| `serve` | Serve motion |
| `return` | Return of serve |
| `volley` | Net volley |
| `groundstroke` | Forehand/backhand from baseline |
| `lob` | High defensive lob |
| `overhead` | Generic overhead |
| `smash` | Power overhead smash |
| `wall_play` | Shot off wall/glass |

### Data Collection

1. Run the pipeline without action recognition to get player tracking:
```bash
python scripts/run_pipeline.py video.mp4 \
  --floor-points '...' --wall-top-points '...' \
  --no-render -j tracking.json
```

2. Extract player crops from tracking output:
```python
import cv2, json

with open("tracking.json") as f:
    data = json.load(f)

cap = cv2.VideoCapture("video.mp4")
for frame_data in data["player_tracking"]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data["frame_idx"])
    ret, frame = cap.read()
    for player in frame_data["players"]:
        x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(
            f"crops/t{player['track_id']}_f{frame_data['frame_idx']}.jpg",
            crop,
        )
```

3. Group into 16-frame sequences per player, label each with action class.

### Dataset Structure

```
data/actions/
  train/
    serve/        # directories of 16-frame clip sequences
    return/
    volley/
    groundstroke/
    lob/
    overhead/
    smash/
    wall_play/
  val/
    ...same...
```

### Training Script

```python
import torch
from src.players.action_recognition import ActionRecognitionModel, ACTION_CLASSES

model = ActionRecognitionModel(num_classes=len(ACTION_CLASSES), pretrained_backbone=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for clips, labels in train_loader:
        # clips: (B, 3, 16, 112, 112)
        logits = model(clips.cuda())
        loss = criterion(logits, labels.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for clips, labels in val_loader:
            preds = model(clips.cuda()).argmax(dim=1)
            correct += (preds == labels.cuda()).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch}: val_acc = {correct/total:.3f}")

torch.save(model.state_dict(), "action_model.pt")
```

Then run with:
```bash
python scripts/run_pipeline.py video.mp4 --action-model action_model.pt ...
```

### Tips

- **Minimum data**: ~100 clips/class for reasonable accuracy, 500+ for good
- **Class imbalance**: `groundstroke` will dominate — use weighted loss or undersample
- **Freeze first**: Train only classifier head for 5-10 epochs (freeze backbone), then unfreeze
- **Augmentation**: random horizontal flip, color jitter, random crop within bbox

## 3. Rally State Classifier Training

### Classes

| Class | Label | Description |
|-------|-------|-------------|
| `rally_active` | 1 | Ball in play, players actively engaged |
| `rally_inactive` | 0 | Between points: walking, resetting, celebrating |

### Data Collection

1. Run the pipeline to get frame timestamps:
```bash
python scripts/run_pipeline.py video.mp4 \
  --floor-points '...' --wall-top-points '...' \
  --no-render -j tracking.json
```

2. Extract frames at regular intervals:
```python
import cv2

cap = cv2.VideoCapture("video.mp4")
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 15 == 0:  # sample every 15 frames
        cv2.imwrite(f"rally_frames/{frame_idx:06d}.jpg", frame)
    frame_idx += 1
```

3. Label each frame as `rally_active` (1) or `rally_inactive` (0) in a CSV:
```csv
frame,label
000000.jpg,0
000015.jpg,1
000030.jpg,1
...
```

### Dataset Structure

```
data/rally/
  train/
    active/      # frames where rally is in progress
    inactive/    # frames between points
  val/
    active/
    inactive/
```

### Training Script

```python
import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = ImageFolder("data/rally/train", transform=transform)
val_ds = ImageFolder("data/rally/val", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
in_features = model.classifier[0].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features, 128),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(128, 1),
)
model = model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(30):
    model.train()
    for images, labels in train_loader:
        logits = model(images.cuda()).squeeze()
        loss = criterion(logits, labels.float().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            preds = (torch.sigmoid(model(images.cuda()).squeeze()) > 0.5).long()
            correct += (preds == labels.cuda()).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch}: val_acc = {correct/total:.3f}")

torch.save(model.state_dict(), "rally_model.pt")
```

Then run with:
```bash
python scripts/run_pipeline.py video.mp4 --rally-model rally_model.pt ...
```

### Tips

- **Minimum data**: ~200 frames/class for reasonable accuracy, 1000+ for good
- **Class balance**: Rallies typically occupy 40-60% of match time — roughly balanced
- **Freeze first**: Train classifier head for 5 epochs (freeze backbone), then unfreeze
- **Augmentation**: horizontal flip, color jitter, random crop
- **Without training**: The heuristic (player speed) works as a reasonable baseline

## 4. Parameter Tuning (No Training Needed)

### Kalman Filter

| Parameter | Effect |
|-----------|--------|
| `process_noise` | Higher → trusts measurements more, less smoothing |
| `measurement_noise` | Higher → trusts model more, more smoothing |

### Event Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `velocity_change_threshold` | 15.0 | Min acceleration to detect bounce/hit |
| `racket_hit_distance` | 40.0 px | Max ball-to-racket center distance for hit |
| `player_hit_distance` | 100.0 px | Max ball-to-player center distance for hit (fallback) |
| `net_zone_width` | 0.5 m | Net detection zone width |
| `net_speed_drop_ratio` | 0.5 | Speed drop ratio for net hit |

These depend on resolution, frame rate, camera angle. Adjust based on false positives/negatives.

### Trajectory Analysis

| Parameter | Default | Description |
|-----------|---------|-------------|
| `post_bounce_window` | 12 | Frames to analyze after the velocity change |
| `pre_bounce_window` | 10 | Frames to analyze before the velocity change |
| `court_y_displacement_threshold` | 1.5 m | Court-Y displacement above which a bounce is classified as wall |
| `curvature_threshold` | 0.01 | Image-space curvature coefficient above which → floor bounce |
| `weights.rules` | 0.35 | Weight for padel rule-based prior signal |
| `weights.court_y` | 0.25 | Weight for post-bounce court-Y displacement signal |
| `weights.curvature` | 0.20 | Weight for image-space curvature signal |
| `weights.rally` | 0.20 | Weight for rally state continuity signal |

### Rally Classifier Heuristic

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speed_threshold` | 3.0 px/frame | Player speed below which they are "stationary" |
| `active_ratio` | 0.5 | Fraction of players that must be moving for rally to be active |
| `window_frames` | 30 | Sliding window for speed averaging (~1 second at 30fps) |
