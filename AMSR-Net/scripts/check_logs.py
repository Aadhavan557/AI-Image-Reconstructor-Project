import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "logs"
runs = sorted(os.listdir(log_dir))
if not runs:
    print("No runs found")
    exit()

latest_run = runs[-1]
ea = EventAccumulator(os.path.join(log_dir, latest_run))
ea.Reload()

print(f"Run: {latest_run}")
for tag in ['Loss/Train/Total', 'Loss/Train/Charb', 'Loss/Train/SSIM', 'Loss/Train/Edge', 'Train/GradNorm', 'Metrics/Val/PSNR_dB']:
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        print(f"{tag}:")
        for e in events[-5:]:
            print(f"  Step {e.step}: {e.value:.4f}")
    else:
        print(f"{tag} not found")
