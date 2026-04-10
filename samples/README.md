# Sample Data

Copy raw frame folders from existing popcon jobs:

```bash
# From a completed popcon job:
cp -r /tmp/popcon/jobs/<JOB_ID>/frames/<EMOJI_NAME>/raw/ samples/emoji_01/

# Example:
cp -r /tmp/popcon/jobs/abc123/frames/00_wave/raw/ samples/wave/
cp -r /tmp/popcon/jobs/abc123/frames/01_laugh/raw/ samples/laugh/
```

Each folder should contain 5-20 PNG frames named `frame_001.png`, `frame_002.png`, etc.
These are the raw white-background frames extracted from the animated video.

Aim for 3-5 emoji sets that cover different scenarios:
- Simple motion (wave, nod)
- Fast motion (jump, spin) — tests for motion blur handling
- Fine details (hair, thin lines) — tests for detail preservation
