# Portrait animation

[Download/watch the 5-second portrait animation](portrait_animation.mp4).

The animation uses the uploaded photograph **as-is**, as requested in the clarification: original hair, mustache, white shirt, and grey backdrop. It starts neutral, gradually adds a restrained closed-lip smile, and blinks twice. The head and camera stay still.

- Duration: 5 seconds; 30 fps; H.264 MP4; 2160 × 2700 pixels (original 4:5 framing).
- **Resolution note:** the supplied reference is only **160 × 200 pixels**. The output is enlarged to a 4K-sized frame but cannot have genuine 4K facial detail.
- The small `blink_keyframe.png` changes only the eyelid areas; the rest of each frame is based on the original photograph.

To reproduce:

```bash
python -m pip install numpy opencv-python-headless imageio-ffmpeg
python animate_portrait.py reference_photo.jpg portrait_animation.mp4
```
