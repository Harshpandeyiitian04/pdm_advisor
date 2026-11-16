# This file turns five dashboard screenshots into one short looping animation for the project readme.

# These lines bring in file paths and the image tools.
from pathlib import Path

from PIL import Image, ImageOps

# The project folder, the screenshot names in the order they should play, and the size of each frame.
ROOT = Path(__file__).resolve().parents[1]
SHOTS = [
    "dashboard-fleet.png",
    "dashboard-engine.png",
    "dashboard-models.png",
    "dashboard-cost.png",
    "dashboard-memo.png",
]
SIZE = (960, 540)


# This opens each screenshot, fits it to the same size, and saves them as one animation.
def main():
    frames = []
    for name in SHOTS:
        # Open the picture and make sure it is a normal color image.
        image = Image.open(ROOT / name).convert("RGB")
        # Fit it into the frame size, pad any empty space, and keep the top of the page in view.
        image = ImageOps.pad(image, SIZE, method=Image.Resampling.LANCZOS, color="#0e1117", centering=(0.5, 0.1))
        # Shrink the color count so the animation file stays smaller.
        frames.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    # Save the frames as a looping animation and print where it went.
    output = ROOT / "dashboard-demo.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=1800, loop=0, optimize=True)
    print(f"Wrote {output} ({len(frames)} frames, {SIZE[0]}x{SIZE[1]})")


# When this file is run directly, build the animation.
if __name__ == "__main__":
    main()
