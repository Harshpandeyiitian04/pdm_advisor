"""Build the README demo GIF from dashboard screenshots in the project root."""
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SHOTS = [
    "dashboard-fleet.png",
    "dashboard-engine.png",
    "dashboard-models.png",
    "dashboard-cost.png",
    "dashboard-memo.png",
]
SIZE = (960, 540)


def main():
    frames = []
    for name in SHOTS:
        image = Image.open(ROOT / name).convert("RGB")
        image = ImageOps.pad(image, SIZE, method=Image.Resampling.LANCZOS, color="#0e1117", centering=(0.5, 0.1))
        frames.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    output = ROOT / "dashboard-demo.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=1800, loop=0, optimize=True)
    print(f"Wrote {output} ({len(frames)} frames, {SIZE[0]}x{SIZE[1]})")


if __name__ == "__main__":
    main()