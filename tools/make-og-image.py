#!/usr/bin/env python3
"""Generate 1200x630 OG card for karpathy.github.io — minimal academic aesthetic."""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (255, 255, 255)
INK = (51, 51, 51)          # #333 site header/text color
MUTED = (80, 80, 80)        # #505050 nav color
ACCENT = (42, 122, 226)     # #2a7ae2 link blue
ACCENT_DARK = (32, 92, 170) # #205caa visited
LINE = (232, 232, 232)      # #e8e8e8 borders

FONT_DIR = os.environ.get("FONT_DIR", "/usr/share/fonts/truetype/dejavu")

def font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

f_name = font(84)
f_tag = font(40)
f_sub = font(30)
f_domain = font(34)

# Top rule like the site's header (border-top: 5px solid #333)
d.rectangle([0, 0, W, 8], fill=INK)

# Name block, centered vertically around y=290
d.text((90, 190), "Andrej Karpathy", font=f_name, fill=INK)
# Tagline under name
d.text((90, 310), "Musings of a Computer Scientist.", font=f_tag, fill=MUTED)

# Topic pills from the site's nav/posts: neural nets, LLMs, RL, vision
pills = [["neural nets", "LLMs"], ["reinforcement learning", "computer vision"]]
y = 400
for row in pills:
    x = 90
    for p in row:
        w = d.textlength(p, font=f_sub) + 44
        d.rounded_rectangle([x, y, x + w, y + 56], radius=28, fill=(239, 243, 252), outline=ACCENT)
        d.text((x + 22, y + 11), p, font=f_sub, fill=ACCENT_DARK)
        x += w + 18
    y += 74

# Domain at bottom-right like the jonbarron card convention
d.text((90, 540), "karpathy.github.io", font=f_domain, fill=MUTED)
# RSS icon hint (feed is prominent on the site): orange square + white dot
d.rounded_rectangle([1030, 528, 1110, 608], radius=8, fill=(200, 120, 30))
d.ellipse([1074, 570, 1098, 594], fill=(255, 255, 255))
d.rounded_rectangle([1052, 548, 1068, 564], radius=3, fill=(255, 255, 255))
d.rectangle([1052, 548, 1068, 564], fill=(255, 255, 255))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "og-image.png")
img.save(out, "PNG")
print("saved", out, img.size)