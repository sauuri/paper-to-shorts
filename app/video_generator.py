import os
import base64
import numpy as np
from openai import OpenAI
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from app.config import settings

WIDTH, HEIGHT = 1080, 1920
ACCENT = (99, 102, 241, 255)      # indigo
ACCENT_DIM = (60, 63, 180, 200)
WHITE = (255, 255, 255, 255)
SHADOW = (0, 0, 0, 170)


def _get_font(size: int, bold: bool = False):
    font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=1 if bold else 0)
            except Exception:
                return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap_text(draw, text, max_width, font):
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_text_shadow(draw, pos, text, font, fill, shadow_offset=3):
    draw.text((pos[0] + shadow_offset, pos[1] + shadow_offset), text, font=font, fill=SHADOW)
    draw.text(pos, text, font=font, fill=fill)


def _apply_gradient(img: Image.Image, top_alpha: int, bottom_alpha: int) -> Image.Image:
    grad = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(grad)
    for y in range(360):
        a = int(top_alpha * (1 - y / 360))
        d.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, a))
    for y in range(HEIGHT // 2, HEIGHT):
        a = int(bottom_alpha * (y - HEIGHT // 2) / (HEIGHT // 2))
        d.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, a))
    return Image.alpha_composite(img.convert("RGBA"), grad)


def _generate_scene_image(prompt: str, output_path: str) -> bool:
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        full_prompt = (
            f"Cinematic vertical background for YouTube Shorts. Theme: {prompt}. "
            "Style: dramatic moody lighting, dark atmosphere, high contrast, "
            "photorealistic, no text, no faces, deep depth of field, "
            "professional cinematography, dark color grading"
        )
        response = client.images.generate(
            model="gpt-image-1",
            prompt=full_prompt,
            size="1024x1536",
            quality="medium",
            n=1,
        )
        data = base64.b64decode(response.data[0].b64_json)
        with open(output_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"[image-gen] failed: {e}")
        return False


def _make_bg_from_image(img_path: str, duration: float) -> ImageClip:
    img = Image.open(img_path).convert("RGB")
    ratio = HEIGHT / img.height
    new_w = int(img.width * ratio)
    img = img.resize((max(new_w, WIDTH), HEIGHT), Image.LANCZOS)
    if img.width > WIDTH:
        left = (img.width - WIDTH) // 2
        img = img.crop((left, 0, left + WIDTH, HEIGHT))
    else:
        padded = Image.new("RGB", (WIDTH, HEIGHT), (8, 8, 18))
        padded.paste(img, ((WIDTH - img.width) // 2, 0))
        img = padded
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return ImageClip(np.array(img)).with_duration(duration)


def _make_fallback_bg(duration: float) -> ImageClip:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (8, 8, 18)
    return ImageClip(img).with_duration(duration)


def _make_title_overlay(title: str, hook: str, duration: float) -> ImageClip:
    base = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    base = _apply_gradient(base, top_alpha=160, bottom_alpha=240)
    draw = ImageDraw.Draw(base)

    # 좌측 강조선
    draw.rectangle([(60, HEIGHT - 590), (68, HEIGHT - 180)], fill=ACCENT)

    # AI SUMMARY 레이블
    lf = _get_font(30, bold=True)
    draw.rounded_rectangle([(80, HEIGHT - 600), (262, HEIGHT - 566)], radius=6, fill=ACCENT)
    draw.text((96, HEIGHT - 598), "AI SUMMARY", font=lf, fill=WHITE)

    # 제목
    tf = _get_font(74, bold=True)
    lines = _wrap_text(draw, title, WIDTH - 160, tf)
    y = HEIGHT - 545
    for line in lines[:3]:
        _draw_text_shadow(draw, (80, y), line, tf, WHITE)
        y += draw.textbbox((0, 0), line, font=tf)[3] + 10

    # 후크
    hf = _get_font(46)
    y += 18
    for line in _wrap_text(draw, hook, WIDTH - 160, hf)[:2]:
        _draw_text_shadow(draw, (80, y), line, hf, (210, 210, 255, 255))
        y += draw.textbbox((0, 0), line, font=hf)[3] + 8

    # 하단 구분선
    draw.line([(60, HEIGHT - 155), (WIDTH - 60, HEIGHT - 155)], fill=(255, 255, 255, 40), width=1)
    sf = _get_font(32)
    draw.text((60, HEIGHT - 140), "▶  영상 계속 보기", font=sf, fill=(180, 180, 220, 200))

    return ImageClip(np.array(base)).with_duration(duration)


def _make_point_overlay(index: int, total: int, point: str, duration: float) -> ImageClip:
    base = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    base = _apply_gradient(base, top_alpha=140, bottom_alpha=235)
    draw = ImageDraw.Draw(base)

    # 진행 바 (상단)
    bar_w = (WIDTH - 120) // total
    for i in range(total):
        x = 60 + i * (bar_w + 5)
        color = ACCENT if i <= index else (70, 70, 90, 180)
        draw.rounded_rectangle([(x, 58), (x + bar_w, 70)], radius=4, fill=color)

    # 원형 번호 뱃지
    nf = _get_font(50, bold=True)
    bx, by = 60, HEIGHT - 580
    draw.ellipse([(bx, by), (bx + 96, by + 96)], fill=ACCENT)
    nt = str(index + 1)
    nb = draw.textbbox((0, 0), nt, font=nf)
    draw.text(
        (bx + (96 - (nb[2] - nb[0])) // 2, by + (96 - (nb[3] - nb[1])) // 2),
        nt, font=nf, fill=WHITE
    )

    # 포인트 텍스트
    pf = _get_font(66, bold=True)
    y = HEIGHT - 462
    for line in _wrap_text(draw, point, WIDTH - 170, pf)[:3]:
        _draw_text_shadow(draw, (80, y), line, pf, WHITE)
        y += draw.textbbox((0, 0), line, font=pf)[3] + 10

    # 진행 표시 텍스트
    cf = _get_font(32)
    draw.text((WIDTH - 160, HEIGHT - 100), f"{index + 1} / {total}", font=cf, fill=(180, 180, 200, 200))

    return ImageClip(np.array(base)).with_duration(duration)


def _make_outro_overlay(title: str, duration: float) -> ImageClip:
    base = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    base = _apply_gradient(base, top_alpha=200, bottom_alpha=250)
    draw = ImageDraw.Draw(base)

    # 중앙 CTA
    cx = WIDTH // 2
    tf = _get_font(56, bold=True)
    for i, line in enumerate(["구독과 좋아요", "잊지 마세요 👍"]):
        bb = draw.textbbox((0, 0), line, font=tf)
        _draw_text_shadow(draw, (cx - (bb[2] - bb[0]) // 2, HEIGHT // 2 - 80 + i * 80), line, tf, WHITE)

    # 제목 recap
    rf = _get_font(36)
    recap = f"📌 {title}"
    rb = draw.textbbox((0, 0), recap, font=rf)
    draw.text((cx - (rb[2] - rb[0]) // 2, HEIGHT // 2 + 200), recap, font=rf, fill=(200, 200, 255, 220))

    return ImageClip(np.array(base)).with_duration(duration)


def create_video(
    script: dict,
    audio_path: str,
    output_path: str,
    bg_images: list[str] | None = None,
) -> str:
    """
    bg_images: user-supplied image paths (used as backgrounds instead of AI generation).
               If None, gpt-image-1 generates backgrounds per scene.
    """
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration

    points = script["points"]
    title_dur = 3.5
    outro_dur = 2.5
    point_dur = (total_duration - title_dur - outro_dur) / len(points)

    # Segment list: (duration, type, index_or_label)
    segments = (
        [(title_dur, "title", None)]
        + [(point_dur, "point", i) for i in range(len(points))]
        + [(outro_dur, "outro", None)]
    )

    # Build backgrounds
    scene_prompts = [script["title"]] + points + [script["title"]]
    generated_paths = []

    clips = []
    for seg_idx, (seg_dur, seg_type, seg_idx_val) in enumerate(segments):
        # --- Background ---
        if bg_images:
            img_src = bg_images[seg_idx % len(bg_images)]
            bg = _make_bg_from_image(img_src, seg_dur)
        else:
            gen_path = f"{settings.output_dir}/scene_{seg_idx}.png"
            ok = _generate_scene_image(scene_prompts[seg_idx], gen_path)
            if ok:
                generated_paths.append(gen_path)
                bg = _make_bg_from_image(gen_path, seg_dur)
            else:
                bg = _make_fallback_bg(seg_dur)

        # --- Overlay ---
        if seg_type == "title":
            overlay = _make_title_overlay(script["title"], script["hook"], seg_dur)
        elif seg_type == "point":
            overlay = _make_point_overlay(seg_idx_val, len(points), points[seg_idx_val], seg_dur)
        else:
            overlay = _make_outro_overlay(script["title"], seg_dur)

        clips.append(CompositeVideoClip([bg, overlay]))

    video = concatenate_videoclips(clips).with_audio(audio)
    video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", logger=None)

    for p in generated_paths:
        if os.path.exists(p):
            os.remove(p)

    return output_path
