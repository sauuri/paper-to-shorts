import os
import requests
import numpy as np
from moviepy import AudioFileClip, VideoFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
from app.config import settings

WIDTH, HEIGHT = 1080, 1920


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
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


FALLBACK_KEYWORDS = ["aerial drone", "city timelapse", "dark sky", "technology screen", "fire explosion"]


def _fetch_pexels_video(keyword: str, output_path: str) -> bool:
    headers = {"Authorization": settings.pexels_api_key}
    keywords_to_try = [keyword] + FALLBACK_KEYWORDS

    for kw in keywords_to_try:
        try:
            res = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers,
                params={"query": kw, "per_page": 10},
                timeout=10,
            )
            if res.status_code != 200:
                continue
            videos = res.json().get("videos", [])
            if not videos:
                continue

            import random
            video = random.choice(videos[:5])
            video_files = video.get("video_files", [])
            hd_files = [f for f in video_files if f.get("quality") in ("hd", "sd")]
            if not hd_files:
                continue

            url = hd_files[0]["link"]
            r = requests.get(url, stream=True, timeout=60)
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

        except Exception:
            continue

    return False


def _make_title_overlay(title: str, hook: str, duration: float) -> ImageClip:
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 하단 그라데이션 오버레이
    for y in range(HEIGHT // 2, HEIGHT):
        alpha = int(220 * (y - HEIGHT // 2) / (HEIGHT // 2))
        draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, alpha))

    # 상단 얇은 그라데이션
    for y in range(0, 200):
        alpha = int(160 * (1 - y / 200))
        draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, alpha))

    # 좌측 강조 바
    draw.rectangle([(60, HEIGHT - 520), (68, HEIGHT - 200)], fill=(255, 50, 50, 255))

    # BREAKING 태그
    tag_font = _get_font(36, bold=True)
    draw.rectangle([(80, HEIGHT - 530), (280, HEIGHT - 490)], fill=(255, 50, 50, 255))
    draw.text((90, HEIGHT - 528), "BREAKING", font=tag_font, fill=(255, 255, 255, 255))

    # 제목
    title_font = _get_font(72, bold=True)
    lines = _wrap_text(draw, title, WIDTH - 160, title_font)
    y = HEIGHT - 480
    for line in lines[:3]:
        draw.text((80, y), line, font=title_font, fill=(255, 255, 255, 255))
        bbox = draw.textbbox((0, 0), line, font=title_font)
        y += (bbox[3] - bbox[1]) + 8

    # 후크 문장
    hook_font = _get_font(46)
    hook_lines = _wrap_text(draw, hook, WIDTH - 160, hook_font)
    y += 20
    for line in hook_lines[:2]:
        draw.text((80, y), line, font=hook_font, fill=(220, 220, 220, 255))
        bbox = draw.textbbox((0, 0), line, font=hook_font)
        y += (bbox[3] - bbox[1]) + 6

    return ImageClip(np.array(img)).with_duration(duration)


def _make_point_overlay(index: int, total: int, point: str, duration: float) -> ImageClip:
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 하단 그라데이션
    for y in range(HEIGHT // 2, HEIGHT):
        alpha = int(230 * (y - HEIGHT // 2) / (HEIGHT // 2))
        draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, alpha))

    # 상단 그라데이션
    for y in range(0, 160):
        alpha = int(140 * (1 - y / 160))
        draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, alpha))

    # 상단 진행 바
    bar_width = (WIDTH - 120) // total
    for i in range(total):
        x = 60 + i * (bar_width + 6)
        color = (255, 50, 50, 255) if i <= index else (100, 100, 100, 180)
        draw.rectangle([(x, 60), (x + bar_width, 72)], fill=color)

    # 번호 뱃지
    num_font = _get_font(44, bold=True)
    badge_text = f"{index + 1:02d}"
    draw.rectangle([(60, HEIGHT - 560), (160, HEIGHT - 490)], fill=(255, 50, 50, 255))
    bbox = draw.textbbox((0, 0), badge_text, font=num_font)
    tx = 60 + (100 - (bbox[2] - bbox[0])) // 2
    draw.text((tx, HEIGHT - 555), badge_text, font=num_font, fill=(255, 255, 255, 255))

    # 포인트 텍스트
    point_font = _get_font(64, bold=True)
    lines = _wrap_text(draw, point, WIDTH - 160, point_font)
    y = HEIGHT - 470
    for line in lines[:3]:
        # 텍스트 그림자
        draw.text((82, y + 2), line, font=point_font, fill=(0, 0, 0, 180))
        draw.text((80, y), line, font=point_font, fill=(255, 255, 255, 255))
        bbox = draw.textbbox((0, 0), line, font=point_font)
        y += (bbox[3] - bbox[1]) + 10

    return ImageClip(np.array(img)).with_duration(duration)


def _make_bg_clip(video_path: str, duration: float) -> VideoFileClip:
    clip = VideoFileClip(video_path)

    clip_ratio = clip.w / clip.h
    target_ratio = WIDTH / HEIGHT

    if clip_ratio > target_ratio:
        clip = clip.resized(height=HEIGHT)
        clip = clip.cropped(x_center=clip.w / 2, y_center=HEIGHT / 2, width=WIDTH, height=HEIGHT)
    else:
        clip = clip.resized(width=WIDTH)
        clip = clip.cropped(x_center=WIDTH / 2, y_center=clip.h / 2, width=WIDTH, height=HEIGHT)

    if clip.duration < duration:
        loops = int(duration / clip.duration) + 1
        clip = concatenate_videoclips([clip] * loops)

    return clip.with_subclip(0, duration)


def _make_fallback_bg(duration: float) -> ImageClip:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = (10, 10, 20)
    return ImageClip(img).with_duration(duration)


def create_video(script: dict, audio_path: str, output_path: str) -> str:
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration

    points = script["points"]
    title_duration = 3.5
    point_duration = (total_duration - title_duration) / len(points)

    keywords = script.get("keywords", ["news breaking", "world event", "technology", "science", "future"])
    while len(keywords) < len(points) + 1:
        keywords.append(keywords[-1])

    segments = [
        (title_duration, "title", keywords[0]),
        *[(point_duration, i, keywords[i + 1]) for i in range(len(points))]
    ]

    clips = []

    for seg_idx, (seg_duration, seg_type, keyword) in enumerate(segments):
        bg_path = f"{settings.output_dir}/bg_{seg_idx}.mp4"
        has_bg = _fetch_pexels_video(keyword, bg_path)

        if has_bg:
            bg = _make_bg_clip(bg_path, seg_duration)
        else:
            bg = _make_fallback_bg(seg_duration)

        if seg_type == "title":
            overlay = _make_title_overlay(script["title"], script["hook"], seg_duration)
        else:
            overlay = _make_point_overlay(seg_type, len(points), points[seg_type], seg_duration)

        segment = CompositeVideoClip([bg, overlay])
        clips.append(segment)

    video = concatenate_videoclips(clips)
    video = video.with_audio(audio)
    video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", logger=None)

    for i in range(len(segments)):
        bg_path = f"{settings.output_dir}/bg_{i}.mp4"
        if os.path.exists(bg_path):
            os.remove(bg_path)

    return output_path
