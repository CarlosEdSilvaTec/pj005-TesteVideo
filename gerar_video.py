import subprocess, os, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = r'C:\Progress\temp\IA\pj005-TesteVideo'
AVATAR_PATH = os.path.join(OUTPUT_DIR, 'avatar.png')
AUDIO_PATH = os.path.join(OUTPUT_DIR, 'audio_video.mp3')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'video_final.mp4')
FFMPEG_PATH = os.path.join(OUTPUT_DIR, 'ffmpeg.exe')

W, H = 1280, 720
FPS = 30
TOTAL_SEC = 60

font_l = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 52)
font_m = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 32)
font_s = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 24)

# Load avatar in RGBA, keep at native resolution
avatar_pil = Image.open(AVATAR_PATH).convert('RGBA')
AVATAR_SIZE = 360
avatar_pil = avatar_pil.resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)

# Pre-compute avatar in BGR format for OpenCV compositing
avatar_bgra = cv2.cvtColor(np.array(avatar_pil), cv2.COLOR_RGBA2BGRA)

scenes = [
    {'start': 0, 'end': 10, 'title': 'IA Generativa', 'sub': 'Criação de Conteúdo para Redes Sociais',
     'tag': 'Posts \u00b7 Legendas \u00b7 Imagens \u00b7 Roteiros', 'color': (100, 200, 255)},
    {'start': 10, 'end': 28, 'title': 'Como funciona?', 'sub': 'Um comando gera posts, legendas, imagens e roteiros',
     'tag': 'Tudo adaptado ao tom da sua marca', 'color': (180, 150, 255)},
    {'start': 28, 'end': 45, 'title': 'Benef\u00edcios', 'sub': 'Economia de horas \u00b7 Consist\u00eancia \u00b7 Fim do bloqueio',
     'tag': 'Perfeito para pequenos neg\u00f3cios e criadores', 'color': (80, 220, 160)},
    {'start': 45, 'end': 55, 'title': 'Na pr\u00e1tica', 'sub': 'Um m\u00eas de posts em 15 minutos',
     'tag': 'Basta revisar e publicar', 'color': (255, 200, 60)},
    {'start': 55, 'end': 60, 'title': 'IA n\u00e3o substitui voc\u00ea', 'sub': 'Ela acelera seu trabalho',
     'tag': 'Experimente voc\u00ea tamb\u00e9m!', 'color': (100, 200, 255)},
]

def overlay_bgra(bg, overlay, x, y):
    oh, ow = overlay.shape[:2]
    if y + oh > bg.shape[0]: oh = bg.shape[0] - y
    if x + ow > bg.shape[1]: ow = bg.shape[1] - x
    ov = overlay[:oh, :ow]
    alpha = ov[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+oh, x:x+ow, c] = (bg[y:y+oh, x:x+ow, c].astype(float) * (1 - alpha)
                                  + ov[:, :, c].astype(float) * alpha).astype(np.uint8)

def text_bgra(text, font, color_rgb, alpha=255, shadow=True):
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0] + 8
    th = bbox[3] - bbox[1] + 8
    img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if shadow:
        draw.text((4, 4), text, font=font, fill=(0, 0, 0, 180))
    draw.text((2, 2), text, font=font, fill=(*color_rgb, alpha))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

def render():
    print("Rendering...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, '_raw.mp4'), fourcc, FPS, (W, H))

    # Pre-render static scene backgrounds
    bg_cache = {}
    for scene in scenes:
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        col = np.array([18, 18, 40], dtype=np.uint8)
        bg[:] = col
        for y in range(H):
            fade = 0.12 * math.sin(y / H * math.pi)
            bg[y] = (bg[y].astype(float) * (1 - fade) + np.array([28, 24, 52]) * fade).astype(np.uint8)
        bg_cache[scene['start']] = bg

    # Eye blink mask for avatar - pre-create open and closed eye versions
    avatar_open = avatar_bgra.copy()
    avatar_closed = avatar_bgra.copy()
    eye_center_y = int(AVATAR_SIZE * 0.42)
    cv2.rectangle(avatar_closed, (0, eye_center_y - 4), (AVATAR_SIZE, eye_center_y + 4), (0, 0, 0, 0), -1)
    eyelid_color = (100, 80, 70, 255)
    cv2.line(avatar_closed, (int(AVATAR_SIZE*0.28), eye_center_y),
             (int(AVATAR_SIZE*0.72), eye_center_y), eyelid_color, 3)

    TOTAL_FRAMES = TOTAL_SEC * FPS
    for fi in range(TOTAL_FRAMES):
        t = fi / FPS
        scene = scenes[-1]
        for s in scenes:
            if s['start'] <= t < s['end']:
                scene = s
                break

        bg = bg_cache[scene['start']].copy()
        sp = (t - scene['start']) / (scene['end'] - scene['start'])
        ac = scene['color']

        # Avatar position with visible animation
        sway = math.sin(t * 1.2) * 8
        bob = math.sin(t * 0.8) * 5
        tilt = math.sin(t * 0.6) * 0.02
        ax = 940 + int(sway)
        ay = 50 + int(bob)

        # Blink: eyes close for ~6 frames every 4 seconds
        blink_frame = fi % 120
        if blink_frame < 6:
            av = avatar_closed.copy()
        else:
            av = avatar_open.copy()

        # Apply subtle tilt rotation
        if abs(tilt) > 0.005:
            M = cv2.getRotationMatrix2D((AVATAR_SIZE//2, AVATAR_SIZE//2), math.degrees(tilt), 1.0)
            av = cv2.warpAffine(av, M, (AVATAR_SIZE, AVATAR_SIZE), flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        overlay_bgra(bg, av, ax, ay)

        card_x, card_y = 50, 150

        def fade_alpha(delay, speed=2.0):
            return max(0, min(255, int(255 * min(1.0, max(0.0, (sp - delay) * speed)))))

        a1 = fade_alpha(0.0)
        if a1 > 0:
            t1 = text_bgra(scene['title'], font_l, ac, a1)
            overlay_bgra(bg, t1, card_x, card_y)

        a2 = fade_alpha(0.12)
        if a2 > 0:
            t2 = text_bgra(scene['sub'], font_m, (235, 235, 235), a2)
            overlay_bgra(bg, t2, card_x, card_y + 65)

        a3 = fade_alpha(0.25)
        if a3 > 0:
            t3 = text_bgra(scene['tag'], font_s, (175, 175, 195), a3)
            overlay_bgra(bg, t3, card_x, card_y + 125)

        if scene['start'] == 28 and sp > 0.2:
            a4 = fade_alpha(0.2)
            items = [('\u23f1 Economia', 'de horas'), ('\U0001f4c8 Consist\u00eancia', 'nas publica\u00e7\u00f5es'), ('\U0001f4a1 Fim do', 'bloqueio criativo')]
            for ii, (l1, l2) in enumerate(items):
                ix = card_x + 20 + ii * 210
                t4 = text_bgra(l1, font_s, (255, 255, 255), a4)
                overlay_bgra(bg, t4, ix, card_y + 175)
                t5 = text_bgra(l2, font_s, (180, 180, 200), a4)
                overlay_bgra(bg, t5, ix, card_y + 205)

        if scene['start'] == 45 and sp > 0.15:
            a5 = fade_alpha(0.15)
            days = ['Seg: Dica de uso', 'Ter: Bastidores', 'Qua: Produto destaque', 'Qui: Depoimento', 'Sex: Promo\u00e7\u00e3o']
            for di, day in enumerate(days):
                td = text_bgra(f'\U0001f4c5 {day}', font_s, (200, 200, 210), a5)
                overlay_bgra(bg, td, card_x + 30, card_y + 175 + di * 32)

        pb = int(W * t / TOTAL_SEC)
        cv2.rectangle(bg, (0, H - 5), (pb, H), ac, -1)

        writer.write(bg)
        if fi % 300 == 0:
            print(f"  {fi}/{TOTAL_FRAMES} ({t:.0f}s)")

    writer.release()

    raw = os.path.join(OUTPUT_DIR, '_raw.mp4')
    print("Muxing audio...")
    subprocess.run([
        FFMPEG_PATH, '-i', raw, '-i', AUDIO_PATH,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '20',
        '-c:a', 'aac', '-b:a', '128k', '-shortest',
        '-pix_fmt', 'yuv420p', '-y', OUTPUT_PATH
    ], check=True, capture_output=True)
    os.remove(raw)

    mb = os.path.getsize(OUTPUT_PATH) / (1024*1024)
    print(f"Done: {OUTPUT_PATH} ({mb:.1f} MB)")

if __name__ == '__main__':
    render()
