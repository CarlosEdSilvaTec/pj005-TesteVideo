import subprocess, os, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = r'C:\Progress\temp\IA\pj005-TesteVideo'
AVATAR_PATH = os.path.join(OUTPUT_DIR, 'avatar.png')
AUDIO_PATH = os.path.join(OUTPUT_DIR, 'audio_video.mp3')
AMPLITUDES_PATH = os.path.join(OUTPUT_DIR, 'amplitudes.npy')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'video_final.mp4')
FFMPEG_PATH = os.path.join(OUTPUT_DIR, 'ffmpeg.exe')

W, H = 1280, 720
FPS = 30
TOTAL_SEC = 60

font_l = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 52)
font_m = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 32)
font_s = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 24)

print("Inicializando...")

# Load and prepare avatar
avatar_full = cv2.imread(AVATAR_PATH)
AVATAR_SIZE = 600
avatar_base = cv2.resize(avatar_full, (AVATAR_SIZE, AVATAR_SIZE), interpolation=cv2.INTER_LANCZOS4)

# Face and mouth detection
gray = cv2.cvtColor(avatar_base, cv2.COLOR_BGR2GRAY)
fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
sc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
faces = fc.detectMultiScale(gray, 1.1, 4)
fx, fy, fw, fh = faces[0]
smiles = sc.detectMultiScale(gray[fy:fy+fh, fx:fx+fw], 1.8, 20)
_mx, _my, mw, mh = smiles[0]
mx = fx + _mx; my = fy + _my
# Expand
p = int(mh * 0.2)
mx = max(0, mx - p)
my = max(0, my - p)
mw = min(AVATAR_SIZE - mx, mw + 2 * p)
mh = min(AVATAR_SIZE - my, mh + 2 * p)
cx, cy = mx + mw // 2, my + mh // 2

print(f"Boca detectada: ({mx},{my},{mw},{mh}) centro=({cx},{cy})")

# Sample colors
lip_sample = avatar_base[my:my + mh, mx:mx + mw]
lip_color = tuple(map(int, cv2.mean(lip_sample)[:3]))
skin_sample = avatar_base[max(0, my - 10):my, mx:mx + mw]
skin_color = tuple(map(int, cv2.mean(skin_sample)[:3]))
dark_color = (10, 5, 3)
teeth_color = (235, 225, 215)

# Round face mask for compositing
face_mask = np.zeros((AVATAR_SIZE, AVATAR_SIZE), dtype=np.uint8)
cv2.circle(face_mask, (AVATAR_SIZE // 2, AVATAR_SIZE // 2), AVATAR_SIZE // 2 - 10, 255, -1)
face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)


def mouth_sync(img, openness):
    r = img[my:my + mh, mx:mx + mw].copy()
    hh, ww = r.shape[:2]
    op = min(1.0, openness * 1.2)

    # Full mouth area overlay with distinct colors
    cv2.ellipse(r, (ww // 2, hh // 2), (ww // 2 - 1, hh // 2 - 1), 0, 0, 360, skin_color, -1)

    if op > 0.05:
        cav_h = int(hh * 0.55 * op)
        cav_w = int(ww * 0.75 * op)

        # Dark oral cavity
        cv2.ellipse(r, (ww // 2, hh // 2), (max(4, cav_w // 2), max(4, cav_h // 2)), 0, 0, 360, (8, 4, 2), -1)

        # Teeth
        if op > 0.12:
            t_h = max(3, int(cav_h * 0.2))
            cy = hh // 2
            cw2 = cav_w // 2
            ch2 = cav_h // 2
            cv2.rectangle(r, (ww // 2 - cw2 + 4, cy - ch2 + 2),
                          (ww // 2 + cw2 - 4, cy - ch2 + 2 + t_h), (245, 235, 225), -1)
            cv2.rectangle(r, (ww // 2 - cw2 + 4, cy + ch2 - 2 - t_h),
                          (ww // 2 + cw2 - 4, cy + ch2 - 2), (245, 235, 225), -1)

    # Lips
    lip_bright = tuple(min(255, c + 50) for c in lip_color)
    lip_dark = tuple(max(0, c - 30) for c in lip_color)
    uv = max(2, int(ww * 0.12))
    lv = max(2, int(ww * 0.10))

    # Upper lip
    cv2.ellipse(r, (ww // 2, 0), (ww // 2 - 1, uv), 0, 0, 360, lip_bright, -1)
    cv2.ellipse(r, (ww // 2, 1), (ww // 2 - 3, uv - 1), 0, 180, 360, lip_dark, 2)

    # Lower lip
    cv2.ellipse(r, (ww // 2, hh), (ww // 2 - 1, lv), 0, 0, 360, lip_bright, -1)
    cv2.ellipse(r, (ww // 2, hh - 1), (ww // 2 - 3, lv - 1), 0, 0, 180, lip_dark, 2)

    if op > 0.05:
        # Open: show separation
        cl = int(ww * 0.45)
        y_line = hh // 2 + int(cav_h * 0.05)
        cv2.line(r, (ww // 2 - cl, y_line), (ww // 2 + cl, y_line), lip_dark, 2)
    else:
        # Closed: single lip line
        cv2.line(r, (2, hh // 2), (ww - 2, hh // 2), lip_dark, 3)

    # Soft blend edges
    msk = np.zeros((hh, ww), dtype=np.uint8)
    cv2.ellipse(msk, (ww // 2, hh // 2), (ww // 2, hh // 2), 0, 0, 360, 255, -1)
    msk = cv2.GaussianBlur(msk, (5, 5), 0)

    roi = img[my:my + mh, mx:mx + mw]
    for c in range(3):
        roi[:, :, c] = (roi[:, :, c].astype(float) * (1 - msk.astype(float) / 255)
                         + r[:, :, c].astype(float) * (msk.astype(float) / 255)).astype(np.uint8)
    return img


# Load amplitudes
amplitudes = np.load(AMPLITUDES_PATH)

scenes = [
    {'start': 0, 'end': 10, 'title': 'IA Generativa', 'sub': 'Cria\u00e7\u00e3o de Conte\u00fado para Redes Sociais',
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

def text_bgra(text, font, color_rgb, alpha=255):
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0] + 12
    th = bbox[3] - bbox[1] + 12
    img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), text, font=font, fill=(*color_rgb, alpha))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

def overlay_bgra(bg, overlay, x, y):
    oh, ow = overlay.shape[:2]
    if y + oh > bg.shape[0]: oh = bg.shape[0] - y
    if x + ow > bg.shape[1]: ow = bg.shape[1] - x
    ov = overlay[:oh, :ow]
    alpha = ov[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+oh, x:x+ow, c] = (bg[y:y+oh, x:x+ow, c].astype(float) * (1 - alpha)
                                  + ov[:, :, c].astype(float) * alpha).astype(np.uint8)


def render():
    print("Renderizando...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, '_raw.mp4'), fourcc, FPS, (W, H))

    # Pre-render backgrounds
    bg_cache = {}
    for s in scenes:
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg[:] = (18, 18, 40)
        for y in range(H):
            fade = 0.12 * math.sin(y / H * math.pi)
            bg[y] = (bg[y].astype(float) * (1 - fade) + np.array([28, 24, 52]) * fade).astype(np.uint8)
        bg_cache[s['start']] = bg

    TOTAL = TOTAL_SEC * FPS
    for fi in range(TOTAL):
        t = fi / FPS
        scene = scenes[-1]
        for s in scenes:
            if s['start'] <= t < s['end']:
                scene = s
                break

        bg = bg_cache[scene['start']].copy()
        sp = (t - scene['start']) / (scene['end'] - scene['start'])
        ac = scene['color']

        # Audio amplitude -> mouth openness
        idx = min(fi, len(amplitudes) - 1)
        openness = amplitudes[idx] ** 0.5

        # Generate avatar frame with lip sync
        av = avatar_base.copy()
        av = mouth_sync(av, openness)

        # To BGRA for compositing
        av_bgra = cv2.cvtColor(av, cv2.COLOR_BGR2BGRA)
        av_bgra[:, :, 3] = face_mask

        # Position - larger avatar, more centered
        ax = 820 + int(math.sin(t * 1.2) * 6)
        ay = 10 + int(math.sin(t * 0.8) * 4)
        overlay_bgra(bg, av_bgra, ax, ay)

        # Text
        cx, cy = 50, 150

        def fa(delay, speed=2.0):
            return max(0, min(255, int(255 * min(1.0, max(0.0, (sp - delay) * speed)))))

        a1 = fa(0.0)
        if a1 > 0:
            overlay_bgra(bg, text_bgra(scene['title'], font_l, ac, a1), cx, cy)
        a2 = fa(0.12)
        if a2 > 0:
            overlay_bgra(bg, text_bgra(scene['sub'], font_m, (235, 235, 235), a2), cx, cy + 65)
        a3 = fa(0.25)
        if a3 > 0:
            overlay_bgra(bg, text_bgra(scene['tag'], font_s, (175, 175, 195), a3), cx, cy + 125)

        if scene['start'] == 28 and sp > 0.2:
            a4 = fa(0.2)
            items = [('\u23f1 Economia', 'de horas'), ('\U0001f4c8 Consist\u00eancia', 'nas publica\u00e7\u00f5es'), ('\U0001f4a1 Fim do', 'bloqueio criativo')]
            for ii, (l1, l2) in enumerate(items):
                ix = cx + 20 + ii * 210
                overlay_bgra(bg, text_bgra(l1, font_s, (255, 255, 255), a4), ix, cy + 175)
                overlay_bgra(bg, text_bgra(l2, font_s, (180, 180, 200), a4), ix, cy + 205)

        if scene['start'] == 45 and sp > 0.15:
            a5 = fa(0.15)
            days = ['Seg: Dica de uso', 'Ter: Bastidores', 'Qua: Produto destaque', 'Qui: Depoimento', 'Sex: Promo\u00e7\u00e3o']
            for di, day in enumerate(days):
                overlay_bgra(bg, text_bgra(f'\U0001f4c5 {day}', font_s, (200, 200, 210), a5), cx + 30, cy + 175 + di * 32)

        cv2.rectangle(bg, (0, H - 5), (int(W * t / TOTAL_SEC), H), ac, -1)
        writer.write(bg)

        if fi % 300 == 0:
            print(f"  {fi}/{TOTAL} ({t:.0f}s) open={openness:.2f}")

    writer.release()

    raw = os.path.join(OUTPUT_DIR, '_raw.mp4')
    print("Adicionando audio...")
    subprocess.run([
        FFMPEG_PATH, '-i', raw, '-i', AUDIO_PATH,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '20',
        '-c:a', 'aac', '-b:a', '128k', '-shortest',
        '-pix_fmt', 'yuv420p', '-y', OUTPUT_PATH
    ], check=True, capture_output=True)
    os.remove(raw)

    mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Concluido: {OUTPUT_PATH} ({mb:.1f} MB)")


if __name__ == '__main__':
    render()
