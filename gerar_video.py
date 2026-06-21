import subprocess, os, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
AVATAR_PATH = os.path.join(OUTPUT_DIR, 'avatar.png')
AUDIO_PATH = os.path.join(OUTPUT_DIR, 'audio_video.mp3')
AMPLITUDES_PATH = os.path.join(OUTPUT_DIR, 'amplitudes.npy')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'video_final.mp4')
import shutil
FFMPEG_PATH = shutil.which('ffmpeg') or r'C:\Users\carlosedsilva\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe'

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
if len(smiles) > 0:
    _mx, _my, mw, mh = smiles[0]
else:
    # Fallback to lower part of detected face
    _mx = int(fw * 0.28)
    _my = int(fh * 0.68)
    mw = int(fw * 0.44)
    mh = int(fh * 0.15)
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


# Pre-capture larger mouth context for natural warping
_pad_t = int(mh * 0.7)
_pad_b = int(mh * 0.5)
_pad_l = int(mw * 0.2)
_pad_r = int(mw * 0.2)
_ctx_y0 = max(0, my - _pad_t)
_ctx_y1 = min(AVATAR_SIZE, my + mh + _pad_b)
_ctx_x0 = max(0, mx - _pad_l)
_ctx_x1 = min(AVATAR_SIZE, mx + mw + _pad_r)
_ctx_h = _ctx_y1 - _ctx_y0
_ctx_w = _ctx_x1 - _ctx_x0
_ctx_ref = avatar_base[_ctx_y0:_ctx_y1, _ctx_x0:_ctx_x1].copy()

# Find lip seam within context (darkest row = lip line)
_mouth_gray = cv2.cvtColor(_ctx_ref, cv2.COLOR_BGR2GRAY)
_row_means = [_mouth_gray[y, :].mean() for y in range(_ctx_h)]
_search_center = my - _ctx_y0 + mh // 2
_search_start = max(0, _search_center - mh // 2)
_search_end = min(_ctx_h, _search_center + mh // 2)
_lip_seam_y = int(np.argmin(_row_means[_search_start:_search_end])) + _search_start
_ctx_seam = _lip_seam_y

# Pre-compute vertical gaussian weight for smooth warping falloff
_ctx_rows = np.arange(_ctx_h, dtype=np.float32)
_sigma = _ctx_h * 0.35
_ctx_weight = np.exp(-((_ctx_rows - _ctx_seam) ** 2) / (2 * _sigma ** 2))

# Context mask for blending
_ctx_mask = np.zeros((_ctx_h, _ctx_w), dtype=np.float32)
_cx_c, _cy_c = _ctx_w // 2, _ctx_seam
for y in range(_ctx_h):
    for x in range(_ctx_w):
        dx = (x - _cx_c) / (_ctx_w / 2)
        dy = (y - _cy_c) / (_ctx_h / 2)
        dist = dx * dx + dy * dy
        _ctx_mask[y, x] = max(0, 1 - dist)
_ctx_mask = cv2.GaussianBlur(_ctx_mask, (15, 15), 0)
_ctx_mask = np.clip(_ctx_mask * 1.5, 0, 1)


def mouth_sync(img, openness, t=0):
    op = min(1.0, openness * 1.4)
    op = op * op * (3 - 2 * op)

    breathe = (math.sin(t * 2.8) * 0.5 + math.sin(t * 5.3) * 0.3) * 0.02
    op = max(0.0, min(1.0, op + breathe))

    # Organic asymmetry in mouth opening
    asymmetry = math.sin(t * 4.7) * 0.15

    max_upper = int(_ctx_h * 0.02)
    max_lower = int(_ctx_h * 0.08)
    upper_shift = int(max_upper * op)
    lower_shift = int(max_lower * op * (1 + asymmetry * 0.3))

    # Smooth displacement curve using gaussian profile
    remap_y = np.zeros(_ctx_h, dtype=np.float32)
    seam = _ctx_seam

    for y in range(_ctx_h):
        w = _ctx_weight[y]
        if y < seam:
            # Upper: compress upward, weighted by gaussian proximity
            offset = upper_shift * w
            if y > seam - offset:
                # Remap near-seam rows upward
                local_t = (seam - y) / max(1, offset)
                remap_y[y] = seam - offset * math.sqrt(local_t)
            else:
                remap_y[y] = y - offset * 0.3
        else:
            # Lower: stretch downward, weighted by gaussian proximity
            offset = lower_shift * w
            if y < seam + offset:
                local_t = (y - seam) / max(1, offset)
                remap_y[y] = seam + offset * math.sqrt(local_t)
            else:
                remap_y[y] = y + offset * 0.3

    remap_y = np.clip(remap_y, 0, _ctx_h - 1)

    # Horizontal remap: slight widening at mouth opening
    remap_x = np.tile(np.arange(_ctx_w, dtype=np.float32), (_ctx_h, 1))
    for y in range(_ctx_h):
        stretch = op * _ctx_weight[y] * 0.04
        for x in range(_ctx_w):
            dx = (x - _cx_c) / (_ctx_w / 2)
            remap_x[y, x] = x + dx * stretch * _ctx_w / 2

    # Apply remap
    r = cv2.remap(_ctx_ref, remap_x.astype(np.float32),
                   np.tile(remap_y.reshape(_ctx_h, 1), (1, _ctx_w)).astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Realistic oral cavity & teeth for closed-mouth avatar
    if op > 0.05:
        c_top = seam - upper_shift * 0.3
        c_bot = seam + lower_shift * 0.7
        cavity_h_actual = max(0, int(c_bot - c_top))

        if cavity_h_actual > 2:
            overlay = r.copy()
            
            # 1. Dark Cavity (Deep maroon/black)
            cavity_pts = []
            half_w = int(_ctx_w * 0.28)
            for i in range(21):
                t_i = i / 20.0
                yy = c_top + t_i * cavity_h_actual
                taper = 1.0 - 0.5 * math.sin(t_i * math.pi) ** 2
                ww_i = max(2, int(half_w * taper))
                cavity_pts.append([_cx_c - ww_i, yy])
            for i in range(20, -1, -1):
                t_i = i / 20.0
                yy = c_top + t_i * cavity_h_actual
                taper = 1.0 - 0.5 * math.sin(t_i * math.pi) ** 2
                ww_i = max(2, int(half_w * taper))
                cavity_pts.append([_cx_c + ww_i, yy])
            
            cavity_pts = np.array(cavity_pts, dtype=np.int32)
            cv2.fillPoly(overlay, [cavity_pts], (20, 15, 18), cv2.LINE_AA)
            
            # 2. Upper Teeth (Soft white arch)
            if op > 0.15:
                t_h = max(2, int(cavity_h_actual * 0.4))
                t_top = int(c_top)
                t_bot = min(int(c_top + t_h), int(c_bot))
                
                teeth_pts = []
                for i in range(11):
                    t_i = i / 10.0
                    yy = t_bot - 2 * math.sin(t_i * math.pi)
                    teeth_pts.append([_cx_c - half_w + 4 + i * (half_w * 2 - 8) // 10, yy])
                for i in range(10, -1, -1):
                    teeth_pts.append([_cx_c - half_w + 4 + i * (half_w * 2 - 8) // 10, t_top])
                    
                teeth_pts = np.array(teeth_pts, dtype=np.int32)
                cv2.fillPoly(overlay, [teeth_pts], (190, 185, 180), cv2.LINE_AA)
                
                # Faint shadow between upper lip and teeth
                cv2.line(overlay, (_cx_c - half_w, t_top), (_cx_c + half_w, t_top), (30, 20, 20), 2)

            # 3. Blend the cavity into the lips with slight blurring
            mask = np.zeros((_ctx_h, _ctx_w), dtype=np.uint8)
            cv2.fillPoly(mask, [cavity_pts], 255)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            alpha_cavity = mask.astype(float) / 255.0
            alpha_cavity = np.stack([alpha_cavity]*3, axis=2)
            
            r = (r.astype(float) * (1 - alpha_cavity) + overlay.astype(float) * alpha_cavity).astype(np.uint8)

    # Sub-blend: composite r back into img at context position
    alpha = _ctx_mask
    roi = img[_ctx_y0:_ctx_y1, _ctx_x0:_ctx_x1]
    for c in range(3):
        roi[:, :, c] = (roi[:, :, c].astype(float) * (1 - alpha) +
                         r[:, :, c].astype(float) * alpha).astype(np.uint8)
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
    tw = bbox[2] + 16
    th = int(bbox[3] * 1.2) + 16
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
        av = mouth_sync(av, openness, t)

        # To BGRA for compositing
        av_bgra = cv2.cvtColor(av, cv2.COLOR_BGR2BGRA)
        av_bgra[:, :, 3] = face_mask

        # Position - larger avatar, more centered
        ax = 820 + int(math.sin(t * 1.2) * 6)
        ay = 10 + int(math.sin(t * 0.8) * 4)
        overlay_bgra(bg, av_bgra, ax, ay)

        # Text - moved up to avoid bottom clipping
        cx, cy = 50, 80

        def fa(delay, speed=2.0):
            return max(0, min(255, int(255 * min(1.0, max(0.0, (sp - delay) * speed)))))

        a1 = fa(0.0)
        if a1 > 0:
            overlay_bgra(bg, text_bgra(scene['title'], font_l, ac, a1), cx, cy)
        a2 = fa(0.12)
        if a2 > 0:
            overlay_bgra(bg, text_bgra(scene['sub'], font_m, (235, 235, 235), a2), cx, cy + 145)
        a3 = fa(0.25)
        if a3 > 0:
            overlay_bgra(bg, text_bgra(scene['tag'], font_s, (175, 175, 195), a3), cx, cy + 205)

        if scene['start'] == 28 and sp > 0.2:
            a4 = fa(0.2)
            items = [('\u23f1 Economia', 'de horas'), ('\U0001f4c8 Consist\u00eancia', 'nas publica\u00e7\u00f5es'), ('\U0001f4a1 Fim do', 'bloqueio criativo')]
            for ii, (l1, l2) in enumerate(items):
                ix = cx + 20 + ii * 210
                overlay_bgra(bg, text_bgra(l1, font_s, (255, 255, 255), a4), ix, cy + 255)
                overlay_bgra(bg, text_bgra(l2, font_s, (180, 180, 200), a4), ix, cy + 285)

        if scene['start'] == 45 and sp > 0.15:
            a5 = fa(0.15)
            days = ['Seg: Dica de uso', 'Ter: Bastidores', 'Qua: Produto destaque', 'Qui: Depoimento', 'Sex: Promo\u00e7\u00e3o']
            for di, day in enumerate(days):
                overlay_bgra(bg, text_bgra(f'\U0001f4c5 {day}', font_s, (200, 200, 210), a5), cx + 30, cy + 255 + di * 32)

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
