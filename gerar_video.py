import subprocess, os, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

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

avatar_pil = Image.open(AVATAR_PATH).convert('RGBA').resize((300, 300), Image.LANCZOS)

scenes = [
    {'start': 0, 'end': 10, 'title': 'IA Generativa', 'sub': 'Criação de Conteúdo para Redes Sociais',
     'tag': 'Posts · Legendas · Imagens · Roteiros', 'color': (100, 200, 255)},
    {'start': 10, 'end': 28, 'title': 'Como funciona?', 'sub': 'Um comando gera posts, legendas, imagens e roteiros',
     'tag': 'Tudo adaptado ao tom da sua marca', 'color': (180, 150, 255)},
    {'start': 28, 'end': 45, 'title': 'Benefícios', 'sub': 'Economia de horas · Consistência · Fim do bloqueio',
     'tag': 'Perfeito para pequenos negócios e criadores', 'color': (80, 220, 160)},
    {'start': 45, 'end': 55, 'title': 'Na prática', 'sub': 'Um mês de posts em 15 minutos',
     'tag': 'Basta revisar e publicar', 'color': (255, 200, 60)},
    {'start': 55, 'end': 60, 'title': 'IA não substitui você', 'sub': 'Ela acelera seu trabalho',
     'tag': 'Experimente você também!', 'color': (100, 200, 255)},
]

def text_to_arr(text, font, fill, shadow=False):
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new('RGBA', (tw + 40, th + 20), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if shadow:
        draw.text((2, 2), text, font=font, fill=(0, 0, 0, 180))
    draw.text((0, 0), text, font=font, fill=fill)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

def overlay_transparent(bg, overlay, x, y):
    oh, ow = overlay.shape[:2]
    if y + oh > bg.shape[0] or x + ow > bg.shape[1]:
        oh = min(oh, bg.shape[0] - y)
        ow = min(ow, bg.shape[1] - x)
        overlay = overlay[:oh, :ow]

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+oh, x:x+ow, c] = (1 - alpha) * bg[y:y+oh, x:x+ow, c] + alpha * overlay[:, :, c]
    return bg

def setup():
    bg_cache = {}
    for scene in scenes:
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg_col = np.array([18, 18, 40], dtype=np.uint8)
        bg[:] = bg_col
        for y in range(H):
            fade = 0.15 * math.sin(y / H * math.pi)
            bg[y] = (bg[y].astype(float) * (1 - fade) + np.array([30, 25, 55]) * fade).astype(np.uint8)
        bg_cache[scene['start']] = bg.copy()
    return bg_cache

bg_cache = setup()

TOTAL_FRAMES = TOTAL_SEC * FPS

def render():
    print("Rendering...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, '_raw.mp4'), fourcc, FPS, (W, H))

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

        for i in range(20):
            px = int((math.sin(t * 0.2 + i * 1.7) * 0.5 + 0.5) * W)
            py = int((math.cos(t * 0.15 + i * 2.3) * 0.5 + 0.5) * H)
            cv2.circle(bg, (px, py), 2, (ac[0]//2, ac[1]//2, ac[2]//2), -1, cv2.LINE_AA)

        ax = 940 + int(math.sin(t * 0.5) * 4)
        ay = 60 + int(math.sin(t * 0.3) * 3)
        av_f = np.array(avatar_pil)
        blink_t = (fi % 150)
        if blink_t < 4:
            ey = int(avatar_pil.height * 0.45)
            av_f[ey-10:ey+10, :, 3] = 0
        overlay_transparent(bg, av_f, ax, ay)

        card_x, card_y = 50, 160

        def fa(delay, speed=2.0):
            return min(1.0, max(0.0, (sp - delay) * speed))

        a1 = fa(0.0)
        if a1 > 0:
            title = text_to_arr(scene['title'], font_l, (ac[0], ac[1], ac[2], int(255*a1)), shadow=True)
            overlay_transparent(bg, title, card_x, card_y)

        a2 = fa(0.12)
        if a2 > 0:
            sub = text_to_arr(scene['sub'], font_m, (230, 230, 230, int(230*a2)), shadow=True)
            overlay_transparent(bg, sub, card_x, card_y + 65)

        a3 = fa(0.25)
        if a3 > 0:
            tag = text_to_arr(scene['tag'], font_s, (180, 180, 200, int(180*a3)), shadow=True)
            overlay_transparent(bg, tag, card_x, card_y + 120)

        if scene['start'] == 28 and sp > 0.2:
            a4 = fa(0.2)
            items = [('⏱ Economia', 'de horas'), ('📈 Consistência', 'nas publicações'), ('💡 Fim do', 'bloqueio criativo')]
            for ii, (l1, l2) in enumerate(items):
                ix = card_x + 20 + ii * 210
                l1_a = text_to_arr(l1, font_s, (255, 255, 255, int(255*a4)), shadow=True)
                overlay_transparent(bg, l1_a, ix, card_y + 170)
                l2_a = text_to_arr(l2, font_s, (180, 180, 200, int(180*a4)), shadow=True)
                overlay_transparent(bg, l2_a, ix, card_y + 200)

        if scene['start'] == 45 and sp > 0.15:
            a5 = fa(0.15)
            days = ['Seg: Dica de uso', 'Ter: Bastidores', 'Qua: Produto destaque', 'Qui: Depoimento', 'Sex: Promoção']
            for di, day in enumerate(days):
                d = text_to_arr(f'📅 {day}', font_s, (200, 200, 210, int(200*a5)), shadow=True)
                overlay_transparent(bg, d, card_x + 30, card_y + 175 + di * 32)

        bar = int(W * (t / TOTAL_SEC))
        cv2.rectangle(bg, (0, H - 4), (bar, H), ac, -1)

        writer.write(bg)

        if fi % 300 == 0:
            print(f"  {fi}/{TOTAL_FRAMES} ({t:.0f}s)")

    writer.release()

    raw = os.path.join(OUTPUT_DIR, '_raw.mp4')
    print("Muxing audio...")
    subprocess.run([
        FFMPEG_PATH, '-i', raw, '-i', AUDIO_PATH,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-c:a', 'aac', '-b:a', '128k', '-shortest',
        '-pix_fmt', 'yuv420p', '-y', OUTPUT_PATH
    ], check=True, capture_output=True)

    os.remove(raw)
    mb = os.path.getsize(OUTPUT_PATH) / (1024*1024)
    print(f"Done: {OUTPUT_PATH} ({mb:.1f} MB)")

if __name__ == '__main__':
    render()
