# 🎬 IA Generativa para Redes Sociais

Projeto que demonstra o potencial das IAs Generativas na criação de conteúdo para redes sociais, com um vídeo de 1 minuto apresentando um avatar virtual brasileiro com narração em português e sincronização labial (lip sync).

## 📋 Conteúdo

- **[gerar_video.py](gerar_video.py)** — Script principal que gera o vídeo completo (avatar, animações, texto, áudio)
- **[apresentacao.html](apresentacao.html)** — Apresentação em slides com progressão automática de 60s
- **[roteiro_video.md](roteiro_video.md)** — Roteiro narrado com divisão de cenas e tempos
- **[audio_video.mp3](audio_video.mp3)** — Narração gerada por TTS (voz Francisca - Microsoft)
- **[avatar.png](avatar.png)** — Imagem do avatar virtual gerado por IA
- **[legendas.vtt](legendas.vtt)** — Legendas do áudio

## 🎯 Estrutura do Vídeo

| Tempo | Seção | Descrição |
|-------|-------|-----------|
| 0s-10s | Abertura | IA Generativa para criação de conteúdo |
| 10s-28s | Como funciona | Geração de posts, legendas, imagens e roteiros |
| 28s-45s | Benefícios | Economia, consistência, fim do bloqueio criativo |
| 45s-55s | Na prática | Exemplo de calendário editorial |
| 55s-60s | Encerramento | IA acelera, não substitui |

## 🛠️ Como usar

### Pré-requisitos

Instale as dependências:

```bash
pip install edge-tts opencv-python pillow numpy
```

Baixe o [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) e coloque o `ffmpeg.exe` na pasta do projeto, ou instale via:

```bash
winget install "FFmpeg (Essentials Build)"
```

### Gerar o vídeo

```bash
python gerar_video.py
```

O vídeo será salvo como `video_final.mp4` (MP4, H.264, 1280x720, ~60s).

### Personalizar

Edite o dicionário `scenes` em `gerar_video.py` para alterar textos, cores e duração. O áudio pode ser regenerado com `edge-tts` ajustando o texto ou a taxa de fala.

## 🧰 Tecnologias

- **Python 3** — OpenCV, Pillow, NumPy
- **Edge TTS** — Voz natural em português brasileiro
- **Stable Diffusion (Pollinations.ai)** — Geração do avatar
- **FFmpeg** — Codificação H.264 e mixagem de áudio
- **GitHub CLI** — Versionamento e publicação

## 🗣️ Sincronização Labial (Lip Sync)

O avatar possui animação de boca sincronizada com o áudio usando warping de textura real:

- **Contexto ampliado** — Warp aplicado em região 2.7x maior que a boca, incluindo pele ao redor para transições naturais
- **Warp 2D orgânico** — `cv2.remap` com perfil de deslocamento gaussiano: suave no centro, zero nas bordas
- **Falloff gaussiano** — Deformação máxima na linha dos lábios, desaparecendo gradualmente no rosto
- **Cavidade oral poligonal** — Preenchimento com curva de taper, bordas suaves e profundidade gradiente
- **Dentes em arco** — Segmentos individuais seguindo curvatura senoidal, superiores e inferiores
- **Assimetria dinâmica** — Variação lado a lado em cada frame para movimento orgânico
- **Micro-variações** — Respiração com frequências combinadas (2.8Hz + 5.3Hz)
- **Máscara elíptica** — Feathering suave para blend imperceptível com o rosto

## 📄 Licença

MIT
