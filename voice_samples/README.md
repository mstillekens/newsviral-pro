# Voice samples — fuentes legales para clonar voz chilanga

Tres caminos para conseguir un sample de voz mexicana sin grabarte:

## 1) Mozilla Common Voice (CC0, gratis, calidad variable)

Mozilla Common Voice tiene miles de grabaciones de hablantes mexicanos
bajo licencia **CC0** (dominio público). Es la fuente más limpia legalmente.

```bash
# Descarga el dataset completo de Spanish: https://commonvoice.mozilla.org/es/datasets
# (es grande, ~50GB). Más práctico: busca un clip mexicano en su buscador web,
# guardas el WAV, y pasas el archivo:
python clone_voice.py --preview voice_samples/common_voice_mx.wav
```

**Tip**: filtra por `accent: Mexican` cuando explores. Las grabaciones son
voluntarias así que la calidad varía — escucha 3-4 antes de elegir.

## 2) YouTube con licencia Creative Commons (`yt-dlp`)

Muchos creadores mexicanos suben videos bajo licencia CC. Filtra YouTube por
"Creative Commons" y busca un narrador chilango con voz que te guste.

```bash
brew install yt-dlp                          # una vez
python clone_voice.py --preview "https://www.youtube.com/watch?v=XXXXX"
```

El script extrae 30 segundos de audio (segundo 5 al 35 — salta intros), lo
entrena, y genera un preview para que escuches antes de comprometerlo.

**Advertencia**: aunque YouTube te dé el video bajo CC, la voz como tal es un
derecho de personalidad. Para uso comercial pide permiso al creador si la
voz es identificable como suya. Para clonar el "estilo" general de un acento
chilango y usarlo en otro contenido, normalmente es uso transformativo.

## 3) Podcasts mexicanos con licencia abierta

Plataformas como Anchor / Spotify a veces tienen episodios CC-BY. Busca:

- *La Octava* (algunos episodios libres)
- *Radio Educación* (parte del catálogo es dominio público — radio pública)
- Podcasts indie chilangos que indiquen "uso libre" en su descripción

```bash
python clone_voice.py --preview "https://podcast.example.com/ep42.mp3"
```

## Recomendación práctica

1. Empieza con `--preview` siempre. Eso te da un audio de prueba sin tocar
   el `.env`.
2. Una vez te guste cómo suena, corre el mismo comando sin `--preview` para
   activar esa voz en el pipeline.
3. Puedes tener varios `MINIMAX_VOICE_ID_*` archivados (rename del campo en
   `.env`) y rotar entre ellos para tener varias voces de noticiero.

## Constraints técnicos (MiniMax)

- Formato: MP3, M4A, o WAV.
- Duración: 10 s mínimo, 5 min máximo.
- Tamaño: < 20 MB.
- Un solo hablante, sin música de fondo, idealmente cuarto silencioso.

## Revertir a la voz default

Borra (o comenta) `MINIMAX_VOICE_ID` en `.env`. El pipeline vuelve a usar
`English_Wiselady` con `language_boost=Spanish`.
