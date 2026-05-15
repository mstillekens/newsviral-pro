# Deploying VOZ DEL PUEBLO en tu Mac mini

Esta guía pone tu Mac mini como host del noticiero, con URL pública estable
para enseñarle a amigos y usar desde el celular en cualquier red.

## Arquitectura

```
  Tu celular / amigos en cualquier red
              │  HTTPS
              ▼
  ┌─────────────────────────────────┐
  │  Cloudflare Edge (gratis)       │
  │  voz.<tu-dominio>               │
  └────────────┬────────────────────┘
               │  túnel saliente
               │  (no abre puertos)
               ▼
  ┌─────────────────────────────────┐
  │  Tu Mac mini en casa            │
  │  ├─ cloudflared (servicio)      │
  │  └─ uvicorn webapp (servicio)   │
  │     └─ Replicate / Anthropic    │
  └─────────────────────────────────┘
```

Cloudflare Tunnel hace una conexión **saliente** desde el Mac mini hacia
Cloudflare. No tienes que abrir puertos en el router, no necesitas IP
pública. Y la URL es estable (no cambia).

## Cero a producción en ~15 min

### Antes de empezar

- Mac mini encendido, con macOS al día y Homebrew instalado
- Acceso SSH al Mac mini (o teclado/mouse físicos)
- (Opcional pero recomendado) **un dominio que controles** en Cloudflare DNS
  — gratis si lo registras en Cloudflare. Sin dominio, las URLs son
  temporales tipo `*.trycloudflare.com` y cambian cada reinicio.

### Paso 1 — Clonar el repo en el Mac mini

```bash
ssh jr@<ip-del-mac-mini>      # o entra físicamente
cd ~
git clone https://github.com/mstillekens/newsviral-pro.git
cd newsviral-pro
```

### Paso 2 — Correr el instalador

```bash
./webapp/deploy/install_macmini.sh
```

Esto va a:
1. Instalar `ffmpeg-full` y `cloudflared` con Homebrew (~3 min)
2. Crear `.venv/` e instalar todas las deps de Python
3. Generar `.env` con campos vacíos para que los llenes
4. Generar los dos `launchd plist` (webapp + cloudflared) en
   `~/Library/LaunchAgents/`

Al final imprime los siguientes pasos manuales.

### Paso 3 — Llenar `.env`

```bash
open -t .env
```

Pon tus valores:

```ini
REPLICATE_API_TOKEN=r8_...
ANTHROPIC_API_KEY=sk-ant-...
MINIMAX_VOICE_ID=                 # opcional, solo si ya clonaste voz

WEBAPP_USERNAME=voz                # o lo que quieras
WEBAPP_PASSWORD=<elige-uno-largo>  # !!! IMPORTANTE — si lo dejas vacío el sitio queda abierto
```

**Sin `WEBAPP_PASSWORD` cualquiera con el URL puede tronar tus créditos de
Replicate.** Pon algo largo y único. Lo puedes cambiar después editando
`.env` y `launchctl unload && load` del servicio webapp.

### Paso 4 — Autenticar Cloudflare Tunnel

Una sola vez:

```bash
cloudflared tunnel login
```

Abre tu navegador del Mac mini, te pide elegir el dominio (de tu cuenta
Cloudflare). Después:

```bash
cloudflared tunnel create voz-del-pueblo
```

Te imprime un UUID y crea un archivo de credenciales en
`~/.cloudflared/<UUID>.json`. Guarda ese UUID.

Ahora ruta DNS para tu subdominio:

```bash
cloudflared tunnel route dns voz-del-pueblo voz.tu-dominio.com
```

Y crea el archivo de config:

```bash
cat > ~/.cloudflared/config.yml <<EOF
tunnel: voz-del-pueblo
credentials-file: /Users/$USER/.cloudflared/<TUNNEL-UUID-AQUI>.json

ingress:
  - hostname: voz.tu-dominio.com
    service: http://localhost:8000
  - service: http_status:404
EOF
```

Reemplaza `<TUNNEL-UUID-AQUI>` con el UUID que Cloudflare te dio.

### Paso 5 — Levantar los dos servicios

```bash
launchctl unload ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist 2>/dev/null || true
launchctl load   ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist

launchctl unload ~/Library/LaunchAgents/com.vozdelpueblo.tunnel.plist 2>/dev/null || true
launchctl load   ~/Library/LaunchAgents/com.vozdelpueblo.tunnel.plist
```

`launchctl load` arranca el servicio **y** lo deja configurado para iniciar
automáticamente cada vez que prendas el Mac mini.

### Paso 6 — Confirmar

```bash
launchctl list | grep vozdelpueblo
# Debe mostrar ambos servicios. Si la columna PID es '-', está caído;
# revisa los logs.

curl -s http://localhost:8000/health
# {"ok":true}

tail -f logs/webapp.stdout.log
# log en vivo de la webapp
```

Y desde tu celular abre `https://voz.tu-dominio.com`. El navegador te
pide usuario + password (los que pusiste en `.env`).

## Sin dominio propio — Quick Tunnel

Si no tienes dominio en Cloudflare, salta los pasos 4 y solo carga el
servicio webapp. Para el túnel, en una terminal:

```bash
cloudflared tunnel --url http://localhost:8000
```

Te imprime una URL `https://random-words-here.trycloudflare.com`. Esa la
compartes. **Cambia cada vez que matas el comando.** Sirve para enseñarle
algo rápido a un amigo, pero para producción usa el túnel con nombre.

## Operaciones del día a día

### Actualizar a la última versión

```bash
cd ~/newsviral-pro
git pull
.venv/bin/pip install -r requirements.txt   # solo si cambiaron deps
launchctl unload ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist
launchctl load   ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist
```

### Ver logs en vivo

```bash
tail -f logs/webapp.stdout.log        # output normal
tail -f logs/webapp.stderr.log        # errores
tail -f ~/Library/Logs/cloudflared.stdout.log
```

### Parar todo

```bash
launchctl unload ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist
launchctl unload ~/Library/LaunchAgents/com.vozdelpueblo.tunnel.plist
```

### Cambiar el password

Edita `.env` y reinicia el webapp:

```bash
launchctl unload ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist
launchctl load   ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist
```

### Cambiar a otro puerto

Edita `~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist`, busca
`<string>8000</string>` y cámbialo. Edita también `~/.cloudflared/config.yml`
con el mismo puerto. Recarga ambos servicios.

### Borrar todo el estado (decisiones + cola)

```bash
rm webapp/state.json
launchctl unload ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist
launchctl load   ~/Library/LaunchAgents/com.vozdelpueblo.webapp.plist
```

## Compartir con amigos

Hay tres URLs públicas (sin password):

- `https://voz.tu-dominio.com/videos` — galería completa de videos producidos
- `https://voz.tu-dominio.com/video/<job_id>` — un mp4 directo (sirve para
  pegarlo en WhatsApp, etc.)
- `https://voz.tu-dominio.com/health` — endpoint de salud

Las URLs que **sí** piden password (tú las usas como operador):

- `/` — interfaz de swipe Sí/No
- `/refresh` — pedir noticias del RSS
- `/decide` — aceptar/rechazar (gasta créditos)
- `/queue` — ver la cola

Si un amigo te pregunta cómo entrar al panel: le das el username + password.
Si solo quieres que vea los videos terminados: mándale `/videos` o el link
directo de un video.

## Costos

- Cloudflare Tunnel: **gratis** (sin límite de bandwidth para uso personal)
- Cloudflare DNS: **gratis** si registras dominio ahí
- Mac mini eléctrico: ~$5–10/mes en luz
- Replicate por video: ~$1.50
- Anthropic por video: ~$0.005
- MiniMax voice cloning: $0 una vez entrenada, $0.005 por reuso

Total fijo: ~$10/mes (luz). Variable: $1.50 por video que generas.

## Troubleshooting

**`launchctl list` muestra "-" en PID**: el servicio se cayó. Revisa
`logs/webapp.stderr.log`. Causa común: `.env` mal formateado o key inválida.

**El túnel está arriba pero la URL devuelve 502**: la webapp no está
respondiendo. Confirma con `curl http://localhost:8000/health` desde el Mac
mini. Si eso no responde, reinicia webapp.

**Browser dice "wrong username or password"**: revisa que `WEBAPP_PASSWORD`
en `.env` no tenga espacios al inicio/fin. Si lo cambias, reinicia el
servicio webapp.

**Las imágenes/videos no cargan**: revisa que `replicate_outputs/` y
`logs/runs/` existan y tengan permisos de escritura para el usuario que
corre el servicio.

**"Address already in use"**: hay otro proceso en el puerto 8000. Mátalo
con `lsof -i :8000` y `kill <PID>`.

## Siguiente paso opcional: HTTPS sin Cloudflare

Si no quieres depender de Cloudflare, puedes usar Tailscale Funnel o un VPS
con Caddy + Let's Encrypt. La webapp es solo HTTP en `localhost:8000`, así
que cualquier reverse proxy con TLS te sirve. Pero Cloudflare Tunnel es la
opción más simple y gratis para Mac mini detrás de NAT.
