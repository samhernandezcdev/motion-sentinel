# Motion Sentinel

Laboratorio de anomalías en la visión artificial ambiental.

## Features

- Detección de movimiento en tiempo real
- Comparación de fotogramas
- Métricas de movimiento
- Clasificación de alertas
- Grabación de instantáneas
- Almacenamiento en búfer de fotogramas seguro para subprocesos
- Configuración YAML con validación tipada
- Registro estructurado
- Modo headless para ejecución sin ventana OpenCV

## Ejecuta

```bash
pip install -e .
motion-sentinel
```

## CLI

```bash
motion-sentinel \
  --config config/default.yaml \
  --profile dev \
  --source data/test_video.mp4 \
  --headless \
  --output-dir data/snapshots
```

Flags disponibles:

- `--config`: ruta al YAML base. Por defecto usa `config/default.yaml`.
- `--profile`: nombre de perfil en `config/profiles/<name>.yaml` o ruta directa a otro YAML. El perfil se mezcla sobre el YAML base.
- `--source`: sobrescribe `capture.source` con índice de webcam, ruta local, RTSP o HTTP/MJPEG.
- `--headless`: desactiva `cv2.imshow()` y `cv2.waitKey()` para servidores, Docker o CI.
- `--output-dir`: sobrescribe `recorder.output_dir`.

## Compatibilidad

`config/default.yaml` sigue siendo compatible. Phase 1 no cambia el algoritmo de detección: el detector continúa usando diferencia absoluta entre frames consecutivos, threshold, dilatación y contornos.

Cambios de comportamiento intencionales:

- Las ROIs configuradas sobre el frame fuente se escalan automáticamente al tamaño del frame procesado cuando `detection.resize_width` reduce la imagen.
- `frame_buffer.drop_oldest: false` ahora bloquea hasta que haya espacio, como indicaba la documentación. `true` mantiene la política de descartar el frame más antiguo para priorizar frames frescos.
- RTSP/HTTP no se tratan como archivos locales rebobinables.
- Credenciales embebidas en URLs RTSP/HTTP se redactan en logs.
- `recorder.image_format` ahora se respeta al construir el recorder.

## Desarrollo

```bash
python -m compileall motion_sentinel
python -m pytest
python -m ruff check .
```

`pytest` y `ruff` son herramientas de desarrollo; si no están instaladas localmente, instálalas en tu entorno antes de ejecutar esos comandos.
