"""Правка конфига Frigate: 15.10.2026, две причины заикания на кухне и в коридоре.

1. **Дубль производителя в коридоре.** В блоке go2rtc поток назывался
   `corridor1` — именем, от которого go2rtc отказался ещё 11.10. Из-за этого
   на камеру .247 приходили ДВА RTSP-производителя: один для шлюза
   (`camera-corridor1`), второй для NVR (`corridor1`). `majestic` однопоточный,
   и при двух видеосессиях его выходной аудиоканал недокармливал —
   `Device 0 is empty`. Имя приведено к тому, что уже читает шлюз.

2. **Цикл падений ffmpeg на кухне.** В логах NVR:
       [rtsp] RTP: PT=60: bad cseq 0
       [segment] Timestamps are unseverable
       [segment] Non-monotonous DTS
       kitchen exceeded fps limit. Exiting ffmpeg...
   `PT=60` — это полезная нагрузка АУДИО, а для детекции NVR нужен только
   видеокадр. Добавлен `-an`, чтобы ffmpeg вообще не договаривался об аудио.

Правки точные, с проверкой, что каждая сработала ровно один раз: молчаливая
замена, ничего не изменившая, здесь стоила бы неделю диагностики.
"""
import shutil
import sys

PATH = "/config/config.yaml"
BAK = "/config/config.yaml.bak.20261015"

EDITS = [
    # 1. имя потока в go2rtc
    (
        "    corridor1:\n      - rtsp://root:2441@192.168.22.247/stream=0#backchannel=1",
        "    camera-corridor1:\n      - rtsp://root:2441@192.168.22.247/stream=0#backchannel=1",
        "go2rtc: поток corridor1 переименован в camera-corridor1",
    ),
    # 1b. камера corridor1 в секции cameras -> читает тот же поток.
    # Шаблон — только строка пути: в этом блоке нет `input_args` (в отличие от
    # balcony), и первая версия правки искала структуру с ним, не нашла ничего
    # и НЕ записала файл — что и было правильным поведением скрипта.
    (
        "        - path: rtsp://127.0.0.1:8554/corridor1",
        "        - path: rtsp://127.0.0.1:8554/camera-corridor1",
        "камера corridor1 читает camera-corridor1",
    ),
    (
        "      stream_name: corridor1",
        "      stream_name: camera-corridor1",
        "stream_name камеры corridor1",
    ),
    # 2. кухня: не договариваться об аудио
    (
        "  kitchen:\n"
        "    live:\n"
        "      stream_name: kitchen\n"
        "    ffmpeg:\n"
        "      inputs:\n"
        "        - path: rtsp://127.0.0.1:8554/kitchen\n"
        "          roles:\n"
        "            - record\n"
        "            - detect\n"
        "          input_args:\n"
        "            - -rtsp_transport\n"
        "            - tcp\n",
        "  kitchen:\n"
        "    live:\n"
        "      stream_name: kitchen\n"
        "    ffmpeg:\n"
        "      inputs:\n"
        "        - path: rtsp://127.0.0.1:8554/kitchen\n"
        "          roles:\n"
        "            - record\n"
        "            - detect\n"
        "          input_args:\n"
        "            - -rtsp_transport\n"
        "            - tcp\n"
        "            # 15.10.2026: NVR не нужен звук, а его согласование роняло\n"
        "            # ffmpeg в цикл (PT=60 bad cseq, unseverable timestamps), и\n"
        "            # каждая переподключка отнимала у камеры ресурс на видео —\n"
        "            # аудиоканал недокармливал и комната заикала.\n"
        "            - -an\n",
        "кухня: -an (аудио не договаривается)",
    ),
]

text = open(PATH, encoding="utf-8").read()
shutil.copyfile(PATH, BAK)
print("  бэкап: %s" % BAK)

failed = 0
for old, new, what in EDITS:
    n = text.count(old)
    if n != 1:
        print("  ПРОПУЩЕНО (совпадений %d): %s" % (n, what))
        failed += 1
        continue
    text = text.replace(old, new, 1)
    print("  ок: %s" % what)

if failed:
    print("  запись НЕ выполнена: %d правок не нашли свой текст" % failed)
    sys.exit(1)

open(PATH, "w", encoding="utf-8").write(text)
print("  записано")
