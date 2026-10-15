"""Кухня NVR: убирает нестандартные `input_args`, из-за которых сторож убивает ffmpeg.

Замерено 15.10.2026. Кухня была **единственной** из семи камер NVR с
нестандартными аргументами:

    kitchen   fps=3   -rtsp_transport tcp -analyzeduration 10000000 -probesize 10000000
    остальные шесть  fps=5   (по умолчанию)

`-analyzeduration 10000000` — это 10 секунд потока, которые ffmpeg буферизует
**до первого кадра**. Сторож NVR убивает камеру, не получив кадров за 20 с, и
поднимает заново: `watchdog.kitchen Terminating the existing ffmpeg process` →
`kitchen: exiting subprocess` каждые ~100 секунд, и так непрерывно.

Пока это так, у кухни остаётся ОДИН потребитель потока (шлюз), потому что NVR её
не тянет, — а по замеру 11:53 именно один потребитель означает `play_audio`
×25.7 и заикание. То есть нестабильный захват NVR — не побочное дело, а причина
заикания комнаты.

Ещё одна деталь, которая делает правку очевидной: эти аргументы адресованы
ПРЯМОЙ связи с камерой (RTSP-подстройка потока), а NVR тяньёт уже из go2rtc,
который нормализует поток сам. Обходные пути для старой проблемы остались в
конфиге после того, как путь изменился.

Проверка КОНЕЧНОГО состояния: после правок у кухни не должно остаться
`input_args`, и YAML обязан разбираться. Скрипт не пишет файл, если это не так
или если исходный блок не найден ровно один раз.
"""
import shutil
import sys

PATH = "/config/config.yaml"
BAK = "/config/config.yaml.bak.20261015.kitchenargs"

OLD = (
    "          input_args:\n"
    "            - -rtsp_transport\n"
    "            - tcp\n"
    "            - -analyzeduration\n"
    '            - "10000000"\n'
    "            - -probesize\n"
    '            - "10000000"\n'
)

text = open(PATH, encoding="utf-8").read()
shutil.copyfile(PATH, BAK)
print("  бэкап: %s" % BAK)

n = text.count(OLD)
if n != 1:
    print("  ПРОПУЩЕНО: блок input_args кухни найден %d раз (нужен ровно 1)" % n)
    print("  запись НЕ выполнена")
    sys.exit(1)
text = text.replace(OLD, "", 1)

# Конечное состояние, а не факт замены. Проверка по РАЗОБРАННОМУ YAML и по
# КОНКРЕТНОЙ камере: первая версия искала `input_args` во всём файле и
# нашла его у balcony, то есть отказалась применить верную правку из-за чужого
# ключа. Проверка должна отвечать на вопрос «а у кухни осталось?», а не
# «а есть ли где-нибудь».
try:
    import yaml
    cfg = yaml.safe_load(text)
except Exception as exc:
    print("  YAML перестал разбираться: %r — запись отменена" % (exc,))
    sys.exit(1)

kitchen = cfg["cameras"]["kitchen"]["ffmpeg"]["inputs"][0]
if "input_args" in kitchen:
    print("  у кухни всё ещё есть input_args=%r — запись отменена"
          % (kitchen["input_args"],))
    sys.exit(1)
others = [n for n, c in cfg["cameras"].items()
          if "input_args" in c["ffmpeg"]["inputs"][0]]
print("  у кухни input_args больше нет; path=%s" % kitchen["path"])
print("  input_args остались у: %s (это не трогаем)" % (others or "—"))
open(PATH, "w", encoding="utf-8").write(text)
print("  записано")
