"""Hızlı örnek: seçili sahnelerin görselleri (eski film tonu uygulanmış, tek küçük JPG) + Türkçe ses örnekleri (mp3)."""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make
from PIL import Image

script = make.load(sys.argv[1])
idx = [int(x) for x in os.environ["ONLY"].split(",")]
make.cmd_images(script, 0, 1)
os.makedirs("sample", exist_ok=True)
tiles = [make.prep_bg(script["scenes"][i]).resize((640, 360)) for i in idx]
grid = Image.new("RGB", (640 * 3, 360 * ((len(tiles) + 2) // 3)))
for k, t in enumerate(tiles):
    grid.paste(t, ((k % 3) * 640, (k // 3) * 360))
grid.save("sample/gorseller.jpg", quality=82)
import edge_tts
text = script["scenes"][0]["text"] + " " + script["scenes"][4]["text"]
for v in ("tr-TR-AhmetNeural", "tr-TR-EmelNeural"):
    asyncio.run(edge_tts.Communicate(text, v, rate="-4%").save(f"sample/ses_{v.split('-')[2]}.mp3"))
print("ok")
