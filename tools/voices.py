"""Gemini ses örnekleri: aynı metni farklı seslerle okur (sample/ses_<isim>.mp3)."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make
from say import chunks

script = make.load(sys.argv[1])
text = " ".join(c for i in (0, 4) for c, _ in chunks(script["scenes"][i]["text"]))
os.makedirs("sample", exist_ok=True)
for v in os.environ["VOICES"].split(","):
    wav = f"sample/{v}.wav"
    make.gemini_tts(text, v, wav)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-b:a", "128k", f"sample/ses_{v}.mp3"], check=True)
    os.remove(wav)
    print("ok", v, flush=True)
