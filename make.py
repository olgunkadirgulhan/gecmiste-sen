"""Geçmişte Sen — "Eğer X yılında ... olsaydın?" resimli anlatı videoları (16:9, ~6-8 dk).

  python make.py images SCRIPT --part i --of n   sahne arka planlarını üret (CPU, DreamShaper 8 + LCM) -> cache/
  python make.py voice SCRIPT                    anlatım sesi (edge-tts) -> cache/
  python make.py render SCRIPT                   görüntü + altyazı + karakter + müzik -> out/video.mp4, out/thumb.jpg
Arka planlarda insan yok; ana karakter ("sen") kodla çizilir, her videoda aynı ve kanala ait."""
import asyncio, hashlib, json, math, os, random, re, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
SR = 48000
# yerli Türkçe ses: çok dilli sesler (Florian) bazı cümleleri İngilizce okuyordu
# İlk videolarda sesler sırayla denenir; izlenme süresi en iyi olan kalıcı ses olur (bkz. voice_for)
ROTATION = ["gemini:Charon", "gemini:Sadaltager", "gemini:Sulafat"]
VOICE = os.environ.get("VOICE") or ""  # "gemini:<ses>" -> Gemini TTS, boş -> sıradaki ses
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-3.8-flash-tts")
TTS_STYLE = ("Read the Turkish transcript below aloud. Voice: a warm, calm, nostalgic documentary narrator "
             "with natural Turkish intonation; pause naturally at punctuation. Speak ONLY the transcript text, "
             "never these instructions.\n\nTRANSCRIPT:")
RATE = os.environ.get("VOICE_RATE", "-4%")
GAP = 0.45          # sahneler arası nefes
XFADE = 0.5         # sahne geçişi
CHANNEL = "Geçmişte Sen"

STYLE = ("retro vintage storybook illustration, old-fashioned, period accurate details, faded old color film look, "
         "warm sepia tones, soft grain, painterly, wide shot")
# CLIP ~77 tokende keser: en önemli yasaklar başta
NEG = ("flat screen tv, widescreen tv, lcd tv, air conditioner, smartphone, laptop, people, person, face, child, "
       "text, letters, watermark, logo, modern, skyscraper, new car, rabbit, animal, photorealistic, blurry, deformed")


def era(script, i):
    """Sahnenin yılı: en son bölüm yılından (\"90'LAR\" gibi değerleri de çözer)."""
    for sc in reversed(script["scenes"][:i + 1]):
        m = re.search(r"(19|20)\d\d", sc.get("year", ""))
        if m:
            return int(m.group(0))
        if "90" in sc.get("year", ""):
            return 1995
    return None


def prompt(script, i):
    sc = script["scenes"][i]
    y = era(script, i)
    when = f"set in Turkey in the {y // 10 * 10}s, " if y else ""
    # SD1.5 metni ~77 tokende keser: dönem ve tarz başta, sahne sonra
    return f"{when}retro vintage storybook illustration, {sc['bg']}, faded old color film look, {STYLE}"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def img_path(sc):
    return os.path.join(CACHE, f"bg_{h(sc['bg'] + STYLE)}.png")


# ------------------------------------------------------------------ görseller

def cmd_images(script, part, of):
    import torch
    from diffusers import StableDiffusionPipeline, LCMScheduler
    os.makedirs(CACHE, exist_ok=True)
    only = os.environ.get("ONLY")
    idx = [int(x) for x in only.split(",")] if only else [i for i in range(len(script["scenes"])) if i % of == part]
    todo = [i for i in idx if not os.path.exists(img_path(script["scenes"][i]))]
    if not todo:
        return
    pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8", torch_dtype=torch.float32, safety_checker=None)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    torch.set_num_threads(os.cpu_count() or 4)
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for i in todo:
        sc = script["scenes"][i]
        seed = int(h(sc["bg"]), 16) % 10 ** 6
        tv = "television" in sc["bg"]
        cands = [pipe(prompt(script, i), negative_prompt=NEG, num_inference_steps=6, guidance_scale=2.0,
                      width=1024, height=576, generator=torch.Generator().manual_seed(seed + k)).images[0]
                 for k in range(4 if tv else 2)]
        best = max(range(len(cands)), key=lambda k: period_score(clip, proc, cands[k], tv))
        cands[best].save(img_path(sc))
        print("image", os.path.basename(img_path(sc)), flush=True)


def period_score(clip, proc, im, tv):
    """Adaylar arasında dönemine en uygun görünen kareyi seçer (CLIP: eski - modern)."""
    import torch
    good = ["a faded old photograph of a 1980s room", "vintage retro objects from the past"]
    bad = ["a modern flat screen television", "an air conditioner on the wall", "a modern contemporary interior"]
    if tv:
        good.append("a bulky old CRT television set with knobs")
    inp = proc(text=good + bad, images=im, return_tensors="pt", padding=True)
    with torch.no_grad():
        p = clip(**inp).logits_per_image.softmax(-1)[0]
    return float(p[:len(good)].sum() - p[len(good):].sum())


# ------------------------------------------------------------------ ses

def voice_path(i):
    return os.path.join(CACHE, f"v_{i:03d}.wav")


def gemini_tts(text, voice, wav, style=TTS_STYLE):
    """Gemini TTS: 24 kHz mono PCM döner; kota/geçici hatada bekleyip yeniden dener."""
    import base64, time, urllib.request, urllib.error
    body = json.dumps({"contents": [{"parts": [{"text": f"{style}\n{text}"}]}],
                       "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                           "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{TTS_MODEL}:generateContent"
    for attempt in range(8):
        req = urllib.request.Request(url, body, {"Content-Type": "application/json",
                                                 "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                part = json.load(r)["candidates"][0]["content"]["parts"][0]
            pcm = base64.b64decode(part["inlineData"]["data"])
            break
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as e:
            msg = e.read().decode()[:1500] if isinstance(e, urllib.error.HTTPError) else str(e)
            print("gemini retry", attempt, msg, flush=True)
            if "PerDay" in msg:  # günlük kota bitti: beklemek boşuna
                raise SystemExit("gemini daily quota exhausted")
            time.sleep(min(90, 10 * 2 ** attempt))
    else:
        raise SystemExit("gemini tts failed")
    with wave.open(wav, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(pcm)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-af",
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse",
                    "-ar", str(SR), "-ac", "1", wav[:-4] + "_t.wav"], check=True)
    os.replace(wav[:-4] + "_t.wav", wav)
    return read_wav(wav)


def gemini_scenes(texts, voice, max_chars=900):
    """Ücretsiz kota günde az istek veriyor: birkaç sahne tek istekte okunur, ses paragraf
    aralarındaki en uzun sessizliklerden sahnelere bölünür."""
    groups, cur = [], []
    for i, t in enumerate(texts):
        if cur and sum(len(texts[j]) for j in cur) + len(t) > max_chars:
            groups.append(cur); cur = []
        cur.append(i)
    groups.append(cur)
    style = TTS_STYLE.replace("\n\nTRANSCRIPT:", " Leave a clear pause of about two seconds between paragraphs."
                              "\n\nTRANSCRIPT:")
    out = {}
    for g, idx in enumerate(groups):
        a = gemini_tts("\n\n".join(texts[i] for i in idx), voice, os.path.join(CACHE, f"g_{g:02d}.wav"), style)
        print("gemini group", g, len(idx), "scenes", round(len(a) / SR, 1), "s", flush=True)
        for i, piece in zip(idx, split_at_pauses(a, [len(texts[i]) for i in idx])):
            out[i] = piece
    return out


def split_at_pauses(a, lens):
    """Sesi len(lens) parçaya böler: her sınır, metin uzunluğuna göre beklenen yere en yakın uzun sessizlikte."""
    if len(lens) == 1:
        return [a]
    fr = SR // 50
    rms = np.sqrt(np.mean(a[:len(a) // fr * fr].reshape(-1, fr) ** 2, axis=1) + 1e-12)
    quiet = rms < max(rms.max() * 0.03, 1e-4)
    runs, k = [], 0
    while k < len(quiet):
        if quiet[k]:
            e = k
            while e < len(quiet) and quiet[e]:
                e += 1
            if e - k >= 12:  # >= 0.24 sn
                runs.append((k, e))
            k = e
        k += 1
    total = len(rms)
    cuts, used = [], set()
    for b in range(1, len(lens)):
        t = total * sum(lens[:b]) / sum(lens)
        win = max(200, total * 0.12)  # 4 sn ya da %12
        cand = [(e - k) - 0.03 * abs((k + e) / 2 - t) for k, e in runs]
        best = max((c, r) for c, r in zip(cand, runs) if abs((r[0] + r[1]) / 2 - t) < win and r not in used)
        used.add(best[1])
        cuts.append(int((best[1][0] + best[1][1]) / 2) * fr)
    pieces = np.split(a, sorted(cuts))
    out = []
    for p in pieces:  # baş/son sessizliği kırp
        loud = np.nonzero(np.abs(p) > 0.01)[0]
        out.append(p[max(0, loud[0] - SR // 50):loud[-1] + SR // 20] if len(loud) else p)
    return out


def tts(text, wav):
    """Tek cümleyi seslendirir, baş/son sessizliği kırpar."""
    if VOICE.startswith("gemini:"):
        return gemini_tts(text, VOICE[7:], wav)
    import edge_tts
    mp3 = wav[:-4] + ".mp3"
    for attempt in range(4):
        try:
            asyncio.run(edge_tts.Communicate(text, VOICE, rate=RATE).save(mp3))
            if os.path.getsize(mp3) > 1000:
                break
        except Exception as e:
            print("tts retry", e, flush=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3, "-af",
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                    "silenceremove=start_periods=1:start_threshold=-50dB,areverse",
                    "-ar", str(SR), "-ac", "1", wav], check=True)
    return read_wav(wav)


def voice_for(path):
    """bank/003_... -> 3. video -> ROTATION[(3-1) % 3]."""
    m = re.match(r"(\d+)", os.path.basename(path))
    return ROTATION[(int(m.group(1)) - 1) % len(ROTATION)] if m else ROTATION[0]


def cmd_voice(script):
    from say import chunks
    os.makedirs(CACHE, exist_ok=True)
    if VOICE.startswith("gemini:"):
        gem = gemini_scenes([" ".join(c for c, _ in chunks(sc["text"])) for sc in script["scenes"]], VOICE[7:])
    durs = []
    for i, sc in enumerate(script["scenes"]):
        parts = []
        if VOICE.startswith("gemini:"):
            parts = [gem[i], None]
        for k, (sent, gap) in enumerate([] if parts else chunks(sc["text"])):
            parts.append(tts(sent, os.path.join(CACHE, f"s_{i:03d}_{k}.wav")))
            parts.append(np.zeros(int(gap * SR), np.float32))
        a = np.concatenate(parts[:-1])  # son sessizliği sahne geçişi zaten veriyor
        with wave.open(voice_path(i), "w") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
        durs.append(len(a) / SR)
        print("voice", i, round(durs[-1], 2), flush=True)
    with open(os.path.join(CACHE, "durations.json"), "w") as f:
        json.dump(durs, f)


def read_wav(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def music(total, seed):
    """Sakin, nostaljik piyano benzeri arpej + yumuşak pad (koddan üretilir, telifsiz)."""
    rng = np.random.default_rng(seed)
    bpm = 72
    beat = 60 / bpm
    progs = [[(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)],
             [(48, 52, 55), (57, 60, 64), (53, 57, 60), (55, 59, 62)],
             [(50, 53, 57), (55, 59, 62), (48, 52, 55), (57, 60, 64)]]
    prog = progs[seed % len(progs)]
    n = int(total * SR) + SR
    out = np.zeros(n, np.float32)
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    bar = 4 * beat
    k = 0
    while True:
        ch = prog[k % 4]
        t0 = int(k * bar * SR)
        if t0 >= n:
            break
        L = int(bar * SR)
        t = np.arange(L) / SR
        pad = sum(np.sin(2 * np.pi * hz(m - 12) * t) for m in ch) / 3
        pad *= 0.05 * np.minimum(1, t / 0.8) * np.minimum(1, (bar - t) / 0.8)
        seg = pad.astype(np.float32)
        notes = [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[1] + 12, ch[0] + 24, ch[2] + 12, ch[1] + 12, ch[2] + 12]
        for j, m in enumerate(notes):
            s0 = int(j * beat / 2 * SR)
            tt = np.arange(int(1.6 * SR)) / SR
            f = hz(m)
            tone = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 2.6)
            tone *= 0.045 * (0.8 + 0.4 * rng.random())
            e = min(len(seg), s0 + len(tone))
            seg[s0:e] += tone[:e - s0]
        e = min(n, t0 + L)
        out[t0:e] += seg[:e - t0]
        k += 1
    return out[:int(total * SR)]


# ------------------------------------------------------------------ karakter

def hexrgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


AGE = {"kid": 0.36, "teen": 0.43, "adult": 0.48}


def draw_char(c, skin, hair, t, blink):
    """Sade çocuk/genç/yetişkin figürü (kanalın kendi karakteri). 2x çizilip küçültülür."""
    S = 2
    hpx = int(H * AGE.get(c.get("age", "kid"), 0.4)) * S
    w = int(hpx * 0.62)
    im = Image.new("RGBA", (w, hpx + 20 * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = w // 2
    ink = (40, 30, 30, 255)
    lw = max(3, hpx // 110)
    head_r = int(hpx * (0.2 if c.get("age") == "kid" else 0.17))
    head_cy = head_r + 6 * S
    body_top = head_cy + head_r - 4 * S
    leg_top = int(hpx * 0.68)
    shirt, pants = hexrgb(c.get("shirt", "#3A7BD5")), hexrgb(c.get("pants", "#2B2B2B"))
    # bacaklar + ayakkabı
    lgw = int(w * 0.16)
    for sx in (-1, 1):
        x0 = cx + sx * int(w * 0.11) - lgw // 2
        d.rounded_rectangle((x0, leg_top, x0 + lgw, hpx - 8 * S), lgw // 3, fill=pants + (255,), outline=ink, width=lw)
        d.ellipse((x0 - 6 * S + (4 * S if sx > 0 else 0), hpx - 22 * S, x0 + lgw + 10 * S, hpx + 4 * S), fill=(45, 35, 30, 255), outline=ink, width=lw)
    # kollar (hafif sallanma)
    sw = math.sin(t * 2.2) * 0.06
    bw = int(w * 0.5)
    for sx in (-1, 1):
        ax = cx + sx * (bw // 2 + int(w * 0.04))
        ang = sx * (0.12 + sw * sx)
        ex = ax + math.sin(ang) * hpx * 0.28
        ey = body_top + 10 * S + math.cos(ang) * hpx * 0.28
        d.line((ax, body_top + 14 * S, ex, ey), fill=ink, width=int(w * 0.13) + 2 * lw)
        d.line((ax, body_top + 14 * S, ex, ey), fill=shirt + (255,), width=int(w * 0.13))
        d.ellipse((ex - w * 0.06, ey - w * 0.06, ex + w * 0.06, ey + w * 0.06), fill=hexrgb(skin) + (255,), outline=ink, width=lw)
    # gövde
    d.rounded_rectangle((cx - bw // 2, body_top, cx + bw // 2, leg_top + 10 * S), int(bw * 0.28), fill=shirt + (255,), outline=ink, width=lw)
    if c.get("collar"):
        d.polygon([(cx - bw * 0.36, body_top + 2 * S), (cx, body_top + bw * 0.32), (cx + bw * 0.36, body_top + 2 * S),
                   (cx + bw * 0.18, body_top - 2 * S), (cx - bw * 0.18, body_top - 2 * S)], fill=(250, 250, 250, 255), outline=ink)
    # baş
    sk = hexrgb(skin) + (255,)
    d.ellipse((cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r), fill=sk, outline=ink, width=lw)
    hr = hexrgb(hair) + (255,)
    d.chord((cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r * 0.9), 180, 360, fill=hr, outline=ink, width=lw)
    d.ellipse((cx - head_r * 0.2, head_cy - head_r * 1.08, cx + head_r * 0.55, head_cy - head_r * 0.62), fill=hr)
    ey = head_cy + head_r * 0.12
    for sx in (-1, 1):
        ex = cx + sx * head_r * 0.38
        if blink:
            d.line((ex - head_r * 0.1, ey, ex + head_r * 0.1, ey), fill=ink, width=lw)
        else:
            d.ellipse((ex - head_r * 0.09, ey - head_r * 0.13, ex + head_r * 0.09, ey + head_r * 0.13), fill=ink)
            d.ellipse((ex - head_r * 0.02, ey - head_r * 0.09, ex + head_r * 0.04, ey - head_r * 0.03), fill=(255, 255, 255, 255))
        d.line((ex - head_r * 0.13, ey - head_r * 0.28, ex + head_r * 0.11, ey - head_r * 0.31), fill=ink, width=lw)
        d.ellipse((ex + sx * head_r * 0.08 - head_r * 0.12, ey + head_r * 0.2, ex + sx * head_r * 0.08 + head_r * 0.12,
                   ey + head_r * 0.34), fill=(240, 140, 130, 110))
    d.arc((cx - head_r * 0.22, ey + head_r * 0.12, cx + head_r * 0.22, ey + head_r * 0.45), 20, 160, fill=ink, width=lw)
    return im.resize((im.width // S, im.height // S), Image.LANCZOS)


# ------------------------------------------------------------------ metin

FONT = os.path.join(HERE, "fonts", "Poppins-SemiBold.ttf")
SERIF = os.path.join(HERE, "fonts", "Merriweather-Black.ttf")
_f = {}


def font(path, sz, weight=None):
    k = (path, sz)
    if k not in _f:
        f = ImageFont.truetype(path, sz)
        if weight:
            try:
                f.set_variation_by_axes([weight] + [None] * 0)
            except Exception:
                try:
                    f.set_variation_by_name("Black")
                except Exception:
                    pass
        _f[k] = f
    return _f[k]


def serif(sz):
    k = ("serif", sz)
    if k not in _f:
        f = ImageFont.truetype(SERIF, sz)
        try:
            axes = f.get_variation_axes()
            vals = []
            for a in axes:
                name = a.get("name", b"")
                name = name.decode() if isinstance(name, bytes) else str(name)
                vals.append(a["maximum"] if "eight" in name else a["default"])
            f.set_variation_by_axes(vals)
        except Exception:
            pass
        _f[k] = f
    return _f[k]


def wrap(text, f, maxw, d):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=f) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def sentences(text):
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    out = []
    for p in parts:
        if out and len(p) < 18:
            out[-1] += " " + p
        else:
            out.append(p)
    return out


# ------------------------------------------------------------------ render

def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def timeline(script, durs):
    t, tl = 0.6, []
    for i, (sc, dv) in enumerate(zip(script["scenes"], durs)):
        lead = 1.2 if sc.get("chapter") else 0.25
        t0 = t
        tl.append(dict(i=i, t0=t0, v0=t0 + lead, v1=t0 + lead + dv, t1=t0 + lead + dv + GAP + (0.6 if sc.get("chapter") else 0)))
        t = tl[-1]["t1"]
    return tl, t + 5.5  # son: abone kartı


def prep_bg(sc):
    if not os.path.exists(img_path(sc)):   # görsel eksikse video yine çıksın: sıcak degrade arka plan
        print("WARNING missing image for scene:", sc["bg"][:60], flush=True)
        g = np.linspace(0, 1, H)[:, None, None]
        arr = (np.array([70, 55, 45]) * (1 - g) + np.array([190, 150, 110]) * g).astype(np.uint8)
        im = Image.fromarray(np.repeat(arr, W, axis=1))
    else:
        im = Image.open(img_path(sc)).convert("RGB")
    im = im.resize((int(W * 1.12), int(H * 1.12)), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 60, 2))
    return vintage(im, int(h(sc["bg"]), 16))


def vintage(im, seed):
    """Eski fotoğraf/film hissi: soluk renk, sıcak ton, kalkık siyahlar, hafif gren ve kenar kararması."""
    a = np.asarray(im).astype(np.float32) / 255
    gray = a.mean(axis=2, keepdims=True)
    a = gray + (a - gray) * 0.72                      # doygunluk -%28
    a = a * np.array([1.06, 1.0, 0.86]) + np.array([0.03, 0.02, 0.0])   # sıcak/sepya
    a = 0.06 + a * 0.9                                # siyahlar kalkık (soluk film)
    rng = np.random.default_rng(seed % 2 ** 32)
    a += rng.normal(0, 0.025, a.shape[:2])[..., None]  # gren
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    r = np.hypot((xx - a.shape[1] / 2) / (a.shape[1] / 2), (yy - a.shape[0] / 2) / (a.shape[0] / 2))
    a *= (1 - 0.35 * np.clip(r - 0.55, 0, 1) ** 1.5)[..., None]
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def bg_frame(bg, u, k):
    z = 1.0 + 0.07 * u
    cw, ch = bg.width / z / 1.12 * 1.0, bg.height / z / 1.12
    cw, ch = W * bg.width / (W * 1.12) / z, H * bg.height / (H * 1.12) / z
    dx = (bg.width - cw) * (0.5 + (0.35 if k % 2 else -0.35) * (u - 0.5))
    dy = (bg.height - ch) * 0.5
    return bg.resize((W, H), Image.BILINEAR, box=(dx, dy, dx + cw, dy + ch))


def cmd_render(script):
    durs = json.load(open(os.path.join(CACHE, "durations.json")))
    tl, total = timeline(script, durs)
    char_cfg = script.get("character", {})
    skin, hair = char_cfg.get("skin", "#F2C9A0"), char_cfg.get("hair", "#4A2C1A")
    os.makedirs(OUT, exist_ok=True)
    # ses
    vo = np.zeros(int(total * SR) + SR, np.float32)
    duck = np.ones_like(vo)
    for seg in tl:
        x = read_wav(voice_path(seg["i"]))
        a = int(seg["v0"] * SR)
        vo[a:a + len(x)] += x
        duck[max(0, a - SR // 4):a + len(x) + SR // 4] = 0.45
    k = np.ones(int(0.3 * SR)) / int(0.3 * SR)
    duck = np.convolve(duck, k, mode="same")
    mus = music(total, int(h(script["id"]), 16) % 1000)
    mix = vo[:len(mus)] * 1.0 + mus * duck[:len(mus)] * 1.6
    fade = np.minimum(1, (total - np.arange(len(mix)) / SR) / 2.5)
    mix *= np.clip(fade, 0, 1)
    raw = os.path.join(OUT, "mix_raw.wav")
    with wave.open(raw, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", str(SR),
                    os.path.join(OUT, "mix.wav")], check=True)

    # altyazı zamanlaması: cümleler sahnenin ses süresine karakter sayısıyla dağıtılır
    subs = []
    for seg in tl:
        ss = sentences(script["scenes"][seg["i"]]["text"])
        tot = sum(len(s) for s in ss)
        t = seg["v0"]
        for s in ss:
            d = (seg["v1"] - seg["v0"]) * len(s) / tot
            subs.append((t, t + d + 0.15, s))
            t += d

    bgs = {}
    def bg_for(i):
        if i not in bgs:
            if len(bgs) > 3:
                bgs.pop(min(bgs))
            bgs[i] = prep_bg(script["scenes"][i])
        return bgs[i]

    char_cache = {}
    def char_img(c, t):
        blink = (t % 3.7) < 0.12
        key = (json.dumps(c, sort_keys=True), blink, int(t * 10) % 63)
        if key not in char_cache:
            if len(char_cache) > 400:
                char_cache.clear()
            char_cache[key] = draw_char(c, skin, hair, int(t * 10) / 10, blink)
        return char_cache[key]

    def scene_frame(i, t):
        seg = tl[i]
        sc = script["scenes"][i]
        u = (t - seg["t0"]) / max(0.1, seg["t1"] - seg["t0"] + XFADE)
        fr = bg_frame(bg_for(i), max(0, min(1, u)), i).convert("RGBA")
        c = sc.get("char", {})
        pos = c.get("pos", "none")
        if pos in ("left", "right") and c.get("age") != "baby":
            ci = char_img(c, t)
            bob = int(math.sin(t * 2 * math.pi * 0.45) * 4)
            x = int(W * (0.17 if pos == "left" else 0.83)) - ci.width // 2
            y = H - 70 - ci.height + bob
            sh = Image.new("RGBA", (ci.width, 40), (0, 0, 0, 0))
            ImageDraw.Draw(sh).ellipse((ci.width * 0.15, 5, ci.width * 0.85, 35), fill=(0, 0, 0, 90))
            fr.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)), (x, H - 70 - 22))
            fr.alpha_composite(ci, (x, y))
        return fr

    def overlay(fr, t):
        d = ImageDraw.Draw(fr)
        # bölüm kartı (yıl + başlık)
        for seg in tl:
            sc = script["scenes"][seg["i"]]
            if sc.get("chapter") and seg["t0"] <= t < seg["t0"] + 3.2:
                a = ease((t - seg["t0"]) / 0.5) * (1 - ease((t - seg["t0"] - 2.6) / 0.6))
                if a <= 0:
                    continue
                yr = sc.get("year", "")
                box = Image.new("RGBA", (W, 330), (0, 0, 0, 0))
                bd = ImageDraw.Draw(box)
                bd.rectangle((0, 0, W, 330), fill=(15, 12, 10, int(150 * a)))
                if yr:
                    bd.text((W // 2, 120), yr, font=serif(150), fill=(245, 225, 180, int(255 * a)), anchor="mm",
                            stroke_width=4, stroke_fill=(30, 20, 10, int(255 * a)))
                bd.text((W // 2, 255), sc["chapter"], font=font(FONT, 58), fill=(255, 255, 255, int(255 * a)), anchor="mm")
                fr.alpha_composite(box, (0, int(H * 0.08)))
                d = ImageDraw.Draw(fr)
        # altyazı
        for (a0, a1, s) in subs:
            if a0 <= t < a1:
                f = font(FONT, 46)
                lines = wrap(s, f, W - 360, d)[:3]
                y = H - 70 - len(lines) * 62
                for j, ln in enumerate(lines):
                    d.text((W // 2, y + j * 62), ln, font=f, fill=(255, 255, 255), anchor="mm", stroke_width=5,
                           stroke_fill=(15, 12, 10))
                break
        return fr

    def outro(fr, t):
        u = ease((t - tl[-1]["t1"]) / 0.8)
        box = Image.new("RGBA", (W, H), (12, 10, 8, int(200 * u)))
        bd = ImageDraw.Draw(box)
        bd.text((W // 2, H // 2 - 90), CHANNEL, font=serif(110), fill=(245, 225, 180, int(255 * u)), anchor="mm")
        bd.text((W // 2, H // 2 + 40), "Geçmişi bir de sen yaşa.", font=font(FONT, 52), fill=(255, 255, 255, int(255 * u)), anchor="mm")
        bd.rounded_rectangle((W // 2 - 230, H // 2 + 120, W // 2 + 230, H // 2 + 210), 45, fill=(200, 40, 40, int(255 * u)))
        bd.text((W // 2, H // 2 + 165), "ABONE OL", font=font(FONT, 50), fill=(255, 255, 255, int(255 * u)), anchor="mm")
        fr.alpha_composite(box)
        return fr

    nf = int(total * FPS)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", os.path.join(OUT, "mix.wav"), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", os.path.join(OUT, "video.mp4")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cur = 0
    for f in range(nf):
        t = f / FPS
        while cur < len(tl) - 1 and t >= tl[cur]["t1"]:
            cur += 1
        if t >= tl[-1]["t1"]:
            fr = outro(scene_frame(len(tl) - 1, t), t)
        else:
            fr = scene_frame(cur, t)
            nxt = cur + 1
            if nxt < len(tl) and t >= tl[cur]["t1"] - XFADE:
                a = (t - (tl[cur]["t1"] - XFADE)) / XFADE
                fr = Image.blend(fr, scene_frame(nxt, t), ease(a))
            fr = overlay(fr, t)
        p.stdin.write(fr.convert("RGB").tobytes())
        if f % 900 == 0:
            print(f"frame {f}/{nf}", flush=True)
    p.stdin.close()
    if p.wait():
        sys.exit("ffmpeg failed")
    thumbnail(script, skin, hair)
    meta = dict(title=script["title"], description=description(script, tl), tags=script.get("tags", []), duration=total,
                voice=open(os.path.join(CACHE, "voice.txt")).read() if os.path.exists(os.path.join(CACHE, "voice.txt")) else VOICE)
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done", round(total, 1), "s")


def description(script, tl):
    lines = [script["description"], "", "📖 BÖLÜMLER"]
    for seg in tl:
        sc = script["scenes"][seg["i"]]
        if sc.get("chapter"):
            t = 0 if seg["i"] == 0 else int(seg["t0"])
            lines.append(f"{t // 60:02d}:{t % 60:02d} {sc['chapter']}")
    lines += ["", f"🎥 {CHANNEL}: geçmişi senin gözünden anlatan resimli hikâyeler. Yeni bölümler için abone ol.", "",
              "#Nostalji #Tarih #GeçmişteSen"]
    return "\n".join(lines)


def thumbnail(script, skin, hair):
    th = script.get("thumb", {})
    sc = script["scenes"][th.get("bg", 0)]
    im = prep_bg(sc).resize((W, H)).convert("RGBA")
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle((0, 0, int(W * 0.58), H), fill=(10, 8, 6, 120))
    im.alpha_composite(shade.filter(ImageFilter.GaussianBlur(60)))
    c = dict(next((s["char"] for s in script["scenes"] if s.get("char", {}).get("pos") in ("left", "right")), {}))
    c["age"] = "adult"
    ci = draw_char(c, skin, hair, 0, False)
    ci = ci.resize((int(ci.width * 1.45), int(ci.height * 1.45)), Image.LANCZOS)
    im.alpha_composite(ci, (int(W * 0.72) - ci.width // 2, H - ci.height + 10))
    d = ImageDraw.Draw(im)
    d.text((90, 330), th.get("big", ""), font=serif(330), fill=(250, 235, 200), anchor="lm", stroke_width=10, stroke_fill=(25, 15, 8))
    d.text((100, 610), th.get("small", ""), font=serif(170), fill=(220, 50, 40), anchor="lm", stroke_width=9, stroke_fill=(25, 15, 8))
    im.convert("RGB").resize((1280, 720), Image.LANCZOS).save(os.path.join(OUT, "thumb.jpg"), quality=92)


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    s = load(path)
    if os.environ.get("QUICK"):   # duman testi: ilk 3 sahne (CI her push'ta çalıştırır)
        s["scenes"] = s["scenes"][:3]
    if cmd == "images":
        a = sys.argv
        cmd_images(s, int(a[a.index("--part") + 1]), int(a[a.index("--of") + 1]))
    elif cmd == "voice":
        VOICE = VOICE or voice_for(path)
        print("narrator", VOICE, flush=True)
        os.makedirs(CACHE, exist_ok=True)
        open(os.path.join(CACHE, "voice.txt"), "w").write(VOICE)
        cmd_voice(s)
    elif cmd == "render":
        cmd_render(s)
