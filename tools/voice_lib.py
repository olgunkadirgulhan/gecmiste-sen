"""Bloop Bonkers ses kütüphanesi (Gemini TTS, bir kez): replikler + anime ünlemleri.
İş başına 1 istek: JOBS="Puck-N,Puck-P,Leda-N,Leda-P,ex-Puck,ex-Leda,ex-Zephyr,ex-Fenrir" (ücretsiz kota ~10/gün).
Her parça Whisper ile doğrulanır; eşleşmeyen (ör. talimatı okuma) atılır.
Çıktı: out/<Ses>/lines/<durum>_<N|P>.wav, out/<Ses>/ex/<tür>_<n>.wav, out/manifest_<iş>.json"""
import difflib, json, os, re, sys, urllib.request, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make

SRC = 'https://api.github.com/repos/olgunkadirgulhan/bloop-bonkers/contents/voice_lines.py'  # raw CDN önbelleği gecikiyor
ns = {}
exec(urllib.request.urlopen(urllib.request.Request(SRC, headers={'Accept': 'application/vnd.github.raw'}), timeout=60).read().decode(), ns)
LINES, EXCLAIM = ns['LINES'], ns['EXCLAIM']
STYLE = {
    'N': ("Read the transcript below aloud as a friendly, cheerful cartoon character with a polite, calm energy. "
          "Each line is a separate reaction: leave a clear pause of about two seconds between lines. "
          "Speak ONLY the transcript lines, never these instructions.\n\nTRANSCRIPT:"),
    'P': ("Read the transcript below aloud as a hyperactive, chaotic, gleeful cartoon character, very over the top and "
          "funny, like an anime character losing control. Each line is a separate reaction: leave a clear pause of about "
          "two seconds between lines. Speak ONLY the transcript lines, never these instructions.\n\nTRANSCRIPT:"),
    'ex': ("Perform the transcript below as a cute, energetic anime cartoon character, very expressive. Each line is a "
           "separate exclamation: leave a clear pause of about two seconds between lines. Speak ONLY the transcript lines, "
           "never these instructions.\n\nTRANSCRIPT:"),
}
from faster_whisper import WhisperModel
ASR = WhisperModel('small', compute_type='int8')


def norm(t):
    return re.sub(r'[^a-z]', '', t.lower())


def segments(a, min_gap=0.9):  # satır arası ~1 sn, satır içi duraklar daha kısa
    fr = make.SR // 100
    rms = np.sqrt(np.mean(a[:len(a) // fr * fr].reshape(-1, fr) ** 2, axis=1))
    loud = rms > rms.max() * 0.04
    segs, k = [], 0
    while k < len(loud):
        if loud[k]:
            e = k
            while e < len(loud) and (loud[e] or (e + int(min_gap * 100) < len(loud) and loud[e:e + int(min_gap * 100)].any())):
                e += 1
            segs.append((k * fr, min(len(a), e * fr)))
            k = e
        else:
            k += 1
    return [(s, e) for s, e in segs if e - s > 0.12 * make.SR]


def heard(x):
    """Whisper'a 16 kHz float dizi verilir (dosya/av kütüphanesi gerekmez); kısa klibe yarım saniye sessizlik eklenir."""
    pad = np.zeros(make.SR // 2)
    y = np.concatenate([pad, x, pad])
    y = np.interp(np.arange(0, len(y), make.SR / 16000), np.arange(len(y)), y).astype(np.float32)
    s, _ = ASR.transcribe(y, language='en', without_timestamps=True)
    return ' '.join(t.text for t in s).strip()


def align(a, texts, thr):
    """Sıralı hizalama: her beklenen satır için sıradaki 1-2 parçadan (birleştirerek) en iyi eşleşeni seç."""
    segs = segments(a)
    if len(segs) == len(texts):  # parça sayısı tutuyorsa sıra yeterli; sadece talimat okuma gibi uzun parçaları ele
        out = {}
        for li, (s0, e0) in enumerate(segs):
            hd = heard(a[s0:e0])
            r = difflib.SequenceMatcher(None, norm(hd), norm(texts[li])).ratio()
            if len(hd.split()) <= len(texts[li].split()) + 4 and r >= thr * 0.6:
                out[li] = (a[max(0, s0 - 2000):e0 + 3000], hd, round(r, 2))
        return out
    if thr < 0.3:  # Japonca ünlemleri Whisper tanıyamıyor: en uzun n-1 sessizlikten böl, sadece uzun (talimat) parçaları ele
        bounds = sorted(sorted(((segs[i + 1][0] - segs[i][1], i) for i in range(len(segs) - 1)), reverse=True)[:len(texts) - 1],
                        key=lambda g: g[1])
        cut = [0] + [g[1] + 1 for g in bounds] + [len(segs)]
        out = {}
        for li in range(min(len(texts), len(cut) - 1)):
            s0, e0 = segs[cut[li]][0], segs[cut[li + 1] - 1][1]
            hd = heard(a[s0:e0])
            if len(hd.split()) <= 4 and (e0 - s0) < 4 * make.SR:
                out[li] = (a[max(0, s0 - 2000):e0 + 3000], hd, 0.0)
        return out
    texts_h = {}
    def h(i, j):
        if (i, j) not in texts_h: texts_h[(i, j)] = heard(a[segs[i][0]:segs[j][1]])
        return texts_h[(i, j)]
    out, p = {}, 0
    for li, t in enumerate(texts):
        best = (0, None)
        for i in range(p, min(len(segs), p + 4)):
            for j in (i, i + 1, i + 2):   # çılgın tonda replik 2-3 parçaya bölünebiliyor
                if j < len(segs):
                    r = difflib.SequenceMatcher(None, norm(h(i, j)), norm(t)).ratio()
                    if r > best[0]: best = (r, (i, j))
        if best[1] and best[0] >= thr:
            i, j = best[1]
            out[li] = (a[max(0, segs[i][0] - 2000):segs[j][1] + 3000], h(i, j), round(best[0], 2))
            p = j + 1
    return out


def save(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = make.strip_noise(x)
    x = np.interp(np.arange(0, len(x), make.SR / 44100), np.arange(len(x)), x)
    with wave.open(path, 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes((np.clip(x / (np.abs(x).max() + 1e-9) * 0.95, -1, 1) * 32767).astype(np.int16).tobytes())


MODELS = ['gemini-3.8-flash-tts', 'gemini-2.5-flash-preview-tts', 'gemini-3.1-flash-tts-preview', 'gemini-2.5-pro-preview-tts',
          'gemini-3.8-flash-lite-tts']
for job in filter(None, os.environ.get('JOBS', '').split(',')):
    prev_m = f'prev/manifest_{job}.json'
    prev = json.load(open(prev_m)) if os.path.exists(prev_m) else {}
    # önceki kliplerden yalnız güvenilir olanlar: replikte benzerlik ≥ 0.65, ünlemde sırayla eşleşmiş (ratio > 0)
    prev = {p: v for p, v in prev.items() if v.get('ratio', 0) >= (0.65 if not job.startswith('ex-') else 0.01)}
    if job.startswith('ex-'):
        voice, kind = job[3:], 'ex'
        items = [(k, n, t) for k, lst in EXCLAIM.items() for n, t in enumerate(lst, 1)
                 if f'out/{voice}/ex/{k}_{n}.wav' not in prev]
        texts = [t for _, _, t in items]; thr = 0.25
    else:
        voice, kind = job.split('-')
        keys = list(LINES)
        items = [(k, kind, LINES[k][0 if kind == 'N' else 1]) for k in keys
                 if f'out/{voice}/lines/{k}_{kind}.wav' not in prev]
        texts = [t for _, _, t in items]; thr = 0.65
    if not items:
        print(f'{job}: hepsi önceden hazır', flush=True); continue
    a = None
    while MODELS:   # her TTS modelinin ayrı ücretsiz kotası var: biri dolunca sıradakine geç
        make.TTS_MODEL = MODELS[0]
        try:
            a = make.gemini_tts('\n'.join(texts), voice, f'/tmp/{job}.wav', STYLE[kind]); break
        except SystemExit as e:
            print(f'{MODELS[0]}: {e} -> sonraki model', flush=True); MODELS.pop(0)
    if a is None:
        print('tüm modellerin kotası dolu', flush=True); break
    got = align(a, texts, thr)
    man = dict(prev)
    for li, (k, n, t) in enumerate(items):
        if li in got:
            x, hd, r = got[li]
            path = f'out/{voice}/ex/{k}_{n}.wav' if kind == 'ex' else f'out/{voice}/lines/{k}_{n}.wav'
            save(x, path); man[path] = dict(text=t, heard=hd, ratio=r)
    json.dump(man, open(f'out/manifest_{job}.json', 'w'), indent=1, ensure_ascii=False)
    print(f'{job}: +{len(man) - len(prev)}/{len(items)} new clips, total {len(man)}, audio {round(len(a) / make.SR)} s, model {make.TTS_MODEL}', flush=True)
