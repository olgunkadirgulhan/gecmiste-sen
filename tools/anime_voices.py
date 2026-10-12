"""Bloop Bonkers için anime tarzı ünlem kütüphanesi: ses başına 1 Gemini isteği, sessizliklerden bölünür.
Çıktı: anime/<ses>/<tür>_<n>.wav (44.1 kHz mono). Bir kez üretilir, bloop-bonkers reposuna eklenir."""
import os, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make

LINES = [('hm', 'Hmm?'), ('yell', 'Nani?!'), ('giggle', 'Hehehe!'), ('yay', 'Yatta!'), ('yell', 'Kyaaa!'),
         ('hm', 'Ehhh?'), ('evil', 'Mwahahaha!'), ('giggle', 'Fufufu...'), ('yay', 'Sugoi!'), ('yell', 'Uwaaah!'),
         ('talk', 'Itadakimasu!'), ('hm', 'Ara?'), ('talk', 'Hmph!'), ('evil', 'Kukuku...')]
STYLE = ("You are voice acting a cute, energetic anime cartoon character. Perform each short line below very "
         "expressively, exactly like an anime character would, with a clear pause of about one second between lines. "
         "Say ONLY the lines, nothing else.\n\nLINES:")
os.makedirs('anime', exist_ok=True)


def islands(a, n):
    """Sesi n parçaya böler: sessizlik eşiğini, tam n ses adacığı kalana kadar ayarlar; olmazsa en uzun n-1 boşluktan keser."""
    fr = make.SR // 100
    rms = np.sqrt(np.mean(a[:len(a) // fr * fr].reshape(-1, fr) ** 2, axis=1))
    quiet = rms < rms.max() * 0.04
    gaps = []  # (uzunluk, başlangıç, bitiş) kare
    k = 0
    while k < len(quiet):
        if quiet[k]:
            e = k
            while e < len(quiet) and quiet[e]: e += 1
            if k > 0 and e < len(quiet): gaps.append((e - k, k, e))
            k = e
        else: k += 1
    cuts = sorted(sorted(gaps, reverse=True)[:n - 1], key=lambda g: g[1])
    bounds = [0] + [((g[1] + g[2]) // 2) * fr for g in cuts] + [len(a)]
    return [a[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

for voice in os.environ.get('VOICES', 'Leda,Zephyr,Puck,Fenrir').split(','):
    texts = [t for _, t in LINES]
    for attempt in range(2):
        a = make.gemini_tts('\n\n'.join(texts), voice, f'anime/{voice}.wav', STYLE)
        if not make.bursts(a): break
    pieces = islands(a, len(texts))
    d = f'anime/{voice}'; os.makedirs(d, exist_ok=True)
    count = {}
    for (kind, _), p in zip(LINES, pieces):
        p = make.strip_noise(p)
        x = np.interp(np.arange(0, len(p), make.SR / 44100), np.arange(len(p)), p)  # 48k -> 44.1k
        count[kind] = count.get(kind, 0) + 1
        with wave.open(f'{d}/{kind}_{count[kind]}.wav', 'w') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    os.remove(f'anime/{voice}.wav')
    print('ok', voice, len(pieces), 'gaps', flush=True) if False else None
    print('ok', voice, len(pieces), [round(len(p) / make.SR, 2) for p in pieces], flush=True)
