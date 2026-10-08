"""Seslendirme öncesi Türkçe metin düzeltme: sayılar yazıya, kısaltmalar harf harf, noktalamaya göre duraklar."""
import re

ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
LETTER = dict(zip("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
                  "a be ce çe de e fe ge yumuşak_ge he ı i je ke le me ne o ö pe re se şe te u ü ve ye ze".split()))
KEEP = {"NATO", "ASELSAN", "TOKİ", "ODTÜ"}  # kelime gibi okunanlar
PAUSE = {".": 0.45, "!": 0.45, "?": 0.5, "…": 0.6}  # cümle arası sessizlik; virgülde ses kendisi durur


def words(n):
    if n == 0:
        return "sıfır"
    out = []
    for size, name in ((10 ** 9, "milyar"), (10 ** 6, "milyon"), (1000, "bin")):
        q, n = divmod(n, size)
        if q:
            out.append(name if q == 1 and size == 1000 else f"{words(q)} {name}")
    h, n = divmod(n, 100)
    if h:
        out.append("yüz" if h == 1 else f"{ONES[h]} yüz")
    if n // 10:
        out.append(TENS[n // 10])
    if n % 10:
        out.append(ONES[n % 10])
    return " ".join(out)


def ordinal(n):
    w = words(n)
    v = [c for c in w if c in "aeıioöuü"][-1]
    suf = {"a": "ıncı", "ı": "ıncı", "e": "inci", "i": "inci", "o": "uncu", "u": "uncu", "ö": "üncü", "ü": "üncü"}[v]
    return w + (suf[1:] if w[-1] in "aeıioöuü" else suf)


def normalize(t):
    t = re.sub(r"%\s?(\d+)", lambda m: "yüzde " + words(int(m.group(1))), t)
    # "2. katta" -> ikinci (sayı+nokta+küçük harf = sıra sayısı); "1985. Türkiye" -> cümle sonu
    t = re.sub(r"\b(\d+)\.(?=\s+[a-zçğıöşü])", lambda m: ordinal(int(m.group(1))), t)
    t = re.sub(r"\b(\d{1,3}(?:\.\d{3})+|\d+)(?:'(\w+))?",
               lambda m: words(int(m.group(1).replace(".", ""))) + (m.group(2) or ""), t)
    t = re.sub(r"\b[A-ZÇĞİÖŞÜ]{2,5}\b",
               lambda m: m.group(0) if m.group(0) in KEEP else
               " ".join(LETTER.get(c, c) for c in m.group(0)).replace("_", " "), t)
    return re.sub(r"\s+", " ", t).strip()


def chunks(t):
    """Metni cümlelere böler: [(cümle, sonrasındaki sessizlik sn)]."""
    out = []
    for m in re.finditer(r"[^.!?…]+[.!?…]*", normalize(t)):
        s = m.group(0).strip()
        if not re.search(r"\w", s):
            continue
        p = s[-1] if s[-1] in PAUSE else "."
        out.append((("İ" if s[0] == "i" else s[0].upper()) + s[1:], PAUSE[p]))
    return out


if __name__ == "__main__":
    for s in ["Yıl 1985. Türkiye'de 4 katlı bir apartmanın 2. katında doğuyorsun.",
              "1990'da ilk özel kanal açılıyor, 80'lerde TRT vardı; 1999'da %50 arttı!", "2000'lere giden yol"]:
        print(chunks(s))
