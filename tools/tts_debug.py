"""Gemini TTS ham yanıtını inceler: parça sayısı, mimeType, baş/son baytlar (cızırtı teşhisi)."""
import base64, json, os, urllib.request
body = json.dumps({"contents": [{"parts": [{"text": "Read aloud in Turkish.\n\nTRANSCRIPT:\nYıl bin dokuz yüz seksen beş. Kapı komşuları ziyarete geliyor."}]}],
                   "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                       "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Charon"}}}}}).encode()
url = f"https://generativelanguage.googleapis.com/v1beta/models/{os.environ.get('TTS_MODEL', 'gemini-3.8-flash-tts')}:generateContent"
r = json.load(urllib.request.urlopen(urllib.request.Request(url, body, {
    "Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]}), timeout=180))
parts = r["candidates"][0]["content"]["parts"]
print("parts", len(parts), [list(p.keys()) for p in parts])
for p in parts:
    if "inlineData" in p:
        d = base64.b64decode(p["inlineData"]["data"])
        print("mime", p["inlineData"].get("mimeType"), "bytes", len(d))
        print("head", d[:64])
        print("tail", d[-96:])
    else:
        print("text part", str(p)[:200])
print("finish", r["candidates"][0].get("finishReason"), "usage", r.get("usageMetadata"))
