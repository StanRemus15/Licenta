import io
from PIL import Image
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)

def imagine_bytes(culoare, dim=(300,300)):
    img = Image.new('RGB', dim, culoare)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

def test_fisier_corupt():
    raspuns = client.post("/diagnostic",files={"file": ("test.jpg", io.BytesIO(b"This is not an image"), "image/jpeg")})
    assert raspuns.status_code == 200
    assert "corrupted" in raspuns.json()["eroare"]

def test_imagine_prea_mica():
    raspuns = client.post("/diagnostic",files={"file": ("mica.jpg", imagine_bytes((34, 139, 34), dim=(100, 100)), "image/jpeg")})
    assert "too small" in raspuns.json()["eroare"]

def test_imagine_fara_frunza():
    raspuns = client.post("/diagnostic",files={"file": ("gri.jpg", imagine_bytes((128, 128, 128)), "image/jpeg")})
    assert "leaf" in raspuns.json()["eroare"]

def test_suprafata_verde_uniforma():
    raspuns = client.post("/diagnostic", files={"file": ("perete.jpg", imagine_bytes((40, 160, 60)), "image/jpeg")})
    assert "eroare" in raspuns.json()