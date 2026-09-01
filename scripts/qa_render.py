import fitz
from pathlib import Path
out = Path("/data1/yansari/Compass/experiments/qa")
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.png"): f.unlink()
doc = fitz.open("/data1/yansari/Compass/COMPASS_results.pdf")
print("pages:", doc.page_count)
for i, page in enumerate(doc):
    r = page.rect
    print(i+1, f"{r.width:.0f}x{r.height:.0f}", "rot", page.rotation)
    pm = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
    pm.save(str(out / f"p{i+1:02d}.png"))
print("done")
