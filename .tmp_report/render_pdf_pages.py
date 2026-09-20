from pathlib import Path

import fitz


pdf_path = Path(r"E:\Code\arm_pracctice\.tmp_report\final_render\report.pdf")
out_dir = pdf_path.parent / "pages"
out_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(pdf_path)
for index, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
    pix.save(out_dir / f"page-{index + 1}.png")
print(f"rendered={len(doc)} dir={out_dir}")
