"""把 docs/手册.md 渲染成 Word 手册（python-docx）。

只处理手册里实际用到的 Markdown 子集：# 标题、表格、代码块、列表、引用、段落。
中文用微软雅黑并显式设置 eastAsia 字体，代码用 Consolas。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "手册.md"
OUT = ROOT / "docs" / "BLOCKWORLD-中文手册.docx"

CJK = "微软雅黑"
MONO = "Consolas"


def set_cjk(run, font: str = CJK) -> None:
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), font)
    rfonts.set(qn("w:ascii"), font)
    rfonts.set(qn("w:hAnsi"), font)


def shade(paragraph, hex_color: str = "F2F3F5") -> None:
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    ppr.append(shd)


INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")


def add_rich_text(paragraph, text: str, *, mono_inline: bool = True) -> None:
    """处理 **粗体** 与 `行内代码`。"""
    tokens = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            run = paragraph.add_run(tok[2:-2])
            run.bold = True
            set_cjk(run)
        elif tok.startswith("`") and tok.endswith("`") and mono_inline:
            run = paragraph.add_run(tok[1:-1])
            run.font.size = Pt(9.5)
            set_cjk(run, MONO)
        else:
            run = paragraph.add_run(tok)
            set_cjk(run)


def parse_table(lines: list[str], i: int) -> tuple[list[list[str]], int]:
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        raw = lines[i].strip().strip("|")
        cells = [c.strip() for c in raw.split("|")]
        if not all(set(c) <= set("-: ") and c for c in cells):
            rows.append(cells)
        i += 1
    return rows, i


def build() -> Path:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    normal = doc.styles["Normal"]
    normal.font.name = CJK
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.25

    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
        st = doc.styles[name]
        st.font.name = CJK
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)

    i = 0
    first_h1 = True
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            if block:
                for ln in block:
                    p = doc.add_paragraph()
                    p.paragraph_format.left_indent = Cm(0.5)
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.0
                    run = p.add_run(ln if ln.strip() else " ")
                    run.font.size = Pt(9)
                    set_cjk(run, MONO)
                    shade(p)
                doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue

        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                cols = max(len(r) for r in rows)
                table = doc.add_table(rows=len(rows), cols=cols)
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for r, row in enumerate(rows):
                    for c in range(cols):
                        cell = table.cell(r, c)
                        cell.text = ""
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_after = Pt(0)
                        add_rich_text(p, row[c] if c < len(row) else "")
                        for run in p.runs:
                            run.font.size = Pt(9)
                            if r == 0:
                                run.bold = True
                doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            p = doc.add_heading(level=0 if first_h1 else 1)
            run = p.add_run(title)
            set_cjk(run)
            run.font.size = Pt(22 if first_h1 else 16)
            first_h1 = False
            i += 1
            continue

        if stripped.startswith("## "):
            p = doc.add_heading(level=1)
            run = p.add_run(stripped[3:].strip())
            set_cjk(run)
            i += 1
            continue

        if stripped.startswith("### "):
            p = doc.add_heading(level=2)
            run = p.add_run(stripped[4:].strip())
            set_cjk(run)
            i += 1
            continue

        if stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            run = p.add_run(stripped.lstrip("> ").strip())
            run.italic = True
            run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            set_cjk(run)
            i += 1
            continue

        if stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_rich_text(p, stripped[2:].strip())
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            # 不用 List Number 样式：Word 里它会让编号跨列表连续（1,2,3…6,7）
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(f"{m.group(1)}. ")
            set_cjk(run)
            add_rich_text(p, m.group(2))
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            ppr = p._p.get_or_add_pPr()
            pbdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "4")
            bottom.set(qn("w:color"), "CCCCCC")
            pbdr.append(bottom)
            ppr.append(pbdr)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        p = doc.add_paragraph()
        add_rich_text(p, stripped)
        i += 1

    doc.core_properties.title = "BLOCKWORLD 中文手册"
    doc.core_properties.author = "Yudong Xu"
    doc.core_properties.language = "zh-CN"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"written: {out} ({out.stat().st_size} bytes)")
