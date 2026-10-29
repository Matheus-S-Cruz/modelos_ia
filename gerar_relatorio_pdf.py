# ==============================================================================
# Gera o relatorio_final.pdf a partir do relatorio_final.md
# Uso: python gerar_relatorio_pdf.py
# ==============================================================================
import re
import xml.etree.ElementTree as ET

import markdown as md
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    ListFlowable,
    ListItem,
)

MD_FILE = "relatorio_final.md"
PDF_FILE = "relatorio_final.pdf"

AZUL = colors.HexColor("#1F4E79")
AZUL_CLARO = colors.HexColor("#D6E4F0")
CINZA = colors.HexColor("#F2F2F2")


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Titulo", parent=styles["Title"], textColor=AZUL,
        fontSize=18, leading=22, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="H1", parent=styles["Heading1"], textColor=AZUL,
        fontSize=14, leading=18, spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="H2", parent=styles["Heading2"], textColor=AZUL,
        fontSize=12, leading=15, spaceBefore=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="H3", parent=styles["Heading3"], textColor=colors.HexColor("#2E5C8A"),
        fontSize=11, leading=14, spaceBefore=8, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="Corpo", parent=styles["BodyText"], fontSize=10, leading=14,
        alignment=TA_LEFT, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Celula", parent=styles["BodyText"], fontSize=8.5, leading=11,
    ))
    styles.add(ParagraphStyle(
        name="CelulaHeader", parent=styles["BodyText"], fontSize=8.5, leading=11,
        textColor=colors.white, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="Quote", parent=styles["BodyText"], fontSize=10, leading=14,
        leftIndent=12, textColor=colors.HexColor("#444444"),
        borderColor=AZUL, borderWidth=0, spaceAfter=6,
    ))
    return styles


def inline(text):
    """Converte marcacao inline Markdown -> mini-HTML do reportlab."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def make_table(header, rows, styles):
    data = [[Paragraph(inline(c), styles["CelulaHeader"]) for c in header]]
    for r in rows:
        data.append([Paragraph(inline(c), styles["Celula"]) for c in r])
    t = Table(data, repeatRows=1, hAlign="LEFT")
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (-1, i), CINZA))
    t.setStyle(TableStyle(ts))
    return t


def parse_table_block(lines, idx):
    """A partir de um indice de linha que comeca uma tabela, retorna (header, rows, novo_idx)."""
    def split_row(line):
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [c.strip() for c in line.split("|")]

    header = split_row(lines[idx])
    idx += 2  # pula o separador |---|---|
    rows = []
    while idx < len(lines) and lines[idx].strip().startswith("|"):
        rows.append(split_row(lines[idx]))
        idx += 1
    return header, rows, idx


def main():
    with open(MD_FILE, encoding="utf-8") as f:
        raw = f.read()

    styles = build_styles()
    story = []
    lines = raw.splitlines()
    i = 0
    bullet_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            items = [ListItem(Paragraph(inline(b), styles["Corpo"]), leftIndent=10)
                     for b in bullet_buffer]
            story.append(ListFlowable(items, bulletType="bullet", start="•",
                                      leftIndent=14))
            story.append(Spacer(1, 2))
            bullet_buffer = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Bloco de codigo ``` ... ```
        if stripped.startswith("```"):
            flush_bullets()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code_text = "<br/>".join(
                l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                 .replace(" ", "&nbsp;")
                for l in code_lines
            )
            p = Paragraph(code_text, ParagraphStyle(
                "Code", fontName="Courier", fontSize=8.5, leading=11,
                backColor=CINZA, borderPadding=6, leftIndent=2, spaceAfter=8,
            ))
            story.append(p)
            continue

        # Tabela
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?$", lines[i + 1].strip()):
            flush_bullets()
            header, rows, i = parse_table_block(lines, i)
            story.append(Spacer(1, 2))
            story.append(make_table(header, rows, styles))
            story.append(Spacer(1, 8))
            continue

        # Linha horizontal
        if stripped in ("---", "***", "___"):
            flush_bullets()
            story.append(Spacer(1, 2))
            story.append(HRFlowable(width="100%", thickness=0.8, color=AZUL))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Titulos
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_bullets()
            level = len(m.group(1))
            text = inline(m.group(2))
            if level == 1:
                story.append(Paragraph(text, styles["Titulo"]))
            elif level == 2:
                story.append(Paragraph(text, styles["H1"]))
            elif level == 3:
                story.append(Paragraph(text, styles["H2"]))
            else:
                story.append(Paragraph(text, styles["H3"]))
            i += 1
            continue

        # Bullets (- ou *), com suporte a continuacao em multiplas linhas
        bm = re.match(r"^[-*]\s+(.*)$", stripped)
        if bm:
            item_text = [bm.group(1)]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if nxt == "" or re.match(r"^[-*]\s+", nxt) or nxt.startswith(("|", ">", "```", "#")) \
                        or nxt in ("---", "***", "___"):
                    break
                item_text.append(nxt)
                i += 1
            bullet_buffer.append(" ".join(item_text))
            continue

        # Blockquote
        if stripped.startswith(">"):
            flush_bullets()
            story.append(Paragraph(inline(stripped.lstrip("> ").strip()), styles["Quote"]))
            i += 1
            continue

        # Linha em branco
        if stripped == "":
            flush_bullets()
            i += 1
            continue

        # Paragrafo normal: acumula linhas consecutivas de texto simples
        # (hard-wrap do Markdown) ate encontrar uma linha em branco ou um
        # elemento de bloco, para nao quebrar marcacao inline (ex.: **...**).
        flush_bullets()
        buf = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if nxt == "":
                break
            if re.match(r"^(#{1,6})\s+", nxt) or nxt.startswith(("|", ">", "```", "- ", "* ")) \
                    or nxt in ("---", "***", "___"):
                break
            buf.append(nxt)
            i += 1
        story.append(Paragraph(inline(" ".join(buf)), styles["Corpo"]))

    flush_bullets()

    doc = SimpleDocTemplate(
        PDF_FILE, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Relatório Final - Otimização Pipeline MLOps",
    )
    doc.build(story)
    print(f"-> PDF gerado com sucesso: {PDF_FILE}")


if __name__ == "__main__":
    main()
