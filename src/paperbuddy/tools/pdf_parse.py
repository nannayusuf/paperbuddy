from docling.document_converter import DocumentConverter
from pathlib import Path

def parse_pdf(pdf_path: str):
    """Extrai texto, tabelas e figuras de PDF"""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    doc = result.document
    
    return {
        "text": doc.export_to_markdown(),
        "tables": [t.export_to_html() for t in doc.tables],
        "figures": [{"image": f.image, "caption": f.caption} for f in doc.pictures]
    }