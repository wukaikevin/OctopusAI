#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert README.md to README.pdf using fpdf2
"""

import re
from fpdf import FPDF
from pathlib import Path
from markdown2 import markdown

class PDFDocument(FPDF):
    """Custom PDF document class"""

    def __init__(self):
        super().__init__()
        self.title_font = 'Arial'
        self.heading_font = 'Arial'
        self.body_font = 'Arial'
        self.code_font = 'Courier'

        # Set margins (left, top, right)
        self.set_margins(20, 20, 20)
        # Set auto page break
        self.set_auto_page_break(auto=True, margin=20)

        # Colors
        self.title_color = (26, 26, 26)
        self.heading_color = (42, 42, 42)
        self.text_color = (51, 51, 51)
        self.code_bg_color = (240, 240, 240)
        self.code_color = (211, 47, 47)
        self.table_header_bg = (26, 26, 26)
        self.table_header_text = (255, 255, 255)

    def header(self):
        """Page header"""
        pass

    def footer(self):
        """Page footer"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title, level=1):
        """Add chapter title"""
        self.ln(10)
        self.set_text_color(*self.title_color if level == 1 else self.heading_color)

        if level == 1:
            self.set_font(self.title_font, 'B', 24)
            self.cell(0, 12, title, 0, 1, 'L')
            self.line(20, self.get_y(), 190, self.get_y())
        elif level == 2:
            self.set_font(self.heading_font, 'B', 18)
            self.cell(0, 10, title, 0, 1, 'L')
            self.line(20, self.get_y(), 190, self.get_y())
        elif level == 3:
            self.set_font(self.heading_font, 'B', 14)
            self.cell(0, 8, title, 0, 1, 'L')
        else:
            self.set_font(self.heading_font, 'B', 12)
            self.cell(0, 7, title, 0, 1, 'L')

        self.ln(5)
        self.set_text_color(*self.text_color)

    def chapter_body(self, text):
        """Add chapter body text"""
        self.set_font(self.body_font, '', 11)
        self.multi_cell(0, 6, text)
        self.ln()

    def add_code_block(self, code):
        """Add code block"""
        self.set_fill_color(*self.code_bg_color)
        self.set_text_color(*self.code_color)
        self.set_font(self.code_font, '', 9)
        self.ln(5)

        # Split code into lines
        lines = code.split('\n')
        for line in lines:
            self.cell(0, 5, line, 0, 1, 'L', fill=True)

        self.ln(5)
        self.set_text_color(*self.text_color)

    def add_list_item(self, text, indent=10):
        """Add list item"""
        self.set_font(self.body_font, '', 11)
        self.cell(indent, 6, chr(149), 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(2)


def parse_markdown_to_pdf(md_path, pdf_path):
    """Convert markdown to PDF"""

    # Read markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Create PDF
    pdf = PDFDocument()
    pdf.add_page()

    # Parse markdown content
    lines = md_content.split('\n')
    i = 0

    in_code_block = False
    code_lines = []
    in_list = False
    in_table = False

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                pdf.add_code_block('\n'.join(code_lines))
                code_lines = []
                in_code_block = False
            else:
                # Start code block
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Headers
        if line.startswith('#'):
            level = len(re.match(r'^#+', line).group())
            title = line.lstrip('#').strip()
            pdf.chapter_title(title, level)
            i += 1
            continue

        # Horizontal rule
        if line.strip() == '---':
            pdf.ln(10)
            i += 1
            continue

        # Lists
        if line.strip().startswith(('- ', '* ', '+ ')) or re.match(r'^\d+\.', line):
            item_text = re.sub(r'^[\s\-\*\+]+\d*\.\s*', '', line)
            if '**' in item_text:
                # Bold text
                parts = re.split(r'\*\*(.+?)\*\*', item_text)
                formatted_text = ''
                for j, part in enumerate(parts):
                    if j % 2 == 1:
                        formatted_text += part
                    else:
                        formatted_text += part
                pdf.add_list_item(formatted_text)
            else:
                pdf.add_list_item(item_text)
            i += 1
            continue

        # Tables
        if '|' in line and not in_table:
            in_table = True
            headers = [h.strip() for h in line.split('|')[1:-1]]

            # Skip separator line
            i += 1
            if i < len(lines) and lines[i].strip().startswith('|---'):
                i += 1

            # Add table
            pdf.set_font('Arial', 'B', 10)
            col_width = 170 / len(headers)

            # Header
            pdf.set_fill_color(*pdf.table_header_bg)
            pdf.set_text_color(*pdf.table_header_text)
            for h in headers:
                pdf.cell(col_width, 7, h, 1, 0, 'C', fill=True)
            pdf.ln()

            # Data rows
            pdf.set_font('Arial', '', 10)
            pdf.set_text_color(*pdf.text_color)
            while i < len(lines) and '|' in lines[i]:
                if lines[i].strip().startswith('|---'):
                    i += 1
                    continue
                cells = [c.strip() for c in lines[i].split('|')[1:-1]]
                fill = False
                for cell in cells:
                    pdf.cell(col_width, 6, cell, 1, 0, 'L', fill)
                    fill = not fill
                pdf.ln()
                i += 1

            in_table = False
            pdf.ln(5)
            continue

        # Blockquote
        if line.strip().startswith('>'):
            text = line.strip()[1:].strip()
            pdf.set_fill_color(248, 248, 248)
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 6, text, 0, 'L', fill=True)
            pdf.ln(3)
            i += 1
            continue

        # Empty line
        if not line.strip():
            pdf.ln(3)
            i += 1
            continue

        # Regular paragraph
        text = line.strip()
        # Handle inline code
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Handle bold
        text = text.replace('**', '')
        # Handle links
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

        if text:
            pdf.chapter_body(text)

        i += 1

    # Save PDF
    pdf.output(pdf_path)
    print(f"Successfully created {pdf_path}")


def main():
    """Main function"""
    md_path = Path('README.md')
    pdf_path = Path('README.pdf')

    if not md_path.exists():
        print(f"Error: {md_path} not found!")
        return

    print(f"Reading {md_path}...")
    print(f"Generating {pdf_path}...")

    try:
        parse_markdown_to_pdf(md_path, pdf_path)
        print(f"PDF generation completed!")
        print(f"File size: {pdf_path.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
