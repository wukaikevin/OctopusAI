#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert README.md to HTML with styling for print/PDF
"""

import markdown
from pathlib import Path

def markdown_to_html():
    """Convert README.md to styled HTML"""

    readme_path = Path('README.md')
    html_path = Path('README.html')

    print(f"Reading {readme_path}...")

    # Read markdown
    md_content = readme_path.read_text(encoding='utf-8')

    # Configure markdown extensions
    extensions = [
        'markdown.extensions.extra',
        'markdown.extensions.codehilite',
        'markdown.extensions.toc',
        'markdown.extensions.tables',
        'markdown.extensions.fenced_code'
    ]

    # Convert markdown to HTML
    html_body = markdown.markdown(md_content, extensions=extensions)

    # Create complete HTML document with print-friendly CSS
    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OctopusAI - 使用说明</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 2cm;
            }}

            body {{
                font-size: 11pt;
            }}

            h1, h2, h3, h4 {{
                page-break-after: avoid;
            }}

            table, figure, img {{
                page-break-inside: avoid;
            }}

            a {{
                text-decoration: none;
                color: #000;
            }}

            a[href^="http"]::after {{
                content: " (" attr(href) ")";
                font-size: 0.8em;
                color: #666;
            }}
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Microsoft YaHei', sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
        }}

        h1 {{
            font-size: 32px;
            font-weight: 700;
            color: #1a1a1a;
            margin-top: 0;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #1a1a1a;
        }}

        h2 {{
            font-size: 24px;
            font-weight: 700;
            color: #1a1a1a;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ccc;
        }}

        h3 {{
            font-size: 20px;
            font-weight: 600;
            color: #2a2a2a;
            margin-top: 30px;
            margin-bottom: 15px;
        }}

        h4 {{
            font-size: 16px;
            font-weight: 600;
            color: #3a3a3a;
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}

        ul, ol {{
            margin: 15px 0 15px 30px;
            padding: 0;
        }}

        li {{
            margin-bottom: 8px;
        }}

        a {{
            color: #0066cc;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        code {{
            background: #f0f0f0;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            color: #d32f2f;
        }}

        pre {{
            background: #1a1a1a;
            color: #f5f5f5;
            padding: 20px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 20px 0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
        }}

        pre code {{
            background: none;
            padding: 0;
            color: inherit;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        table th {{
            background: #1a1a1a;
            color: #fff;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
        }}

        table td {{
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
            color: #555;
        }}

        table tr:hover {{
            background: #f8f8f8;
        }}

        table tr:last-child td {{
            border-bottom: none;
        }}

        blockquote {{
            margin: 20px 0;
            padding: 15px 25px;
            border-left: 5px solid #666;
            background: #f8f8f8;
            color: #555;
        }}

        blockquote p {{
            margin: 0;
        }}

        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 40px 0;
        }}

        strong {{
            font-weight: 600;
            color: #1a1a1a;
        }}

        em {{
            font-style: italic;
        }}

        sup {{
            font-size: 0.7em;
            vertical-align: super;
            color: #999;
        }}

        .toc {{
            background: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 25px;
            margin: 30px 0;
        }}

        .toc h2 {{
            margin-top: 0;
            border-bottom: none;
        }}

        .toc ul {{
            list-style-type: none;
            margin-left: 0;
        }}

        .toc li {{
            margin-bottom: 8px;
        }}

        .toc a {{
            color: #0066cc;
            text-decoration: none;
            font-weight: 500;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}

        /* Print button */
        .print-btn {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            background: #1a1a1a;
            color: #fff;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }}

        .print-btn:hover {{
            background: #333;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}

        @media print {{
            .print-btn {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ 打印/PDF</button>
    {html_body}

    <script>
        // Auto-hide print button after printing
        window.onafterprint = function() {{
            console.log('Document printed');
        }};
    </script>
</body>
</html>
    """

    # Write HTML file
    html_path.write_text(html_template, encoding='utf-8')
    print(f"Successfully created {html_path}")
    print(f"\nTo create PDF:")
    print(f"1. Open {html_path} in your browser")
    print(f"2. Click the 'Print/PDF' button (top right)")
    print(f"3. Select 'Save as PDF' as the destination")
    print(f"4. Click Save")


if __name__ == '__main__':
    markdown_to_html()
