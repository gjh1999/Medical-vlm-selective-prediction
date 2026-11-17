"""Render the local research plan with Python standard library only."""
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "研究计划.md"
TARGET = ROOT / "研究计划.html"


def inline(text):
    text = html.escape(text, quote=True)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
                  r'<a href="\2" target="_blank" rel="noopener noreferrer">\1 ↗</a>', text)
    return text


def render(source):
    lines = source.splitlines()
    out, toc, paragraph = [], [], []
    i = 0

    def flush():
        if paragraph:
            out.append("<p>" + inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
            continue
        if line.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush()
            level, text = len(heading[1]), heading[2]
            anchor = f"section-{i}"
            out.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            if level == 2:
                toc.append(f'<a href="#{anchor}">{inline(text)}</a>')
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|\-]+\|$", lines[i+1]):
            flush()
            cells = lambda row: [inline(c.strip()) for c in row.strip().strip("|").split("|")]
            out.append('<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{c}</th>' for c in cells(line)) + '</tr></thead><tbody>')
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                out.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells(lines[i])) + '</tr>')
                i += 1
            out.append('</tbody></table></div>')
            continue
        item = re.match(r"^(?:([-])\s+|(\d+)\.\s+)(.+)$", line)
        if item:
            flush()
            tag = 'ul' if item[1] else 'ol'
            out.append(f'<{tag}>')
            while i < len(lines):
                match = re.match(r"^(?:([-])\s+|(\d+)\.\s+)(.+)$", lines[i])
                if not match or ('ul' if match[1] else 'ol') != tag:
                    break
                out.append('<li>' + inline(match[3]) + '</li>')
                i += 1
            out.append(f'</{tag}>')
            continue
        paragraph.append(line)
        i += 1
    flush()
    return '\n'.join(out), '\n'.join(toc)


body, toc = render(SOURCE.read_text(encoding='utf-8'))
page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>医学 VLM 层级分类与拒识研究计划</title><style>
:root{--ink:#213b33;--green:#28674c;--line:#d6dfd2}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:28px}body{margin:0;background:#f3f3ec;color:var(--ink);font:15px/1.9 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}header{background:var(--ink);color:#edf3e2;padding:24px max(5%,calc((100vw - 1280px)/2));display:flex;justify-content:space-between;align-items:center;gap:20px}header strong{font-size:13px;letter-spacing:1px}header span{font-size:11px;color:#c4d4c0}.actions{display:flex;gap:18px;align-items:center}.actions a{color:#e2edbe;font-size:12px}button{cursor:pointer;background:#d5e990;color:var(--ink);border:0;border-radius:3px;padding:10px 15px;font:inherit;font-size:12px}main{max-width:1280px;margin:35px auto;display:grid;grid-template-columns:230px minmax(0,1fr);gap:40px;padding:0 25px}nav{position:sticky;top:25px;align-self:start;background:#e7edde;padding:20px;border-radius:4px}nav strong{font-size:12px;display:block;margin-bottom:10px}nav a{display:block;font-size:12px;padding:5px 0;text-decoration:none}article{background:#fffef9;padding:40px 45px;border:1px solid var(--line);border-radius:4px;min-width:0}h1{font-size:30px;line-height:1.5;margin:0 0 20px}h2{font-size:24px;border-top:1px solid var(--line);padding-top:32px;margin-top:40px}h3{font-size:18px;margin-top:28px}p{margin:14px 0}a{color:var(--green);text-underline-offset:3px;overflow-wrap:anywhere}a:focus-visible,button:focus-visible{outline:3px solid #93ad65;outline-offset:4px}.table-wrap{overflow:auto;margin:22px 0;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;min-width:620px;font-size:12px;line-height:1.8}th,td{border-bottom:1px solid var(--line);padding:13px 15px;vertical-align:top;text-align:left}th{background:#e9efdf}tr:last-child td{border-bottom:0}li{margin:8px 0}pre{overflow:auto;background:#edf0e5;padding:20px;font-size:12px}code{font-family:Consolas,monospace;font-size:.9em}footer{font-size:11px;color:#647365;border-top:1px solid var(--line);padding-top:20px;margin-top:30px}@media(max-width:900px){main{grid-template-columns:1fr;gap:20px}nav{position:static;display:flex;gap:12px;flex-wrap:wrap}nav a{font-size:11px}nav strong{flex-basis:100%}article{padding:25px}h1{font-size:26px}}@media(max-width:560px){header{flex-direction:column;align-items:start}main{padding:0 12px;margin-top:18px}article{padding:20px 15px}h1{font-size:23px}h2{font-size:21px}body{font-size:14px}}@media print{body{background:white;font-size:10pt}header,nav{display:none}main{display:block;padding:0;margin:0;max-width:none}article{border:0;padding:0}h1{font-size:22pt}h2{font-size:17pt;break-after:avoid;padding-top:15px;margin-top:22px}h3{break-after:avoid}table{min-width:0;font-size:8pt}.table-wrap{overflow:visible}tr{break-inside:avoid}pre{white-space:pre-wrap}a{color:inherit}}
</style></head><body><header><div><strong>研究计划 · v1.0 · 2026-11-17</strong><br><span>有限校准数据 / 冻结视觉语言模型 / 分阶段验证</span></div><div class="actions"><a href="../../docs/vision-report/index.html">选题报告 ↗</a><a href="研究计划.md" download>下载 Markdown</a><button onclick="window.print()">打印 / 保存 PDF</button></div></header><main><nav aria-label="研究计划目录"><strong>研究计划目录</strong>__TOC__</nav><article>__BODY__<footer>本地研究计划；没有执行模型实验。参数、风险预算和阶段门均为研究设计，正式变更需更新协议版本。</footer></article></main></body></html>'''
TARGET.write_text(page.replace('__TOC__', toc).replace('__BODY__', body), encoding='utf-8')
print(f'Built {TARGET.name}; {len(toc.splitlines())} sections; {TARGET.stat().st_size} bytes')
