import re
from bs4 import BeautifulSoup
from typing import List, Dict, Set
from transformers import AutoTokenizer


def flatten_table(table_tag) -> List[str]:
    rows = table_tag.find_all('tr')
    if not rows: return []
    headers = []
    first_row = rows[0]
    cols = first_row.find_all(['th', 'td'])
    headers = [c.get_text(" ", strip=True) for c in cols]
    if not headers or all(h == '' for h in headers):
        headers = [f"Col_{i+1}" for i in range(len(cols))]
    start_idx = 1 if first_row.find('th') else 0
    flattened = []
    for row in rows[start_idx:]:
        cells = row.find_all(['td', 'th'])
        if not cells: continue
        parts = []
        for h, cell in zip(headers, cells):
            val = cell.get_text(" ", strip=True)
            if val: parts.append(f"{h}: {val}")
        if parts: flattened.append(" | ".join(parts))
    return flattened

def parse_confluence_logical_blocks(html_content: str, page_title: str = "") -> List[Dict]:
    soup = BeautifulSoup(html_content, 'html.parser')
    blocks = []
    current_section = "General Information" 
    for tag in soup.find_all(['script', 'style', 'noscript']): tag.decompose()
    target_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table', 'ac:structured-macro', 'pre', 'ul', 'ol']
    elements = soup.find_all(target_tags)
    processed: Set = set()

    for element in elements:
        if element in processed: continue
        
        # Header Detection
        is_header = False
        header_text = ""
        if re.match(r'^h[1-6]$', element.name):
            is_header = True
            header_text = element.get_text(" ", strip=True)
        elif element.name == 'p' and 'auto-cursor-target' in element.get('class', []):
            text = element.get_text(" ", strip=True)
            if text and len(text) < 150: is_header = True; header_text = text
        
        if is_header:
            if header_text: current_section = header_text
            processed.add(element)
            continue

        # Content Extraction
        block_data = None
        if element.name == 'table':
            rows = flatten_table(element)
            for r in rows:
                blocks.append({"text": r,"type": "table_row","metadata": {"source": page_title, "section": current_section}})
            processed.add(element); [processed.add(c) for c in element.descendants]; continue

        elif element.name in ['ul', 'ol']:
            for li in element.find_all('li', recursive=False):
                prefix = "- " if element.name == 'ul' else "1. "
                li_text = li.get_text(" ", strip=True)
                if len(li_text) > 3:
                    blocks.append({"text": f"{prefix}{li_text}","type": "list_item","metadata": {"source": page_title, "section": current_section}})
            processed.add(element); [processed.add(c) for c in element.descendants]; continue

        elif (element.name == 'ac:structured-macro' and element.get('ac:name') == 'code') or element.name == 'pre':
            c_text = element.get_text(strip=True) if element.name == 'pre' else (element.find('ac:plain-text-body').get_text(strip=True) if element.find('ac:plain-text-body') else "")
            if len(c_text) > 5:
                block_data = {"text": c_text,"type": "code","metadata": {"source": page_title, "section": current_section}}
            processed.add(element); 
            if element.name == 'ac:structured-macro': [processed.add(c) for c in element.descendants]

        elif element.name == 'p':
            p_text = element.get_text(" ", strip=True)
            if len(p_text) > 20:
                block_data = {"text": p_text,"type": "text","metadata": {"source": page_title, "section": current_section}}
            processed.add(element)

        if block_data: blocks.append(block_data)
    return blocks


class ConfluenceTokenChunker:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens - 2

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_large_text(self, text: str, metadata: Dict, chunk_type: str) -> List[Dict]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(token_ids)
        
        chunks = []
        overlap = int(self.max_tokens * 0.2) 
        stride = self.max_tokens - overlap
        
        for i in range(0, total_tokens, stride):
            chunk_ids = token_ids[i : i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            
            chunks.append({
                "text": chunk_text,
                "type": f"{chunk_type}_part", 
                "metadata": metadata,
                "token_count": len(chunk_ids)
            })
            if i + self.max_tokens >= total_tokens:
                break
        return chunks

    def chunk_html(self, html_content: str, page_title: str) -> List[Dict]:
        """
        """
        logical_blocks = parse_confluence_logical_blocks(html_content, page_title)
        
        final_chunks = []
        
        buffer_text = []
        current_buffer_tokens = 0
        current_section_context = None

        for block in logical_blocks:
            block_text = block['text']
            block_tokens = self.count_tokens(block_text)
            block_section = block['metadata']['section']

            section_changed = (current_section_context is not None) and (block_section != current_section_context)
            buffer_full = (current_buffer_tokens + block_tokens > self.max_tokens)

            if (section_changed or buffer_full) and buffer_text:
                combined_text = "\n".join(buffer_text)
                final_chunks.append({
                    "text": combined_text,
                    "type": "combined_text", # Loại hỗn hợp
                    "metadata": {"source": page_title, "section": current_section_context},
                    "token_count": current_buffer_tokens
                })
                buffer_text = []
                current_buffer_tokens = 0
                current_section_context = None

            if block_tokens > self.max_tokens:
                if buffer_text:
                     combined_text = "\n".join(buffer_text)
                     final_chunks.append({
                        "text": combined_text, "type": "combined_text",
                        "metadata": {"source": page_title, "section": current_section_context},
                        "token_count": current_buffer_tokens
                    })
                     buffer_text = []; current_buffer_tokens = 0
                
                large_chunks = self._split_large_text(block_text, block['metadata'], block['type'])
                final_chunks.extend(large_chunks)
                current_section_context = block_section

            else:
                buffer_text.append(block_text)
                current_buffer_tokens += block_tokens
                if current_section_context is None:
                    current_section_context = block_section

        if buffer_text:
            combined_text = "\n".join(buffer_text)
            final_chunks.append({
                "text": combined_text,
                "type": "combined_text",
                "metadata": {"source": page_title, "section": current_section_context},
                "token_count": current_buffer_tokens
            })
            
        return final_chunks

if __name__ == "__main__":
    long_code = "SELECT * FROM TBL_VERY_LONG_NAME WHERE " + " AND ".join([f"COL_{i} = 'VALUE_{i}'" for i in range(200)])
    
    html_sample = f"""
    <h1>Spec ETL DWH</h1>
    <p>Giới thiệu chung về quy trình.</p>
    
    <h2>1. Logic trích xuất (Extraction)</h2>
    <p>Đây là đoạn code SQL chính, nó rất dài và chắc chắn vượt quá 512 token.</p>
    <ac:structured-macro ac:name="code">
        <ac:plain-text-body>{long_code}</ac:plain-text-body>
    </ac:structured-macro>
    <p>Lưu ý: Cần chạy job này vào ban đêm.</p>

    <h2>2. Các bảng liên quan</h2>
    <ul>
        <li>Bảng nguồn: CORE_DB.TBL_A</li>
        <li>Bảng đích: DWH.TBL_A_FACT</li>
    </ul>
    """

    chunker = ConfluenceTokenChunker(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    final_chunks = chunker.chunk_html(html_sample, "Spec Page Title")
    
    
    for i, chunk in enumerate(final_chunks):
        print(chunk)