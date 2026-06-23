# Báo cáo GraphRAG Lab

## Cấu hình

- LLM: `accounts/fireworks/models/deepseek-v4-pro` qua Fireworks API
- Graph framework: NetworkX
- Dataset documents: 70
- Runtime this run: 979.2 seconds

## Indexing và Graph Construction

- Nodes: 1476
- Edges/triples: 1003
- Cached extraction files: `outputs/cache/extractions/`
- Graph file: `outputs/graph.graphml`
- Graph image: `E:\GitClone\2A202600865-LuongDinhBut-Day19\outputs\knowledge_graph.svg`

## Querying

- Flat RAG: truy xuất chunk văn bản bằng TF-IDF local, sau đó gửi context cho LLM.
- GraphRAG: LLM trích xuất focus entities từ câu hỏi, NetworkX tìm node gần nhất, duyệt 2-hop, textualize triples, sau đó gửi context graph cho LLM.

## Evaluation

- Questions evaluated: 20
- Winner counts: `{'graph': 11, 'flat': 8, 'tie': 1}`
- Bảng benchmark: `outputs/benchmark_table.md`

## Các trường hợp GraphRAG tốt hơn Flat RAG

- **q01**: Những yếu tố nào liên quan đến việc tăng trưởng doanh số xe điện tại Mỹ chậm lại trong Q1 2024? Graph advantage: Provides a complete, specific list of factors with precise numerical data and clear citations.
- **q06**: Những hãng xe nào được nhắc tới là có tăng trưởng doanh số EV hơn 50% so với cùng kỳ trong Q1 2024? Graph advantage: Compared citation count, answer coverage, and overlap with question terms.
- **q07**: Giá bán thấp hơn, ưu đãi và leasing đã tác động ra sao đến thị trường EV tại Mỹ? Graph advantage: Compared citation count, answer coverage, and overlap with question terms.
- **q10**: Vai trò của Trung Quốc trong thị trường EV và chuỗi cung ứng pin được nêu ra như thế nào? Graph advantage: Compared citation count, answer coverage, and overlap with question terms.
- **q12**: Những rào cản nào có thể khiến người mua trì hoãn chuyển từ xe động cơ đốt trong sang EV? Graph advantage: Compared citation count, answer coverage, and overlap with question terms.
- **q15**: Các nguồn trong corpus nói gì về vai trò của xe điện thương mại, xe buýt, xe hai bánh hoặc ba bánh? Graph advantage: Identified specific mention of Tesla Semi electric truck from doc_61
- **q16**: Những con số benchmark quan trọng nhất về thị phần, doanh số, hạ tầng sạc hoặc pin trong corpus là gì? Graph advantage: Compared citation count, answer coverage, and overlap with question terms.
- **q17**: So sánh các tín hiệu tích cực và tiêu cực về sentiment của ngành EV tại Mỹ. Graph advantage: Compared citation count, answer coverage, and overlap with question terms.

## Chi phí và token usage

- LLM calls this run: 87
- Prompt tokens: 124569
- Completion tokens: 50378
- Total tokens: 174947

Lưu ý: nếu một số extraction lấy từ cache, token của các lần chạy trước sẽ không được cộng vào số liệu trên.