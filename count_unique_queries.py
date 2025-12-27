import json

def count_unique_queries():
    files = [
        r"data\MuggleMATH\AugGSM8K_part1.jsonl",
        r"data\MuggleMATH\AugGSM8K_part2.jsonl"
    ]
    
    all_queries = []  # 存储所有query（含重复）
    file_counts = []  # 每个文件的行数
    
    for file_path in files:
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    query = data.get('query', '')
                    all_queries.append(query)
                    count += 1
        file_counts.append(count)
    
    unique_queries = set(all_queries)
    total_lines = sum(file_counts)
    duplicates = total_lines - len(unique_queries)
    
    print(f"File 1: {files[0]} -> {file_counts[0]} lines")
    print(f"File 2: {files[1]} -> {file_counts[1]} lines")
    print(f"Total lines: {total_lines}")
    print(f"Unique queries: {len(unique_queries)}")
    print(f"Duplicate queries: {duplicates}")
    print(f"Duplicate rate: {duplicates / total_lines * 100:.4f}%")

if __name__ == "__main__":
    count_unique_queries()

