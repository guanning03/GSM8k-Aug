import json
from typing import Optional
from datasets import Dataset
from tqdm import tqdm

# ============ Answer Extraction Utility ============

try:
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not installed. Fallback extraction will not be available.")

# Extraction config - matches the reward scoring config
_PRED_EXTRACTION_CONFIG = (ExprExtractionConfig(), LatexExtractionConfig()) if MATH_VERIFY_AVAILABLE else None


def last_boxed_only_string(string: str) -> Optional[str]:
    """Find the last \\boxed{} or \\fbox{} in the string and return it."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> Optional[str]:
    """Remove the \\boxed{} wrapper and return the content inside."""
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left):]

    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left): -1]
    
    # Handle \fbox{} as well
    left = "\\fbox{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left): -1]
    
    return None


def extract_answer(text: str) -> str:
    """
    Extract the answer from a solution string.
    
    Uses a two-stage extraction strategy:
    1. First try extract_solution() which specifically looks for \\boxed{} content
    2. If that fails, fallback to math_verify.parse with the same extraction 
       config as reward scoring for consistency
    """
    if not text:
        return ""
    
    # Stage 1: Try extract_solution (specifically designed for \boxed{} format)
    try:
        boxed_str = last_boxed_only_string(text)
        if boxed_str is not None:
            ans = remove_boxed(boxed_str)
            if ans is not None:
                ans = str(ans).strip()
                if ans:
                    return ans
    except Exception:
        pass
    
    # Stage 2: Fallback to parse with reward scoring config
    if MATH_VERIFY_AVAILABLE:
        try:
            parsed = parse(text, _PRED_EXTRACTION_CONFIG)
            if parsed:
                # parse returns a list; take the last extracted expression
                # (typically the final answer in a solution)
                ans = str(parsed[-1]).strip()
                if ans:
                    return ans
        except Exception:
            pass
    
    return ""


# ============ Dataset Creation ============

def count_lines(files):
    """统计文件总行数"""
    total = 0
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            total += sum(1 for _ in f)
    return total

def create_dataset():
    files = [
        r"data/MuggleMATH/AugGSM8K_part1.jsonl",
        r"data/MuggleMATH/AugGSM8K_part2.jsonl"
    ]
    
    # 先统计总行数
    print("Counting total lines...")
    total_lines = count_lines(files)
    print(f"Total lines to process: {total_lines}")
    
    data = []
    skipped = 0
    
    with tqdm(total=total_lines, desc="Extracting answers") as pbar:
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        query = item.get('query', '')
                        response = item.get('response', '')
                        
                        # 提取answer
                        answer = extract_answer(response)
                        
                        if answer:  # 非空字符串
                            data.append({
                                'problem': query,
                                'answer': answer,
                                'datasource': 'gsm8k-mugglemath'
                            })
                        else:
                            skipped += 1
                    pbar.update(1)
    
    print(f"\nTotal valid samples: {len(data)}")
    print(f"Skipped samples (no answer found): {skipped}")
    
    # 创建HuggingFace Dataset
    dataset = Dataset.from_list(data)
    print(f"\nDataset created with {len(dataset)} rows")
    
    # 上传到HuggingFace
    print("\nUploading to HuggingFace...")
    dataset.push_to_hub("guanning-ai/gsm8k-mugglemath")
    print("Upload complete!")
    
    return dataset

if __name__ == "__main__":
    create_dataset()
