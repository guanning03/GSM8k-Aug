import re
from typing import Optional
from datasets import Dataset, load_dataset
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


def extract_answer_from_text(text: str) -> Optional[str]:
    """
    Extract the answer from 'The answer is: XXX' format.
    """
    if not text:
        return None
    
    patterns = [
        r"[Tt]he answer is[:\s]+([^\n]+?)\.?$",
        r"[Tt]he answer is[:\s]+([^\n]+)",
        r"#### ([^\n]+?)\.?$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            answer = match.group(1).strip()
            answer = answer.rstrip('.')
            if answer:
                return answer
    
    return None


def extract_answer(text: str) -> str:
    """
    Extract the answer from a solution string.
    """
    if not text:
        return ""
    
    # Stage 1: Try extracting from "The answer is:" format
    try:
        ans = extract_answer_from_text(text)
        if ans:
            return ans.strip()
    except Exception:
        pass
    
    # Stage 2: Try extract_solution (for \boxed{} format)
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
    
    # Stage 3: Fallback to parse with reward scoring config
    if MATH_VERIFY_AVAILABLE:
        try:
            parsed = parse(text, _PRED_EXTRACTION_CONFIG)
            if parsed:
                ans = str(parsed[-1]).strip()
                if ans:
                    return ans
        except Exception:
            pass
    
    return ""


# ============ Dataset Creation ============

def create_dataset():
    print("正在下载 meta-math/MetaMathQA 数据集...")
    dataset = load_dataset('meta-math/MetaMathQA')
    
    print(f"\n数据集结构: {dataset}")
    print(f"数据集splits: {list(dataset.keys())}")
    
    # 获取train split
    train_data = dataset['train']
    print(f"\n原始数据集大小: {len(train_data)}")
    print(f"字段: {train_data.column_names}")
    
    # 筛选: type以GSM_开头，但不是GSM_AnsAug
    data = []
    skipped_not_gsm = 0
    skipped_gsm_ansaug = 0
    skipped_no_answer = 0
    
    for item in tqdm(train_data, desc="Processing"):
        item_type = item.get('type', '')
        
        # 筛选条件: type以GSM_开头，但不是GSM_AnsAug
        if not item_type.startswith('GSM_'):
            skipped_not_gsm += 1
            continue
        
        if item_type == 'GSM_AnsAug':
            skipped_gsm_ansaug += 1
            continue
        
        query = item.get('query', '')
        response = item.get('response', '')
        
        # 提取answer
        answer = extract_answer(response)
        
        if answer:
            data.append({
                'problem': query,
                'answer': answer,
                'datasource': 'gsm8k-metamath',
                'type': item_type
            })
        else:
            skipped_no_answer += 1
    
    print(f"\n筛选统计:")
    print(f"  - 跳过 (非GSM_开头): {skipped_not_gsm}")
    print(f"  - 跳过 (GSM_AnsAug): {skipped_gsm_ansaug}")
    print(f"  - 跳过 (无法提取answer): {skipped_no_answer}")
    print(f"  - 有效样本数: {len(data)}")
    
    # 统计各type的数量
    type_counts = {}
    for item in data:
        t = item['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\n各type数量分布:")
    for t, count in sorted(type_counts.items()):
        print(f"  - {t}: {count}")
    
    # 创建HuggingFace Dataset
    dataset = Dataset.from_list(data)
    print(f"\nDataset created with {len(dataset)} rows")
    
    # 上传到HuggingFace
    print("\nUploading to HuggingFace...")
    dataset.push_to_hub("guanning-ai/gsm8k-metamath")
    print("Upload complete!")
    
    return dataset


if __name__ == "__main__":
    create_dataset()

