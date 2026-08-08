"""1. Set original_sentence 2. - jieba - Jaccard - 0.5 70% JSONL 'original_sentence' - _dedup.jsonl: - _discard.txt:"""

import json
import jieba
import re
import os

# --- Configuration Parameters ---

INPUT_FILE = r'D:\01project\wordorder\dataset\dataset03\TCCBench.jsonl'
OUTPUT_FILE = r'D:\01project\wordorder\dataset\dataset03\TCCBench-out.jsonl'
DISCARD_FILE = r'D:\01project\wordorder\dataset\dataset03\TCCBench-out_discard.txt'

SIMILARITY_THRESHOLD = 0.5

# Built-in minimal English stopword list ( "the", "is" )
ENGLISH_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "that", "this", "it", "he", "she", "they"
}

def get_tokens(text, lang='zh'):

    if not text:
        return set()

    if lang == 'zh':
        # --- Chinese processing logic ---
        # Tokenize with jieba, filter non-alphanumeric
        return set([w for w in jieba.lcut(text) if w.isalnum()])

    else:
        # --- English processing logic ---
        # 1. Convert to lowercase
        text = text.lower()
        # 2. Extract words using regex (\b\w+\b)
        words = re.findall(r'\b\w+\b', text)
        # 3. Filter stopwords and pure numbers
        clean_words = set([w for w in words if w not in ENGLISH_STOP_WORDS and not w.isdigit()])
        return clean_words

def get_jaccard_similarity(set1, set2):
    """Jaccard"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def run_deduplication():
    kept_data = []
    kept_token_sets = []    # token (set)
    exact_hashes = set()

    discard_count = 0
    total_count = 0

    print(f"正在读取文件: {INPUT_FILE}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
            open(DISCARD_FILE, 'w', encoding='utf-8') as f_dis:

        for line in f_in:
            if not line.strip():
                continue

            total_count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告: 第 {total_count} 行 JSON 格式错误, 跳过.")
                continue

            sentence = item.get("original_sentence", "").strip()
            lang = item.get("language", "zh")    # default: Chinese

            if not sentence:
                continue

            # 1. (Exact Match)
            if sentence in exact_hashes:
                f_dis.write(f"[精确重复]| {sentence} | 无\n")
                discard_count += 1
                continue

            # 2. (Tokens)
            current_tokens = get_tokens(sentence, lang)

            # (1), or
            # ,  token notand, (or)
            if not current_tokens:
                exact_hashes.add(sentence)
                kept_token_sets.append(current_tokens)
                kept_data.append(item)
                continue

            # 3. (Fuzzy Match)
            is_similar = False
            #
            for idx, existing_tokens in enumerate(kept_token_sets):
                if not existing_tokens:
                    continue

                similarity = get_jaccard_similarity(current_tokens, existing_tokens)

                if similarity > SIMILARITY_THRESHOLD:
                    hit_sentence = kept_data[idx]["original_sentence"]
                    f_dis.write(f"[结构相似]| {sentence} | 匹配原句: {hit_sentence} | 相似度: {similarity:.2f}\n")
                    is_similar = True
                    break

            if is_similar:
                discard_count += 1
                continue

            # 4.
            exact_hashes.add(sentence)
            kept_token_sets.append(current_tokens)
            kept_data.append(item)

    # 5.
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in kept_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("-" * 30)
    print(f"处理完成!")
    print(f"总行数: {total_count}")
    print(f"保留: {len(kept_data)}")
    print(f"丢弃: {discard_count} (重复率: {discard_count / total_count:.2%})")
    print(f"结果文件: {OUTPUT_FILE}")
    print(f"日志文件: {DISCARD_FILE}")

if __name__ == "__main__":
    run_deduplication()
