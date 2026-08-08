"""
处理中英文语句，把句子离散成为单词，dataset02到dataset03
"""
import jsonlines
import os
import random
import re
import requests
import secrets
from typing import List, Dict

# ==================== Configuration Parameters ====================
CONFIG = {
    "input_file": r"D:\01project\wordorder\buchong2.jsonl",
    "output_file": r"D:\01project\wordorder\buchong_discrete2.jsonl",
    "deepseek_api_url": "https://api.deepseek.com/v1/chat/completions",
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY"),
    "target_fields": ["discipline", "language", "original_sentence", "word_count", "word_only_once"]
}

# =========================================

def is_capital_needed_api(word: str) -> bool:
    """
    [仅针对英文句子首词] 调用DeepSeek API判断单词本身是否需要保持大写
    """
    if word.lower() == "i": return True
    if not word[0].isalpha(): return True

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system",
             "content": "You are a linguistic expert. Determine if the provided word is a Proper Noun (e.g., Person Name 'Mike', Location 'London', Organization 'NASA', Specific Brand 'Nike', or Proprietary Term) that MUST remain capitalized even in the middle of a sentence. Return 'yes' for Proper Nouns. Return 'no' for common nouns (e.g., 'Reaction', 'Cell', 'He', 'That'), verbs, or adjectives."},
            {"role": "user", "content": f"Word: '{word}'"}
        ],
        "temperature": 0
    }
    headers = {"Authorization": f"Bearer {CONFIG['deepseek_api_key']}"}
    try:
        response = requests.post(CONFIG["deepseek_api_url"], json=payload, headers=headers, timeout=5)
        content = response.json()["choices"][0]["message"]["content"].strip().lower()
        return "yes" in content
    except:
        return True    # API

class AdvancedTokenizer:
    def __init__(self):
        self.placeholders = {}
        self.ph_counter = 0

    def _create_ph(self, text):
        """创建不包含任何特殊标点的纯字母ID，防止被分词器切碎"""
        ph = f"HTOKEN{self.ph_counter}END"
        self.placeholders[ph] = text
        self.ph_counter += 1
        return ph

    def protect_latex(self, text):
        """保护 Latex 结构（仅英文模式生效）"""
        pattern = re.compile(r'(?P<latex>\{\\displaystyle.*?\}|\$.*?\$)')
        matches = list(pattern.finditer(text))
        if not matches:
            return text

        new_text = ""
        last_end = 0

        for match in matches:
            latex_content = match.group('latex')
            start_pos = match.start()
            end_pos = match.end()

            prefix_end = start_pos
            prefix_start = start_pos

            while prefix_start > last_end:
                char = text[prefix_start - 1]
                if re.match(r'[\u4e00-\u9fff., , ; : ?!""()[]~–]', char):
                    break
                prefix_start -= 1

            normal_part = text[last_end: prefix_start]
            formula_unit = text[prefix_start: end_pos]
            ph = self._create_ph(formula_unit)

            new_text += normal_part + " " + ph + " "
            last_end = end_pos

        new_text += text[last_end:]
        return new_text

    def split_text(self, sentence: str, lang: str) -> List[str]:
        self.placeholders = {}
        self.ph_counter = 0
        text = sentence
        tokens = []

        # Determine split mode
        is_chinese_mode = (lang == 'zh' or re.search(r'[\u4e00-\u9fff]', sentence))

        if is_chinese_mode:
            # === (, )===
            # Step 1Remove Chinese punctuation except enumeration comma
            chinese_punctuation_except_dunhao = r'., ; : ?!""()[]《》''·'
            text_clean = re.sub(f'[{chinese_punctuation_except_dunhao}]', '', text)

            # Step 2: Define regex pattern - Match Chinese chars, enumeration commas, or non-Chinese blocks
            pattern = re.compile(r'([\u4e00-\u9fff])|(, )|([^\u4e00-\u9fff, ]+)')

            # Step 3: Iterate matches
            matches = pattern.findall(text_clean)
            for zh_char, dunhao, en_num_block in matches:
                if zh_char:
                    # Single Chinese character, add directly
                    tokens.append(zh_char)
                elif dunhao:
                    # Enumeration comma, skip as delimiter
                    continue
                elif en_num_block:
                    # Non-Chinese block, clean and add
                    clean_block = en_num_block.strip()
                    if clean_block:
                        tokens.append(clean_block)
        else:
            # ======
            # Step 0 - $
            formula_pattern = re.compile(r'(\$.*?\$)')
            # ,
            formula_placeholders = {}
            formula_ph_counter = 0

            def replace_formula(match):
                nonlocal formula_ph_counter
                ph = f"FORMULA{formula_ph_counter}END"
                formula_placeholders[ph] = match.group(1)
                formula_ph_counter += 1
                return f" {ph} "

            text = formula_pattern.sub(replace_formula, text)

            # 1. Protect LaTeX structures
            text = self.protect_latex(text)

            # 2. Protect specific complex semantic blocks
            complex_patterns = [
                r'\d+\s+ATP\s+/\s+\d+\s+H\+',
                r'OS\s*=\s*[\+\-]?\d+(?:\s+and\s+[\+\-]?\d+)?',
                r'\d+\s*[~–-]\s*\d+',
                r'[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+',
            ]

            for pat in complex_patterns:
                text = re.sub(pat, lambda m: " " + self._create_ph(m.group(0)) + " ", text)

            # Core tokenization
            raw_parts = text.split()
            for i, part in enumerate(raw_parts):
                if "FORMULA" in part:
                    # ID
                    match = re.search(r'(FORMULA\d+END)', part)
                    if match:
                        ph_key = match.group(1)
                        original_formula = formula_placeholders.get(ph_key, part)
                        tokens.append(original_formula)
                        continue

                # HTOKEN
                if "HTOKEN" in part:
                    match = re.search(r'(HTOKEN\d+END)', part)
                    if match:
                        ph_key = match.group(1)
                        original = self.placeholders[ph_key]
                        tokens.append(original)
                        continue

                clean_part = part.strip('.,;:"?!')
                if not clean_part:
                    continue

                if len(tokens) == 0:
                    if clean_part[0].isupper() and clean_part.isalpha():
                        if not is_capital_needed_api(clean_part):
                            clean_part = clean_part[0].lower() + clean_part[1:]

                tokens.append(clean_part)

        # Final cleanup: remove empty strings
        tokens = [t for t in tokens if t.strip()]
        return tokens

def process_line(item: Dict) -> Dict:
    lang = item.get("language", "en")
    sentence = item.get("original_sentence", "")

    tokenizer = AdvancedTokenizer()
    word_list = tokenizer.split_text(sentence, lang)

    # attempts
    word_list = [w for w in word_list if w.strip()]

    # Random shuffle
    seed = secrets.randbits(128)
    random.seed(seed)
    random.shuffle(word_list)

    new_item = {}
    new_item["discipline"] = item.get("discipline", "")
    new_item["language"] = lang
    new_item["original_sentence"] = sentence
    new_item["word_count"] = str(len(word_list))
    new_item["word_only_once"] = item.get("word_only_once", "")
    new_item["word_list"] = word_list

    return new_item

def main():
    if not os.path.exists(os.path.dirname(CONFIG["output_file"])):
        os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)

    print("任务开始：正在执行高精度分词处理...")
    count = 0

    try:
        with jsonlines.open(CONFIG["input_file"], 'r') as reader:
            with jsonlines.open(CONFIG["output_file"], 'w') as writer:
                for obj in reader:
                    processed = process_line(obj)
                    writer.write(processed)
                    count += 1
                    if count % 50 == 0:
                        print(f"已处理 {count} 条...")
    except Exception as e:
        print(f"发生严重错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"处理完成！总计: {count} 条。")
    print(f"输出文件: {CONFIG['output_file']}")

if __name__ == "__main__":
    main()

# # ==================== Configuration Parameters ====================
# }
# # =========================================
# }
# break
# prefix_start -= 1
# new_text += normal_part + " " + ph + " "
# new_text += text[last_end:]
# # Determine split mode
# # === (, not)===
# # Step 1Remove Chinese punctuation except enumeration comma
# # Step 2: Define regex pattern - Match Chinese chars, enumeration commas, or non-Chinese blocks
# # Step 3: Iterate matches
# # Single Chinese character, add directly
# tokens.append(zh_char)
# elif dunhao:
# # Enumeration comma, skip as delimiter
# continue
# elif en_num_block:
# # Non-Chinese block, clean and add
# tokens.append(clean_block)
# else:
# # ======
# # 1. Protect LaTeX structures (Priority)
# # 2. Protect specific complex semantic blocks (Regex List)
# # (A) /: 1 ATP / 3 H+
# r'\d+\s+ATP\s+/\s+\d+\s+H\+',
# # (B) /: OS = +1 and +2
# r'OS\s*=\s*[\+\-]?\d+(?:\s+and\s+[\+\-]?\d+)?',
# # (C) : 3~12, 30–300, 2-9
# r'\d+\s*[~–-]\s*\d+',
# # (D) : Wallace Hume Carothers
# r'[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+',
# # Core tokenization
# tokens.append(original)
# continue
# continue
# tokens.append(clean_part)
# # Final cleanup: remove empty strings
# # attempts
# # Random shuffle
# random.seed
# random.shuffle(word_list)
# writer.write(processed)
# count += 1
# traceback.print_exc
