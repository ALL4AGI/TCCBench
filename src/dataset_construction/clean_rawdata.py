"""D:\01project\wordorder\dataraw\wikipedia\physics_zh.jsonl D:\01project\wordorder\dataraw\wikipedia\physics_zh_clean.jsonl"""

import json
import re
import os

def clean_physics_data_smart_parens(input_file, output_file):
    """Wikipedia JSONL - == == - {\displaystyle ...}"""

    # --- Path Configuration ---
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 已创建输出目录: {output_dir}")

    # --- Regex Patterns ---

    # 1. Title cleaning
    pattern_header = re.compile(r'={1,5}\s*[^=]{0,10}?\s*={1,5}')

    # 2. Protect formulas {\displaystyle ...} (re.DOTALL )
    pattern_latex = re.compile(r'(\{\\displaystyle\s*.*?\s*\})', re.DOTALL)

    # 3.
    pattern_parentheses_zh = re.compile(r'([^)]*)')
    pattern_parentheses_en = re.compile(r'\([^)]*\)')

    processed_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
                open(output_file, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️ 警告: 无法解析 JSON 行: {line[:50]}...")
                    continue

                raw_content = data.get("raw_content", "")

                if raw_content:

                    # ----------------------------------------------------
                    # --- 0: ---
                    # ----------------------------------------------------

                    placeholders = []

                    def protect_latex(match):

                        original_formula = match.group(1)
                        # ,
                        placeholder = f"__FORMULA_{len(placeholders)}__"
                        placeholders.append(original_formula)
                        return placeholder

                    cleaned_content = pattern_latex.sub(protect_latex, raw_content)

                    # ----------------------------------------------------
                    # 1 & 2: Execute cleaning ---
                    # ----------------------------------------------------

                    # 1: Remove headings
                    cleaned_content = pattern_header.sub('', cleaned_content)

                    # 2: Chinese full-width parentheses
                    cleaned_content = pattern_parentheses_zh.sub('', cleaned_content)

                    # 3: English half-width parentheses
                    cleaned_content = pattern_parentheses_en.sub('', cleaned_content)

                    # ----------------------------------------------------
                    # 4: Restore formulas ---
                    # ----------------------------------------------------

                    for i, original_formula in enumerate(placeholders):
                        placeholder = f"__FORMULA_{i}__"
                        cleaned_content = cleaned_content.replace(placeholder, original_formula)

                    # Additional cleanup
                    cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()

                    # Update dictionary
                    data["raw_content"] = cleaned_content

                # Write to new file
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1

        print(f"🎉 处理完成!\n输入文件: {input_file}\n输出文件: {output_file}\n共处理行数: {processed_count}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到输入文件: {input_file}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == "__main__":
    # Configure paths
    input_path = r"D:\01project\wordorder\dataraw\wikipedia\geography_en3.jsonl"
    output_path = r"D:\01project\wordorder\dataraw\wikipedia\geography_en3_clean.jsonl"

    clean_physics_data_smart_parens(input_path, output_path)