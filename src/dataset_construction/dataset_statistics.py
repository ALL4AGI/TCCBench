import json
from collections import defaultdict

# File Paths
input_file_path = r"D:\01project\wordorder\dataset02\wikipedia\physics_en02_dedup.jsonl"
output_file_path = r"D:\01project\wordorder\dataset02\wikipedia\physics_en02_dedup_statis.txt"

# Initialize statistics variables
word_count_stats = defaultdict(int)
word_only_once_stats = {"yes": 0, "no": 0}
# word_countattemptsattempts
word_count_detail = defaultdict(lambda: {"count": 0, "first_line": None})

# Define word-count interval function
def get_word_count_level(word_count):

    if 5 <= word_count < 15:
        return "level1 [5-15)"
    elif 15 <= word_count < 25:
        return "level2 [15-25)"
    elif 25 <= word_count < 35:
        return "level3 [25-35)"
    elif 35 <= word_count < 45:
        return "level4 [35-45)"
    elif 45 <= word_count < 55:
        return "level5 [45-55)"
    elif 55 <= word_count < 65:
        return "level6 [55-65)"
    elif 65 <= word_count < 75:
        return "level7 [65-75)"
    elif 75 <= word_count < 85:
        return "level8 [75-85)"
    elif word_count >= 85:
        return "level9 [85-∞)"
    else:    # 5
        return "level0 [0-5)"

# Read and process JSONL file
try:
    with open(input_file_path, "r", encoding="utf-8") as f:
        line_num = 0
        for line in f:
            line_num += 1
            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            try:
                # Parse JSON line
                data = json.loads(line)

                # Handle various word_count types
                word_count_val = data.get("word_count")
                word_count = 0
                # Handle null values
                if word_count_val is None:
                    print(f"第{line_num}行的word_count为空, 跳过")
                    continue
                # If string, try to convert to number
                elif isinstance(word_count_val, str):
                    # /,
                    clean_val = word_count_val.strip()
                    if clean_val.isdigit():
                        word_count = int(clean_val)
                    else:
                        print(f"第{line_num}行的word_count是非数字字符串({word_count_val}), 跳过")
                        continue
                # If number, convert to int directly
                elif isinstance(word_count_val, (int, float)):
                    word_count = int(word_count_val)
                else:
                    print(f"第{line_num}行的word_count类型无效({type(word_count_val)}), 跳过")
                    continue

                # Basic word-count interval statistics
                level = get_word_count_level(word_count)
                word_count_stats[level] += 1

                # word_count
                if word_count_detail[word_count]["first_line"] is None:
                    word_count_detail[word_count]["first_line"] = line_num
                word_count_detail[word_count]["count"] += 1

                # Count word_only_once
                word_only_once = data.get("word_only_once", "").lower()
                if word_only_once in ["yes", "no"]:
                    word_only_once_stats[word_only_once] += 1
                else:
                    print(f"第{line_num}行的word_only_once值无效({word_only_once}), 跳过")

            except json.JSONDecodeError:
                print(f"第{line_num}行JSON格式错误, 跳过")
            except Exception as e:
                print(f"第{line_num}行处理出错: {str(e)}, 跳过")

    # word_count10
    # Sort by word_count descending10
    sorted_word_count = sorted(word_count_detail.items(), key=lambda x: x[0], reverse=True)
    top10_word_count = sorted_word_count[:10]

    # Generate statistics output text
    result_lines = []
    result_lines.append("=== 字数区间统计(细分版)===")
    level_order = [
        "level0 [0-5)",
        "level1 [5-15)",
        "level2 [15-25)",
        "level3 [25-35)",
        "level4 [35-45)",
        "level5 [45-55)",
        "level6 [55-65)",
        "level7 [65-75)",
        "level8 [75-85)",
        "level9 [85-∞)"
    ]
    for level in level_order:
        result_lines.append(f"{level}: {word_count_stats[level]} 条")

    # top10 word_count
    result_lines.append("\n=== 字数最大的10个数值统计 ===")
    result_lines.append("排名 | 字数 | 首次出现行号(数据ID) | 出现数量")
    result_lines.append("-" * 50)
    for idx, (wc, detail) in enumerate(top10_word_count, 1):
        result_lines.append(f"{idx:2d}   | {wc:3d} | {detail['first_line']:10d}        | {detail['count']:3d}")

    result_lines.append("\n=== word_only_once 统计 ===")
    result_lines.append(f"yes: {word_only_once_stats['yes']} 条")
    result_lines.append(f"no: {word_only_once_stats['no']} 条")

    # Compute totals
    total_lines = sum(word_count_stats.values())
    result_lines.append(f"\n=== 总计 ===")
    result_lines.append(f"有效数据行数: {total_lines} 条")
    # word_only_oncetotal
    total_word_only = word_only_once_stats['yes'] + word_only_once_stats['no']
    result_lines.append(f"word_only_once有效行数: {total_word_only} 条")
    result_lines.append(f"唯一字数种类数: {len(word_count_detail)} 种")

    # Write results to output file
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    print(f"统计完成!结果已保存至: {output_file_path}")
    print("\n统计结果预览: ")
    print("\n".join(result_lines))

except FileNotFoundError:
    print(f"错误: 输入文件 {input_file_path} 不存在, 请检查路径是否正确")
except PermissionError:
    print(f"错误: 没有权限读取/写入文件, 请检查文件权限")
except Exception as e:
    print(f"程序运行出错: {str(e)}")