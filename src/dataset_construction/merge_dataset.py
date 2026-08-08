import os

input_files = [
    r"D:\01project\wordorder\dataset03\cc_stories1_en03.jsonl",
    r"D:\01project\wordorder\dataset03\zh-Hans_0-5w_zh03.jsonl",
    r"D:\01project\wordorder\dataset03\biology_en03.jsonl",
    r"D:\01project\wordorder\dataset03\biology_zh03.jsonl",
    r"D:\01project\wordorder\dataset03\chemistry_en03.jsonl",
    r"D:\01project\wordorder\dataset03\chemistry_zh03.jsonl",
    r"D:\01project\wordorder\dataset03\geography_en03.jsonl",
    r"D:\01project\wordorder\dataset03\geography_zh03.jsonl",
    r"D:\01project\wordorder\dataset03\history_en03.jsonl",
    r"D:\01project\wordorder\dataset03\history_zh03.jsonl",
    r"D:\01project\wordorder\dataset03\physics_en03.jsonl",
    r"D:\01project\wordorder\dataset03\physics_zh03.jsonl",
]

output_file = r"D:\01project\wordorder\dataset03\TCCBench.jsonl"

def merge_jsonl(files, output_path):
    total_lines = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for file_path in files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            print(f"正在合并: {file_path}")
            with open(file_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    fout.write(line + "\n")
                    total_lines += 1

    print(f"\n合并完成 ✅")
    print(f"输出文件: {output_path}")
    print(f"总行数: {total_lines}")

if __name__ == "__main__":
    merge_jsonl(input_files, output_file)
