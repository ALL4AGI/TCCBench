"""1. dataset01 2. DeepSeek-API 3. dataset02"""
# (1)29, ; (2)121, 122, 129ID

import openai
import json
import os
from typing import List, Dict, Any
import time

class LLMBenchmarkFilter:
    """GLM https://open.bigmodel.cn/api/paas/v4/"""
    """deepseek https://api.deepseek.com/v1"""
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def build_judge_prompt(self, sentence: str) -> str:
        return f"""
        你是一位专业的语言评估专家, 负责判断给定文本(句子或段落)是否合格.

        背景及任务说明: 要构建一个基准数据集, 通过离散单词重构文本来测评LLM的综合能力.现在要使用给定的句子或段落打乱成离散单词.给定一个句子或段落, 你的任务是判断该文本是否适合纳入该数据集.

        一个合适的句子或段落必须满足以下所有标准:
        1. 语法正确且表达流畅, 没有事实或技术性错误.允许包含公式或符号表达.
        2. 语义清晰且相对完整, 不是片段或截断的内容; 剔除开头是"其中", "而且", "然而", "另外"等与上下文衔接的句子.
        3. 主题不带有偏见, 歧视, 政治色彩.
        4. 句子中不包含不常见的符号, 如表情符号.(公式中的符号除外).
        4. 使用正确的标点符号(., ?, !等).
        5. 不是对话, 列表等不适合上述任务的特殊格式文本(例如: "张三: 你好!"或"1. 第一步").
        6. 文本是生物学相关领域相关的原理性, 事实性和现象相关的描述, 有实际意义的句子或段落.剔除空洞, 泛泛而谈或无实际信息量的句子, 例如: "n 可以取任意正整数.", "在接下来的一个世纪，化学家们一直在寻找一个更准确的分类体系。"

        先明确回答"YES"或"NO", 然后简要说明理由.
        只返回判断结果和简要说明理由, 不要包含其他内容.

        待评估句子: {sentence}

        """.strip()

    def is_suitable(self, sentence_data: Dict[str, str], model: str = "deepseek-chat") -> bool:
        """deepseek-chat glm-4.6"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": self.build_judge_prompt(sentence_data["original_sentence"])}],
                temperature=0.2,
            )
            result = response.choices[0].message.content.strip()
            return "YES" in result.split("\n")[0]
        except:
            return False

    def process_and_save(self, sentences: List[Dict[str, str]], output_file: str,
                         batch_size: int = 5, save_interval: int = 10, delay: float = 1.0):

        total_batches = (len(sentences) + batch_size - 1) // batch_size
        suitable_count = 0

        for batch_num in range(total_batches):
            # attempts
            batch_start = batch_num * batch_size
            batch_end = batch_start + batch_size
            current_batch = sentences[batch_start:batch_end]
            print(f"处理第 {batch_num + 1}/{total_batches} 批，当前批句子数：{len(current_batch)}")

            # attempts
            suitable_in_batch = []
            for sentence_data in current_batch:
                if self.is_suitable(sentence_data):
                    suitable_in_batch.append(sentence_data)

            # attempts
            if suitable_in_batch:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for item in suitable_in_batch:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                suitable_count += len(suitable_in_batch)
                print(f"✅ 本批新增 {len(suitable_in_batch)} 条适合的句子 | 累计：{suitable_count} 条")

            # attempts
            if batch_num + 1 < total_batches:
                time.sleep(delay)

        print(f"\n🎉 处理完成！共找到 {suitable_count} 条适合的句子")
        print(f"结果已保存到：{output_file}")

def load_sentences_from_jsonl(file_path: str, start_line: int = 1) -> List[Dict[str, str]]:
    """JSONL"""
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num < start_line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    sentences.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  第 {line_num} 行解析错误，已跳过：{e}")
        print(f"📥 成功加载 {len(sentences)} 条句子数据（从第 {start_line} 行开始）")
        return sentences
    except Exception as e:
        print(f"❌ 加载文件失败：{str(e)}")
        return []

def main():
    # API
    # #GLM
    # #deepseek
    api_key = "YOUR_API_KEY"
    if not api_key:
        print("❌ 请先设置 DEEPSEEK_API_KEY 环境变量")
        return

    # Initialize filter
    filter = LLMBenchmarkFilter(api_key)

    # and
    start_line = 0   # 2501
    input_file = r"D:\01project\wordorder\dataset01\wikipedia\biology_zh01.jsonl"
    sentences = load_sentences_from_jsonl(input_file, start_line)
    if not sentences:
        print("❌ 无数据可处理，程序退出")
        return

    output_file = r"D:\01project\wordorder\dataset02\wikipedia\biology_zh02.jsonl"
    batch_size = 5
    save_interval = 10    # 10

    print(f"\n🚀 开始处理：共 {len(sentences)} 条句子 | 批大小：{batch_size}")
    filter.process_and_save(
        sentences=sentences,
        output_file=output_file,
        batch_size=batch_size,
        save_interval=save_interval
    )

if __name__ == "__main__":
    main()
