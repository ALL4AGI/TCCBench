"""
代码功能: 批量处理JSONL文件中的句子zh-Hans_0-5w_cn.jsonl, cc_stories_sentence_en.jsonl
通过DeepSeek API进行分类标注
添加一些字段, "discipline", "l1_chapter", "l2_section", "l3_topic", "l4_keyword", "language", 得到如下样式的数据:
{"discipline": "物理", "l1_chapter": "電磁學", "l2_section": "電路理論", "l3_topic": "電容與磁場效應", "l4_keyword": "磁場",  "language": "zh", "original_sentence": "电流会产生磁场.", "word_count": "7", "word_only_once": "yes"}

主要流程:
1. 加载待处理的句子数据(每行一个JSON对象)
2. 调用DeepSeek API, 将句子分类到预设的4级生活分类大纲中
3. 将分类结果保存为新JSONL文件, 不符合的句子存入丢弃文件
4. 支持断点续传(通过checkpoint.txt记录进度)
"""

import json
import re
import os
import time
from openai import OpenAI

# ================= API Configuration =================
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com")

# ================= File Paths =================
INPUT_PATH = r"D:\01project\wordorder\dataset02\cc_stories\cc-stories3_en_45words.jsonl"
# ##Processed file output
OUTPUT_PATH = r"D:\01project\wordorder\dataset02\cc_stories\cc-stories3_en02_45words.jsonl"
# #Discarded output
DISCARD_PATH = r"D:\01project\wordorder\dataset02\cc_stories\cc-stories3_en_45words_dis.jsonl"
CHECKPOINT_PATH = r"checkpoint.txt"    # checkpoint position

# ================= Complete Outline =================
OUTLINE_TEXT = """1. (Personal and Physiological Needs) 1.1. (Time Management and Habits) 1.1.1. (Daily routines) 1.1.2. (Goal setting and to-do lists) 1.1.3. (Procrastination and overcoming strategies) 1.2. (Diet and Health) 1.2.1. (Dietary structure and nutrition) 1.2.2. (Exercise and physical activity) 1.2.3. (Sleep hygiene and quality management) 1.3. (Self-Care and Emotion) 1.3.1. (Emotion recognition and stress management) 1.3.2. (Personal hygiene and grooming) 1.3.3. (Self-reflection and introspection) 1.4. (Life Perception and Sense of Time) 1.4.1. (Realization of dreams and life milestones) 1.4.2. (Reflections on time, past memories, and significant years) 1.4.3. (Instant physiological reactions like yawning or sneezing) 2. (Living and Environment) 2.1. (Home Life Management) 2.1.1. (Home cleaning and organization) 2.1.2. (Energy and resource conservation) 2.1.3. (Home appliances and technology use) 2.2. (Shopping and Finance) 2.2.1. (Daily purchasing and consumption decisions) 2.2.2. (Personal budgeting and expense tracking) 2.2.3. (Online shopping and digital payment habits) 2.3. (Material Evaluation and Life Aesthetics) 2.3.1. (Evaluation of cost-effectiveness, utility, and efficacy) 2.3.2. (Personal aesthetic preferences and style choices) 3. (Work and Social Connection) 3.1. (Work and Study Activities) 3.1.1. (Workday workflow and productivity) 3.1.2. (Remote vs. office work differences) 3.1.3. (Skill learning and professional development) 3.1.4. (Academic workload, attending lectures, and learning process) 3.2. (Interpersonal Relationships and Communication) 3.2.1. (Family relationship maintenance and interaction) 3.2.2. (Social occasions and etiquette) 3.2.3. (Digital communication and social media use) 3.2.4. (Social observation: crowds, commuting, and public order) 3.2.5. (Administrative notifications and daily contact) 4. (Leisure and Cultural Activities) 4.1. (Entertainment and Relaxation) 4.1.1. (Media consumption habits) 4.1.2. (Outdoor activities and travel planning) 4.1.3. (Games and hobbies) 4.2. (Public Participation and Culture) 4.2.1. (Reading and knowledge acquisition) 4.2.2. (Community activities and volunteering) 4.2.3. (Art appreciation and cultural experiences) 5. (Life Philosophy and Behavioral Motivations) 5.1. (Attitude towards life and motivational analysis) 5.1.1. (Involuntary choices and survival motivation) 5.1.2. (Value judgments on simplicity vs. complexity)"""

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fa5]', text))

def call_model_with_retry(batch, retries=3):

    sentences = [item['original_sentence'] for item in batch]
    prompt = (
        f"You are a classification expert. Classify the following sentences into the categories provided in the outline below:\n{OUTLINE_TEXT}\n\n"
        "Rules:\n"
        "1. Return a JSON object with a 'results' key containing a list of objects: [{\"l1\":..., \"l2\":..., \"l3\":..., \"l4_keyword\":...}, ...]\n"
        "2. If a sentence does not belong to 'Daily Life' (e.g., academic, political, nonsense), set its object to null.\n"
        "3. Language Consistency: If the sentence is Chinese, use Chinese for l1/l2/l3. If English, use the English titles provided in parentheses.\n"
        "4. l4_keyword: Generate 1 core keyword based on the sentence.\n"
        "5. Maintain the order of the input sentences."
        "Note: 'Daily Life' is a broad category. Include personal reflections, common social observations, commuting scenes, and basic life evaluations."
    )

    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(sentences, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"},
                timeout=30    # timeout
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("results", [])
        except Exception as e:
            print(f"网络异常或模型错误 (尝试 {i + 1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(5)    # 5sec
            else:
                print("已达到最大重试次数, 跳过当前批次.")
                return [None] * len(batch)

def save_checkpoint(line_idx):
    with open(CHECKPOINT_PATH, 'w') as f:
        f.write(str(line_idx))

def get_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return int(f.read().strip())
    return 0

def main():
    start_line = get_checkpoint()
    print(f"从第 {start_line} 行开始重启任务...")

    # Append mode for resume support
    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_PATH, 'a', encoding='utf-8') as f_out, \
            open(DISCARD_PATH, 'a', encoding='utf-8') as f_dis:

        batch = []
        batch_size = 10

        for idx, line in enumerate(f_in):
            # Skip processed lines
            if idx < start_line:
                continue

            try:
                batch.append(json.loads(line.strip()))
            except:
                continue

            if len(batch) == batch_size:
                results = call_model_with_retry(batch)

                for item, res in zip(batch, results):
                    if res is None:
                        f_dis.write(json.dumps(item, ensure_ascii=False) + "\n")
                    else:
                        lang = "zh" if is_chinese(item['original_sentence']) else "en"
                        processed = {
                            "discipline": "日常生活" if lang == "zh" else "daily life",
                            "l1_chapter": res.get("l1"),
                            "l2_section": res.get("l2"),
                            "l3_topic": res.get("l3"),
                            "l4_keyword": res.get("l4_keyword"),
                            "language": lang,
                            "original_sentence": item['original_sentence'],
                            "word_count": item['word_count'],
                            "word_only_once": item['word_only_once']
                        }
                        f_out.write(json.dumps(processed, ensure_ascii=False) + "\n")

                # Update progress
                current_processed_count = idx + 1
                save_checkpoint(current_processed_count)
                f_out.flush()
                f_dis.flush()
                print(f"进度: 已处理至第 {current_processed_count} 行")
                batch = []

        # Process remaining batch
        if batch:
            # Same logic as above
            results = call_model_with_retry(batch)
            for item, res in zip(batch, results):
                if res:
                    # ...
                    pass
            save_checkpoint(idx + 1)

    print("任务执行完毕.")

if __name__ == "__main__":
    # Pre-run checklist:
    # 1. openai library installed
    # 2. API key configured
    main()
    pass