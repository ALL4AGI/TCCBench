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

    # # [ 1]: ,  History,
    # You are a professional linguistic assessment expert. Your task is to determine whether a given text (sentence or paragraph) is suitable for a "Word Reordering Benchmark Dataset".
    # Background: We are building a dataset to evaluate LLMs by shuffling words and asking the model to reconstruct the text.
    # A suitable sentence or paragraph MUST meet ALL the following criteria:
    # 1. **Grammar & Fluency**: Grammatically correct English with no factual or technical errors. Math formulas or symbols are allowed.
    # 2. **Semantic Completeness**: The meaning must be clear and relatively complete.
    # 3. **Neutrality**: No bias, discrimination, or strong political coloring.
    # 4. **Cleanliness**: No unusual symbols (e.g., emojis). Standard punctuation (., ? !) is required.
    # 5. **Format**: NOT a dialogue (e.g., "Alice: Hello") or a list item (e.g., "1. First step").
    # 6. **Domain Relevance**: The text must be descriptive, factual, or explanatory related to the field of **Chemistry**.
    # Please reply with "YES" or "NO" first, followed by a brief reason.
    # Sentence to evaluate: {sentence}
    def build_judge_prompt(self, sentence: str) -> str:
        return f"""
        You are a professional language evaluation expert responsible for determining whether a given text (sentence or paragraph) is qualified.

        Background and Task Description:
        The goal is to construct a benchmark dataset to evaluate the comprehensive capabilities of LLMs through reconstructing texts from shuffled, discrete words. To this end, given an original sentence or paragraph that will later be tokenized and shuffled into discrete words, your task is to determine whether the text is suitable for inclusion in this dataset.

        A suitable sentence or paragraph must satisfy **all** of the following criteria:

        1. Grammatically correct and fluent, with no factual or technical errors. Formulas or symbolic expressions are allowed.
        2. Semantically clear and relatively complete, not a fragment or truncated content. Sentences that begin with connectors dependent on prior context (such as "其中", "而且", "然而", "另外", etc.) should be excluded.
        3. The topic must be free of bias, discrimination, or political content.
        4. The sentence must not contain uncommon symbols, such as emojis (symbols appearing in formulas are excluded from this restriction).
        5. Proper punctuation is used (e.g., periods, commas, question marks, exclamation marks).
        6. The text is not in a special format unsuitable for this task, such as dialogue or lists (e.g., "Zhang San: Hello!" or "1. Step one").
        7. The text belongs to the field of **Geography** and describes principles, facts, or phenomena with real scientific meaning. Empty, overly generic, or low-information-content sentences should be excluded, such as: "n can take any positive integer," or "Over the next century, chemists continued to search for a more accurate classification system."

        First, explicitly answer "YES" or "NO", and then briefly explain the reason.
        Return only the judgment result and a brief explanation, and do not include any other content.

        Sentence to evaluate: {sentence}
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
            # ,  YES
            return result.upper().startswith("YES")
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
            print(f"Processing Batch {batch_num + 1}/{total_batches}, items: {len(current_batch)}")

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
                print(f"✅ Added {len(suitable_in_batch)} sentences | Total: {suitable_count}")

            # attempts
            if batch_num + 1 < total_batches:
                time.sleep(delay)

        print(f"\n🎉 Processing Complete! Total suitable sentences: {suitable_count}")
        print(f"Saved to: {output_file}")

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
                    print(f"⚠️  Line {line_num} decode error, skipped: {e}")
        print(f"📥 Successfully loaded {len(sentences)} sentences (starting from line {start_line})")
        return sentences
    except Exception as e:
        print(f"❌ Failed to load file: {str(e)}")
        return []

def main():
    # API
    # , ornotKey
    # Key, Key
    api_key = "YOUR_API_KEY"

    if not api_key:
        print("❌ Please set API_KEY")
        return

    # Initialize filter
    filter = LLMBenchmarkFilter(api_key)

    start_line = 0    # 0
    # dataset01 history_en01.jsonl
    input_file = r"D:\01project\wordorder\dataset01\wikipedia\geography_en3_01_45words.jsonl"

    sentences = load_sentences_from_jsonl(input_file, start_line)
    if not sentences:
        print("❌ No data to process.")
        return

    output_file = r"D:\01project\wordorder\dataset02\wikipedia\geography_en3_02.jsonl"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    batch_size = 10
    save_interval = 10

    print(f"\n🚀 Start Processing: {len(sentences)} sentences | Batch size: {batch_size}")
    filter.process_and_save(
        sentences=sentences,
        output_file=output_file,
        batch_size=batch_size,
        save_interval=save_interval
    )

if __name__ == "__main__":
    main()