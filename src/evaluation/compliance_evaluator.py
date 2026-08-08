import re
from collections import Counter
from typing import List

class ComplianceEvaluator:
    def __init__(self, logger):
        self.logger = logger

    def split_into_words(self, sentence: str) -> List[str]:

        # , , /,
        pattern = r"""(?:[A-Z][.']?[A-Z]+(?:[.][A-Z]+)*) # |(?:[A-Z][a-z'-]*(?:\s+[A-Z][a-z'-]*)*) # |[\w'-]+(?:-[^\s]+)? # / |\d+ #"""
        # VERBOSEand,
        words = re.findall(pattern, sentence, re.VERBOSE)
        # ,
        return [word.strip() for word in words if word.strip()]

    def evaluate(self, original_word_list: List[str], generated_sentence: str):

        # 1. Split generated sentence into words
        generated_words = self.split_into_words(generated_sentence)
        self.logger.info(f"生成句子拆分后单词列表: {generated_words}")

        # 2. Compute word frequency for original and generated
        original_counter = Counter(original_word_list)
        generated_counter = Counter(generated_words)
        self.logger.info(f"原单词列表词频: {dict(original_counter)}")
        self.logger.info(f"生成单词列表词频: {dict(generated_counter)}")

        # 3. Compute extra words
        extra_words = {}
        for word, gen_count in generated_counter.items():
            orig_count = original_counter.get(word, 0)
            if gen_count > orig_count:
                extra_words[word] = gen_count - orig_count

        # 4. Compute missing words
        missing_words = {}
        for word, orig_count in original_counter.items():
            gen_count = generated_counter.get(word, 0)
            if gen_count < orig_count:
                missing_words[word] = orig_count - gen_count

        # 5. Compute total error count
        total_original = len(original_word_list)
        extra_count = sum(extra_words.values())
        missing_count = sum(missing_words.values())
        error_count = extra_count + missing_count

        # 6. Compute consistency percentage
        if total_original == 0:
            consistency = 0.0
        else:
            consistency = ((total_original - error_count) / total_original) * 100

        # 7. Determine consistency level
        consistency_level = "below_80%"
        if consistency >= 100:
            consistency_level = "100%"
        elif consistency >= 90:
            consistency_level = "90%"
        elif consistency >= 80:
            consistency_level = "80%"

        # Build evaluation result
        result = {
            "no_extra_words": len(extra_words) == 0,
            "no_missing_words": len(missing_words) == 0,
            "no_repeated_words": error_count == 0,    # /
            "extra_words": extra_words,
            "missing_words": missing_words,
            "error_count": error_count,
            "consistency_percentage": round(consistency, 2),
            "consistency_level": consistency_level
        }

        self.logger.info(f"合规性评估结果: 一致性 {result['consistency_percentage']}%, 等级 {result['consistency_level']}")
        return result
