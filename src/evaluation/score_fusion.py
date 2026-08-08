from typing import Dict, List, Optional

class ScoreFusion:
    def __init__(self, config: Dict, logger):
        self.config = config
        self.quality_eval_type = config["evaluation_strategy"]["quality_eval_type"]
        self.both_include = config["evaluation_strategy"].get("both_include_in_overall", True)
        self.logger = logger

    def _get_single_quality_score(self, quality_eval: Dict, eval_type: str) -> float:

        if eval_type == "agent":
            # Agent quality score""
            return quality_eval.get("agent_quality", {}).get("质量总分", 0.0)
        elif eval_type == "metric":
            # Metric quality score""
            return quality_eval.get("metric_quality", {}).get("指标质量分数", 0.0)
        else:
            self.logger.warning(f"未知的质量评价类型: {eval_type}")
            return 0.0

    def _get_level(self, score: float) -> str:

        if score >= 90:
            return "优秀"
        elif 80 <= score < 90:
            return "良好"
        elif 70 <= score < 80:
            return "中等"
        elif 60 <= score < 70:
            return "较差"
        else:
            return "极差"

    def fuse(self, compliance_result: Dict, quality_eval: Dict) -> Dict:
        """/"""
        # 1. Extract compliance core data
        compliance_percent = compliance_result.get("consistency_percentage", 0.0)
        compliance_coefficient = compliance_percent / 100    # 0-1

        # 2. Initialize result dict
        fusion_result = {
            "合规性基础数据": {
                "合规性百分比": round(compliance_percent, 2),
                "合规性系数(0-1)": round(compliance_coefficient, 4)
            },
            "融合规则": "综合分数 = 合规性系数(0-1) × 对应质量分数(0-100分)",
            "综合评分结果": {}
        }

        # 3. Compute overall score by config type
        if self.quality_eval_type == "agent":
            # Agent quality score only
            agent_score = self._get_single_quality_score(quality_eval, "agent")
            overall_score = round(compliance_coefficient * agent_score, 2)
            fusion_result["质量分数详情"] = {
                "评价类型": "智能体评价",
                "质量分数(0-100)": round(agent_score, 2)
            }
            fusion_result["综合评分结果"] = {
                "综合分数": overall_score,
                "综合等级": self._get_level(overall_score)
            }

        elif self.quality_eval_type == "metric":
            # Metric quality score only
            metric_score = self._get_single_quality_score(quality_eval, "metric")
            overall_score = round(compliance_coefficient * metric_score, 2)
            fusion_result["质量分数详情"] = {
                "评价类型": "指标评价(PPL)",
                "质量分数(0-100)": round(metric_score, 2)
            }
            fusion_result["综合评分结果"] = {
                "综合分数": overall_score,
                "综合等级": self._get_level(overall_score)
            }

        elif self.quality_eval_type == "both":
            # Both types: compute separately
            agent_score = self._get_single_quality_score(quality_eval, "agent")
            metric_score = self._get_single_quality_score(quality_eval, "metric")

            agent_overall = round(compliance_coefficient * agent_score, 2)
            metric_overall = round(compliance_coefficient * metric_score, 2)

            fusion_result["质量分数详情"] = {
                "智能体评价": {"质量分数(0-100)": round(agent_score, 2)},
                "指标评价(PPL)": {"质量分数(0-100)": round(metric_score, 2)}
            }

            if self.both_include:
                # and
                fusion_result["综合评分结果"] = {
                    "智能体+合规性": {
                        "综合分数": agent_overall,
                        "综合等级": self._get_level(agent_overall)
                    },
                    "指标+合规性": {
                        "综合分数": metric_overall,
                        "综合等级": self._get_level(metric_overall)
                    }
                }
            else:
                fusion_result["综合评分结果"] = {
                    "综合分数": agent_overall,
                    "综合等级": self._get_level(agent_overall),
                    "说明": "已配置both_include_in_overall=false, 仅展示智能体+合规性综合分"
                }

        # 4. Log output
        self.logger.info(f"分数融合完成 - 合规性系数: {compliance_coefficient:.4f}, 质量分数详情: {fusion_result['质量分数详情']}, 综合结果: {fusion_result['综合评分结果']}")
        return fusion_result