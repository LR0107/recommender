from openai import OpenAI
import re

# ========== DeepSeek 配置 ==========
client = OpenAI(
    api_key="sk-0f43bd9617cf4d21b3e107bd2dde8cd3",  # 替换为你的 DeepSeek API Key
    base_url="https://api.deepseek.com"
)

# ========== 示例输入 ==========
user_context = {
    "click_history": ["Nike 跑鞋", "Adidas 运动上衣", "Lululemon 健身裤"],
    "search_query": "轻便运动装备",
    "gender": "female",
    "price_range": "200-400"
}

recommendations = [
    "Nike Air Max",
    "Under Armour 运动背心",
    "华为 Watch",
    "Adidas 慢跑短裤",
    "Gucci 高跟鞋"
]

# ========== 分类映射 ==========
category_map = {
    "Nike Air Max": "鞋",
    "Under Armour 运动背心": "上衣",
    "华为 Watch": "设备",
    "Adidas 慢跑短裤": "裤子",
    "Gucci 高跟鞋": "鞋"
}

# ========== Prompt 构造 ==========
def build_prompt(user_ctx, recs):
    history = "、".join(user_ctx["click_history"])
    prompt = f"""你是一位推荐分析专家。

请基于以下用户信息，分析推荐列表中每一项与用户偏好的匹配程度，并写出自然语言解释，不要打分：

- 用户最近浏览产品：{history}
- 搜索关键词：{user_ctx['search_query']}
- 性别：{user_ctx['gender']}
- 价格偏好：{user_ctx['price_range']} 元

推荐列表：
{chr(10).join([f"{i+1}. {item}" for i, item in enumerate(recs)])}

请为每条推荐生成自然语言解释，说明是否契合、是否合理、是否偏离偏好或风格、是否矛盾。"""
    return prompt

# ========== LLM 调用 ==========
def get_llm_explanations(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位推荐系统语义评估助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# ========== 指标计算 ==========
def calculate_match_score(lines):
    scores = []
    for line in lines:
        if any(k in line for k in ["完美契合", "强相关", "完全匹配", "逻辑最严密"]):
            scores.append(2)
        elif any(k in line for k in ["合理推荐", "匹配", "相关", "同一品牌", "补充", "拓展"]):
            scores.append(1)
        elif any(k in line for k in ["偏离", "弱关联", "无关", "突兀", "矛盾", "违背", "完全不符"]):
            scores.append(0)
    return round((sum(scores) / (2 * len(lines))) * 10, 2) if scores else 0.0

def calculate_explanation_rate(lines):
    count = sum(1 for line in lines if any(k in line for k in [
        "因为", "由于", "所以", "属于", "构成", "匹配",
        "合理性", "契合点", "功能", "品牌延续", "关键词响应"
    ]))
    return round(count / len(lines), 2) if lines else 0.0

def calculate_contradiction_count(lines):
    count = sum(1 for line in lines if any(k in line for k in [
        "矛盾", "违背", "完全偏离", "根本不符", "奢侈品牌", "高价", "冲突"
    ]))
    return round(count / len(lines), 2) if lines else 0.0

def calculate_diversity_index(categories):
    return round(len(set(categories)) / len(categories), 2) if categories else 0.0

def calculate_novelty_index(recommendations, history):
    new_items = [item for item in recommendations if all(h not in item for h in history)]
    return round(len(new_items) / len(recommendations), 2) if recommendations else 0.0

def calculate_total_score(match, diversity, novelty, contradiction,
                          alpha=0.5, beta=0.2, gamma=0.2, delta=0.1):
    total = alpha * match + beta * diversity * 10 + gamma * novelty * 10 - delta * contradiction * 10
    return round(total, 2)

# ========== 主流程 ==========
def main():
    # 构造 prompt 并调用 LLM
    prompt = build_prompt(user_context, recommendations)
    llm_output = get_llm_explanations(prompt)
    print("LLM 返回解释：\n", llm_output)

    # 提取解释段落
    lines = re.findall(r"\d+\.\s*\**.*?\**\s*\n\s*(.*)", llm_output)
    if not lines:
        # 回退：按换行分段
        lines = llm_output.strip().split("\n")

    # 分类信息
    categories = [category_map.get(item, "其他") for item in recommendations]

    # 计算各指标
    match_score = calculate_match_score(lines)
    explanation_rate = calculate_explanation_rate(lines)
    contradiction_count = calculate_contradiction_count(lines)
    diversity_index = calculate_diversity_index(categories)
    novelty_index = calculate_novelty_index(recommendations, user_context["click_history"])
    total_score = calculate_total_score(
        match_score, diversity_index, novelty_index, contradiction_count
    )

    # 输出结果
    print("\n语义评估指标：")
    print(f"- Match Score (契合度): {match_score}")
    print(f"- Explanation Rate (解释率): {explanation_rate}")
    print(f"- Diversity Index (多样性): {diversity_index}")
    print(f"- Novelty Index (新颖性): {novelty_index}")
    print(f"- Contradiction Count (矛盾比例): {contradiction_count}")
    print(f"- Total Score (加权总分): {total_score}")

if __name__ == "__main__":
    main()
