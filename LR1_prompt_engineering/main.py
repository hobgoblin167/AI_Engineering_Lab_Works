import sys
import re
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "/content/drive/MyDrive/models/qwen3-14b/Qwen3-14B-Q4_K_M.gguf"

N_CTX = 6144
MAX_TOKENS = 6144
TEMPERATURE = 0.2
TOP_P = 0.95

START_IDX = 0 # ← НАЧИНАЕМ С 49-й СТРОКИ (0-based)

SYSTEM_PROMPT_TEMPLATE = (
    "Ты — эксперт в области {category}. "
    "Твоя задача — выбрать один правильный вариант ответа."
)

USER_PROMPT_TEMPLATE = """
Вопрос:

{question}

Варианты ответа:
{options}

Формат ответа (обязательно):
ответ: <номер варианта>

"""

# ----------------------------
# UTILS
# ----------------------------
def extract_answer(text: str):
    match = re.search(r"ответ\s*:\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def format_options(raw_options: str):
    options = re.findall(r"'([^']*)'", raw_options)

    formatted = []
    for i, opt in enumerate(options):
        opt = opt.strip()
        if opt:
            formatted.append(f"{i}. {opt}")

    if not formatted:
        formatted.append(f"0. {raw_options.strip()}")

    return "\n".join(formatted)


# ----------------------------
# MAIN
# ----------------------------
def main(input_csv, output_csv):
    print("🚀 Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=-1,
        verbose=False
    )

    print("📄 Loading data...")
    df = pd.read_csv(input_csv)

    answers = []

    # 🔒 заглушки для первых 49 вопросов
    for _ in range(START_IDX):
        answers.append(0)

    print(f"▶️ Starting from row {START_IDX}")

    for idx, row in tqdm(
        df.iloc[START_IDX:].iterrows(),
        total=len(df) - START_IDX
    ):
        question = row["question"]
        category = row["category"]
        options = format_options(row["options"])

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(category=category)

        prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            options=options
        )

        print("\n" + "=" * 80)
        print(f"🧠 Question ID: {row.iloc[0]}")
        print(f"📚 Category: {category}")
        print(prompt)

        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        text = output["choices"][0]["message"]["content"]

        print("📝 MODEL OUTPUT:")
        print(text)

        answer = extract_answer(text)

        if answer is None:
            print("⚠️ Не удалось извлечь ответ, ставлю 0")
            answer = 0

        print(f"✅ FINAL ANSWER: {answer}")
        answers.append(answer)

    df["answer"] = answers

    print(f"\n💾 Saving results to {output_csv}")
    df.to_csv(output_csv, index=False)

    print("🎉 Done!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py input.csv output.csv")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
