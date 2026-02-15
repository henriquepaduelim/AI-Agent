from ollama import chat

MODEL = "llama3.2:3b"


def run_agent(task):
    print("Processando resposta...\n")
    stream = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Você é um assistente inteligente e objetivo."},
            {"role": "user", "content": task},
        ],
        stream=True,
        options={"num_predict": 300},
    )

    parts = []
    for chunk in stream:
        content = chunk["message"]["content"]
        if content:
            print(content, end="", flush=True)
            parts.append(content)
    print()
    return "".join(parts)


if __name__ == "__main__":
    while True:
        try:
            user_input = input("Digite a tarefa: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not user_input:
            continue

        try:
            run_agent(user_input)
        except Exception as exc:
            print(
                f"\nErro ao consultar o Ollama: {exc}\n"
                "Verifique se o servidor está ativo com `ollama serve`."
            )
