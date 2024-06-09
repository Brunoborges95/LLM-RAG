import requests # type: ignore
import json

perguntas = [
    "1. Olá, qual é o seu nome?",
    "2. Pode me falar um pouco sobre sua experiência profissional?",
    "3. Quais são suas principais habilidades?",
    "4. Por que você quer trabalhar conosco?",
    "5. Onde você se vê em cinco anos?"
]

base_url = "http://127.0.0.1:5000"
# Upload de arquivos e configuração da cadeia QA

upload_url = f"{base_url}/upload"
with open("configs/config_test.json", "r", encoding="utf-8") as file:
    data = json.load(file)
data["path_selected_filename"] = "docs\\test_cv\\Curriculo_Christian_Freitas_2024.pdf"

for i in range(len(perguntas)):
    if i==0:
        print(f"Entrevistador: {perguntas[i]}")
    resposta = input("Entrevistado: ")
    if 0<i<4:
        data["prompt_template"] = (
            f"""A questão é a resposta do Entrevistado para a pergunta {i} do conjunto de {perguntas}.
                Você é um entrevistador muito mal-humorado e irônico, e não gosta muito do candidato. 
                Você como Entrevistador deve reagir à resposta e fazer a pergunta {i+1}, 
                podendo usar o contexto para enriquecer a pergunta.
                """
        )
        data["prompt_template"] = data["prompt_template"]+" {context}."
    if i==4:
        data["prompt_template"] = (
            f"""A questão é a resposta do Entrevistado para a pergunta {i} do conjunto de {perguntas}.
                Você é um entrevistador muito mal-humorado e irônico, e não gosta muito do candidato.
                Você como Entrevistador deve reagir à resposta e finalizar a entrevista,
                podendo usar o contexto para enriquecer a pergunta.
                """
        )
        data["prompt_template"] = data["prompt_template"]+" {context}."
    requests.post(upload_url, json=data)
    chat_url = f"{base_url}/chat"
    question = {"message": f"{resposta}"}
    response = requests.post(chat_url, json=question)

    print(f"Entrevistador: {response.json()}")