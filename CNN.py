import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
from urllib.parse import urlparse, parse_qs, quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar o modelo e o tokenizer
model_name = "lucas-leme/FinBERT-PT-BR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Faixa de datas com timezone
brazil_tz = pytz.timezone('America/Sao_Paulo')
start_date = brazil_tz.localize(datetime.strptime("01/06/2024", "%d/%m/%Y"))
end_date = brazil_tz.localize(datetime.strptime("01/08/2024", "%d/%m/%Y"))

# Função para formatar a data para a URL de busca
def format_date_for_url(date):
    return quote(date.strftime("%Y-%m-%dT%H:%M:%S%z"))

def get_news_urls(query):
    # Formatar a URL de busca com a query fornecida e a faixa de datas
    url = f"https://www.cnnbrasil.com.br/?s={quote(query)}&orderby=date&order=desc"
    
    # Enviar uma solicitação GET para a URL
    response = requests.get(url)
    
    # Verificar se a solicitação foi bem-sucedida
    if response.status_code != 200:
        print(f"Erro ao acessar a página: {response.status_code}")
        return []
    
    # Parsear o conteúdo HTML da página
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Encontrar todos os elementos que contêm os URLs das notícias
    results_content = soup.find_all('a', class_='home__list__tag')
    if not results_content:
        print(f"Nenhum resultado encontrado para a query '{query}'")
        return []

    news_urls = []
    for item in results_content:
        href = item['href']
        # Obter o link a partir do segundo caractere
        link = href[1:]
        if not link.startswith("http"):
            link = "h" + link
        news_urls.append(link)
    
    return news_urls


def clean_text_with_links(element):
    # Adicionar espaços antes e depois de links clicáveis
    for a in element.find_all('a'):
        a.insert_before(" ")
        a.insert_after(" ")
    return element

def get_news_text(url):
    # Enviar uma solicitação GET para a URL da notícia
    response = requests.get(url)
    
    # Verificar se a solicitação foi bem-sucedida
    if response.status_code != 200:
        print(f"Erro ao acessar a notícia: {response.status_code}")
        return None
    
    # Parsear o conteúdo HTML da página
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Encontrar o título da notícia
    main_content = soup.find('main', class_='posts col__list')
    if not main_content:
        print(f"Não foi possível encontrar o conteúdo principal para a URL '{url}'")
        return None
    
    # Obter o título da notícia
    title_tag = main_content.find('h1', class_='single-header__title')
    if not title_tag:
        print(f"Não foi possível encontrar o título para a URL '{url}'")
        return None
    news_title = title_tag.get_text(strip=True)
    
    # Obter a data e hora da publicação
    time_tag = main_content.find('time', class_='single-header__time')
    if not time_tag:
        print(f"Não foi possível encontrar a data de publicação para a URL '{url}'")
        return None
    date_text = time_tag.get_text()
    date_str = date_text
    publication_date = (date_str)
    
    
    # Obter o texto da notícia
    content_paragraphs = main_content.find_all('p')
    
    news_content = []
    for paragraph in content_paragraphs:
        cleaned_paragraph = clean_text_with_links(paragraph)
        news_content.append(cleaned_paragraph.get_text(" ", strip=True))
    
    full_content = "\n".join(news_content)
    
    return {
        "title": news_title,
        "publication_date": publication_date,
        "content": full_content
    }


def decimal_to_percentage(decimal_number):
    percentage = decimal_number * 100
    return f"{percentage:.1f}%"

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=512)
    outputs = model(**inputs)
    logits=outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    scores = softmax.cpu().detach().numpy()

    positivo=decimal_to_percentage(scores[0][0])
    negativo=decimal_to_percentage(scores[0][1])
    neutro=decimal_to_percentage(scores[0][2])

    return positivo,negativo, neutro

# Lista de palavras para buscar notícias
queries = ["ELON MUSK", "MARK ZUCKERBERG","LUIZA TRAJANO","BETO SICUPIRA"]

# Obter URLs de notícias para cada palavra na lista
for query in queries:
    print(f"RESULTADOS PARA: '{query}':")
    print()
    news_urls = get_news_urls(query)
    if not news_urls:
        continue
    for url in news_urls:
        news_data = get_news_text(url)
        if news_data:
            positivo, negativo, neutro = analyze_sentiment(news_data['content'])
            print(f"Título: {news_data['title']}")
            print(f"Data de Publicação: {news_data['publication_date']}")
            print(f"URL: {url}")
            print(f"SENTIMENTOS: Positivo: {positivo}, Negativo: {negativo}, Neutro: {neutro}")
            print()
            print()
