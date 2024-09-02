import sys
import os
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import pytz
from urllib.parse import quote


# Redirecionar stderr para nul para suprimir os erros
sys.stderr = open(os.devnull, 'w')

from selenium import webdriver
from selenium.webdriver.edge.options import Options

edge_options = Options()
edge_options.add_argument('--headless')
edge_options.add_argument('--disable-gpu')
edge_options.add_argument('--no-sandbox')
driver = webdriver.Edge(options=edge_options)

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
    url = f"https://br.investing.com/search/?q={quote(query)}&tab=news"
    driver.get(url)

    # Esperar pelo carregamento dos links
    try:
        links = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.img"))
        )
    except:
        print(f"Nenhum resultado encontrado para a query '{query}'")
        return []

    news_urls = []
    for link in links:
        href = link.get_attribute('href')
        if href.startswith("/"):
            href = "https://br.investing.com" + href
        print(f"URL encontrada: {href}")
        news_urls.append(href)
    
    return news_urls

def clean_text_with_links(element):
    # O Selenium não precisa disso, mas você pode ajustar como preferir
    return element.text

def get_news_text(url):
    driver.get(url)

    # Esperar pelo carregamento do conteúdo principal
    try:
        main_content = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.article_container"))
        )
    except:
        print(f"Não foi possível encontrar o conteúdo principal para a URL '{url}'")
        return None

    # Obter o título da notícia
    try:
        data_hora_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.mt-2.flex.flex-col.gap-2.text-xs.md\\:mt-2\\.5.md\\:gap-2\\.5 div.flex.flex-row.items-center span"))
        )
        data_hora = data_hora_element.text
        publication_date = data_hora
    except:
        print(f"Não foi possível encontrar a data de publicação para a URL '{url}'")
        return None
    
    # Encontrar o título da notícia
    try:
        titulo_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.mx-0.mt-1 h1#articleTitle"))
        )
        titulo = titulo_element.text
        news_title = titulo
    except:
        print("Título da notícia não encontrado")

    
    

    # Obter o texto da notícia
    content_paragraphs = main_content.find_elements(By.CSS_SELECTOR, "p")

    news_content = []
    for paragraph in content_paragraphs:
        cleaned_paragraph = clean_text_with_links(paragraph)
        news_content.append(cleaned_paragraph)
    
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
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    scores = softmax.cpu().detach().numpy()

    positivo = decimal_to_percentage(scores[0][0])
    negativo = decimal_to_percentage(scores[0][1])
    neutro = decimal_to_percentage(scores[0][2])

    return positivo, negativo, neutro

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

# Fechar o navegador
driver.quit()
