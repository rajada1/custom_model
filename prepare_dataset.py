#!/usr/bin/env python3
"""
Script para preparar o dataset de cupons fiscais para treinamento.
Este script extrai os textos do arquivo Dart e os converte para um formato adequado
para treinamento de um modelo NER (Named Entity Recognition).
"""
import re
import json
import os
from typing import List, Dict, Any, Tuple

# Define as entidades que queremos extrair
ENTITIES = [
    "TIPO_DOCUMENTO",
    "CNPJ",
    "CHAVE_ACESSO",
    "DATA_EMISSAO",
    "VALOR_TOTAL",
    "NUMERO_DOCUMENTO",
    "SERIE"
]

# Mapeamento dos códigos de modelo de documento para seus tipos
DOCUMENT_TYPE_MAP = {
    '01': 'NF',
    '02': 'NFVC',
    '04': 'NFP',
    '06': 'NFCE',
    '07': 'NFST',
    '08': 'CTRC',
    '09': 'CTAC',
    '10': 'CA',
    '11': 'CTFC',
    '13': 'BPR',
    '14': 'BPA',
    '15': 'BPNB',
    '16': 'BPF',
    '17': 'DT',
    '18': 'RMD',
    '20': 'OCC',
    '21': 'NFSC',
    '22': 'NFST',
    '23': 'GNRE',
    '24': 'AC',
    '25': 'MC',
    '26': 'CTMC',
    '27': 'NFTFC',
    '28': 'NFCG',
    '29': 'NFCA',
    '30': 'BRP',
    '2D': 'CFECF',
    '2E': 'BPEC',
    '55': 'NFE',
    '57': 'CTE',
    '59': 'CF',
    '60': 'CFEECF',
    '65': 'NFCE',
    '67': 'CTE',
    '8B': 'CTCA',
}

def extract_texts_from_dart(dart_file_path: str) -> List[str]:
    """
    Extrai os textos do arquivo Dart.
    """
    with open(dart_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Encontra todas as strings multi-linhas no arquivo Dart
    pattern = r"static const \w+ = r?'''(.*?)''';"
    matches = re.findall(pattern, content, re.DOTALL)
    
    texts = []
    for match in matches:
        # Normaliza o texto, removendo espaços extras e caracteres especiais
        text = match.strip()
        text = re.sub(r'\s+', ' ', text)
        texts.append(text)
    
    return texts

def is_valid_access_key(key: str) -> bool:
    """
    Verifica se a chave de acesso é válida com base nos prefixos conhecidos.
    """
    # Lista de prefixos válidos para chaves de acesso
    valid_prefixes = [
        "11", "12", "13", "14", "15", "16", "17", "21", "22", "23", "24", "25", 
        "26", "27", "28", "29", "31", "32", "33", "35", "41", "42", "43", "50", 
        "51", "52", "53"
    ]
    
    # Verifica se a chave tem 44 dígitos e começa com um prefixo válido
    if len(key) == 44 and key[:2] in valid_prefixes:
        return True
    return False

def extract_date_info_from_key(key: str) -> tuple:
    """
    Extrai o ano e mês da chave de acesso.
    
    Args:
        key: Chave de acesso de 44 dígitos
        
    Returns:
        Tupla contendo (ano, mês) da chave
    """
    if len(key) != 44:
        return None, None
    
    # Ano está nas posições 2-4 (2 dígitos)
    year = key[2:4]
    # Mês está nas posições 4-6 (2 dígitos)
    month = key[4:6]
    
    # Converte para números, adiciona 2000 ao ano para ter formato de 4 dígitos
    try:
        year = int(year)
        month = int(month)
        if 0 < month <= 12:  # Valida o mês
            # Formata para 4 dígitos (assumindo anos 2000)
            full_year = 2000 + year
            return str(full_year), f"{month:02d}"
    except ValueError:
        pass
    
    return None, None

def extract_document_type_from_key(key: str) -> str:
    """
    Extrai o tipo de documento a partir da chave de acesso.
    
    Args:
        key: Chave de acesso de 44 dígitos
        
    Returns:
        String com o tipo de documento ou string vazia se não for reconhecido
    """
    if len(key) != 44:
        return ""
    
    # O tipo de documento está nas posições 20-22
    model_code = key[20:22]
    
    # Retorna o tipo de documento correspondente ou string vazia se não encontrado
    return DOCUMENT_TYPE_MAP.get(model_code, "")

def clean_currency_value(value: str) -> str:
    """
    Limpa e padroniza um valor monetário.
    
    Args:
        value: Valor monetário como string (ex: "114,54", "R$ 50.00", etc.)
        
    Returns:
        Valor padronizado como string decimal com ponto (ex: "114.54")
    """
    if not value:
        return ""
    
    # Remove qualquer caractere que não seja dígito, vírgula ou ponto
    value = re.sub(r'[^\d,.]', '', value)
    
    # Substitui vírgula por ponto para padronização decimal
    value = value.replace(',', '.')
    
    # Se houver mais de um ponto, mantém apenas o último (caso de milhares)
    if value.count('.') > 1:
        parts = value.split('.')
        last_part = parts[-1]
        rest = ''.join(parts[:-1]).replace('.', '')
        value = f"{rest}.{last_part}"
    
    return value

def create_annotations(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Extrai as informações solicitadas dos textos e retorna um dataset estruturado
    no formato JSON especificado.
    """
    annotations = []
    
    for i, text in enumerate(texts):
        # Inicializa o dicionário de dados com valores padrão
        data = {
            "tipo": "",
            "cnpj": "",
            "chave_acesso": "",
            "data_emissao": "",
            "valor_total_pago": "",
            "numero_documento": "",
            "serie": ""
        }
        
        # Extrai chave de acesso (44 dígitos) - PRIORIDADE ALTA
        chave_patterns = [
            r'(\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4})',  # Com espaços
            r'(\d{44})'  # Sem espaços
        ]
        
        # Procura pela chave de acesso primeiro
        chave_acesso = ""
        for pattern in chave_patterns:
            chave_matches = re.findall(pattern, text)
            if chave_matches:
                for match in chave_matches:
                    # Remove todos os caracteres não numéricos
                    potential_key = re.sub(r'[^0-9]', '', match)
                    if len(potential_key) == 44 and is_valid_access_key(potential_key):
                        chave_acesso = potential_key
                        data["chave_acesso"] = chave_acesso
                        break
                if chave_acesso:
                    break
        
        # Variáveis para armazenar informações extraídas da chave
        key_year, key_month = None, None
        document_type_from_key = ""
        
        # Se encontrou a chave de acesso, extrai informações dela
        if chave_acesso:
            # Extrai CNPJ da chave de acesso (posições 6-20)
            data["cnpj"] = chave_acesso[6:20]
            
            # Extrai número do documento da chave de acesso (posições 31-37)
            data["numero_documento"] = str(int(chave_acesso[31:37]))  # Remove zeros à esquerda
            
            # Extrai ano e mês da chave de acesso
            key_year, key_month = extract_date_info_from_key(chave_acesso)
            
            # Extrai o tipo de documento da chave de acesso
            document_type_from_key = extract_document_type_from_key(chave_acesso)
            if document_type_from_key:
                data["tipo"] = document_type_from_key
        
        # Se não encontrou o tipo de documento na chave ou não encontrou a chave,
        # tenta identificar pelo texto (método original)
        if not document_type_from_key:
            tipo_documento = ""
            if "NFC-e" in text or "NOTA FISCAL DE CONSUMIDOR ELETRONICA" in text or "NFCe" in text:
                tipo_documento = "NFCE"
            elif "NF-e" in text or "NOTA FISCAL ELETRONICA" in text or "NFe" in text:
                tipo_documento = "NFE"
            elif "SAT" in text or "CUPOM FISCAL ELETRÔNICO - SAT" in text:
                tipo_documento = "SAT"
            elif "CT-e" in text or "CTE" in text or "CONHECIMENTO DE TRANSPORTE" in text:
                tipo_documento = "CTE"
            elif "CUPOM FISCAL" in text:
                tipo_documento = "CF"
            elif "DANFE" in text:
                tipo_documento = "NFE"
            data["tipo"] = tipo_documento
        
        # Se não encontrou a chave ou outras informações, continua com os métodos alternativos
        if not chave_acesso:
            # Tenta extrair CNPJ pelo método alternativo
            cnpj_matches = re.findall(r'CNPJ[:\s]*(\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[/\.\s]?\d{4}[-\.\s]?\d{2}|\d{14})', text, re.IGNORECASE)
            if cnpj_matches:
                cnpj = cnpj_matches[0]
                # Remove todos os caracteres não numéricos
                cnpj = re.sub(r'[^0-9]', '', cnpj)
                data["cnpj"] = cnpj
            
            # Tenta extrair número do documento pelo método alternativo apenas se não foi extraído da chave
            if not data["numero_documento"]:
                numero_patterns = [
                    r'N[°º\.]?[:\s]*(?:0*)(\d+)',
                    r'Nº[:\s]*(?:0*)(\d+)',
                    r'N[°\.]?[:\s]*(?:0*)(\d+)',
                    r'n°[:\s]*(?:0*)(\d+)',
                    r'(?:NF(?:C|E)?-e|SAT)[:\s]*(?:n[°\.]?)?[:\s]*(?:0*)(\d+)',
                    r'Extrato\s+(?:N[°º\.]?)?[:\s]*(?:0*)(\d+)'
                ]
                
                for pattern in numero_patterns:
                    numero_matches = re.findall(pattern, text, re.IGNORECASE)
                    if numero_matches:
                        data["numero_documento"] = numero_matches[0]
                        break
        
        # Extrai data de emissão
        data_patterns = [
            r'Data\s+de\s+[Ee]miss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})',
            r'[Ee]miss[ãa]o[:\s]*(\d{2}/\d{2}/\d{4})',
            r'DATA\s+DE\s+EMISSÃO[:\s]*(\d{2}/\d{2}/\d{4})',
            r'(\d{2}/\d{2}/\d{4})\s*-\s*\d{2}:\d{2}',  # Padrão data - hora
            r'(\d{2}/\d{2}/\d{4})'  # Qualquer data no formato DD/MM/AAAA
        ]
        
        # Armazena todas as datas encontradas para posterior validação
        all_dates = []
        for pattern in data_patterns:
            data_matches = re.findall(pattern, text, re.IGNORECASE)
            all_dates.extend(data_matches)
        
        # Se temos informações de data da chave de acesso, vamos usá-las para validar
        if key_year and key_month and all_dates:
            best_date = None
            for date_str in all_dates:
                # Extrair componentes da data encontrada (formato DD/MM/AAAA)
                day, month, year = date_str.split('/')
                
                # Verificar se o ano e mês correspondem com os da chave de acesso
                if year[-2:] == key_year[-2:] and month == key_month:
                    best_date = date_str
                    break
            
            # Se encontrou uma data que corresponde, use-a; caso contrário, use a primeira data encontrada
            if best_date:
                data["data_emissao"] = best_date
            elif all_dates:
                data["data_emissao"] = all_dates[0]
        elif all_dates:
            # Se não temos informações da chave ou nenhuma data correspondeu, use a primeira data encontrada
            data["data_emissao"] = all_dates[0]
        
        # Extrai valor total com padrões aprimorados
        valor_patterns = [
            # Padrões específicos com R$
            r'TOTAL\s*R\$\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            r'[Tt]otal\s*R\$\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            r'[Vv]alor\s+[Tt]otal\s*R\$\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            r'[Vv]alor\s+[aA]\s+[Pp]agar\s*R\$\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            
            # Padrões com variações de pontuação
            r'TOTAL\s*[\.:]\s*R?\$?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            r'[Tt]otal\s*[\.:]\s*R?\$?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            
            # Padrões com "Valor a Pagar" e variações
            r'[Vv]alor\s+[aA]\s+[Pp]agar\s*R?\$?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            r'[Vv]alor\s+[Pp]ago\s*R?\$?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            
            # Variações com o R$ após o valor
            r'TOTAL\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\s*R\$',
            r'[Tt]otal\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\s*R\$',
            
            # Padrões para captação de totais sem R$ explícito
            r'[Vv]alor\s+[Tt]otal\s*[\.:]*\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            r'TOTAL\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            
            # Padrões para linhas de totais com formatação diferente
            r'TOTAL\s+(?:R\$)?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            
            # Padrões para capturar valores próximos a palavras chave
            r'VALOR\s+(?:TOTAL|PAGO)[\s:]*R?\$?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            
            # Padrões com abreviações e variações
            r'VL\.?\s*(?:TOTAL|TOT)[\s:]*R?\$?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            
            # Padrão para capturar valores em contexto de pagamento
            r'Cartão\s+de\s+(?:Crédito|Débito|Credito|Debito)[\s:]*R?\$?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',
            r'VALOR[\s:]*R?\$?\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})'
        ]
        
        # Variável para armazenar o melhor resultado encontrado
        best_value = None
        
        # Procura por todos os valores que correspondem aos padrões
        all_values = []
        for pattern in valor_patterns:
            valor_matches = re.findall(pattern, text, re.IGNORECASE)
            if valor_matches:
                for match in valor_matches:
                    clean_value = clean_currency_value(match)
                    if clean_value:
                        all_values.append(clean_value)
        
        # Se encontramos valores, tenta selecionar o melhor
        if all_values:
            # Dá preferência a valores que aparecem mais de uma vez
            value_counts = {}
            for value in all_values:
                value_counts[value] = value_counts.get(value, 0) + 1
            
            # Ordena por frequência (decrescente) e depois por valor (decrescente)
            sorted_values = sorted(value_counts.items(), key=lambda x: (-x[1], -float(x[0])))
            best_value = sorted_values[0][0]
        
        # Define o valor encontrado
        if best_value:
            data["valor_total_pago"] = best_value
        
        # Extrai série do documento
        serie_patterns = [
            r'[Ss][ée]rie[:\s]*(\d+)',
            r'[Ss]erie[:\s]*(\d+)'
        ]
        
        for pattern in serie_patterns:
            serie_matches = re.findall(pattern, text, re.IGNORECASE)
            if serie_matches:
                data["serie"] = serie_matches[0]
                break
        
        # Cria a anotação no formato solicitado
        annotation = {
            "text": text,
            "data": data
        }
        
        annotations.append(annotation)
    
    return annotations

def save_annotations(annotations: List[Dict[str, Any]], output_file: str):
    """
    Salva as anotações em um arquivo JSON.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
        
def save_structured_data(annotations: List[Dict[str, Any]], output_file: str):
    """
    Salva apenas os dados estruturados extraídos, no formato solicitado.
    """
    extracted_data = [annotation["data"] for annotation in annotations]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)

def main():
    # Caminho para o arquivo Dart e diretório de saída
    dart_file_path = '/Users/matheus.oliveira/Documents/custom_model/dataset.dart'
    output_dir = '/Users/matheus.oliveira/Documents/custom_model/dataset_prepared'
    
    # Cria o diretório de saída se ele não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrai os textos do arquivo Dart
    texts = extract_texts_from_dart(dart_file_path)
    print(f"Extraídos {len(texts)} textos do arquivo Dart.")
    
    # Extrai os dados estruturados
    annotations = create_annotations(texts)
    
    # Salva os dados estruturados no formato JSON solicitado
    structured_data_file = os.path.join(output_dir, "extracted_data.json")
    save_structured_data(annotations, structured_data_file)
    print(f"Dados estruturados salvos em {structured_data_file}")
    
    # Também salva as anotações completas para uso posterior
    annotations_file = os.path.join(output_dir, "annotations.json")
    save_annotations(annotations, annotations_file)
    print(f"Anotações completas salvas em {annotations_file}")
    
    # Salva os textos brutos para referência
    raw_texts_file = os.path.join(output_dir, "raw_texts.txt")
    with open(raw_texts_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts):
            f.write(f"=== TEXTO {i+1} ===\n{text}\n\n")
    print(f"Textos brutos salvos em {raw_texts_file}")
    
    print("\nPróximos passos:")
    print("1. Verifique os dados extraídos no arquivo 'extracted_data.json'")
    print("2. Use os dados para treinar seu modelo ou para qualquer outra finalidade")

if __name__ == "__main__":
    main()
