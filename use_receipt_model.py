
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de exemplo para usar o modelo treinado para extrair informações de notas fiscais
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Caminho para o modelo treinado
MODEL_DIR = "receipt_ner_model"

def extract_info_from_receipt(text):
    """
    Extrai informações de um texto de nota fiscal usando o modelo treinado
    """
    # Carrega o modelo e o tokenizador
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    
    # Prepara o texto
    tokens = text.split()
    inputs = tokenizer(tokens, truncation=True, is_split_into_words=True, return_tensors="pt", padding=True)
    
    # Faz a previsão
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtém as previsões
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_token_class_ids = predictions[0].tolist()
    
    # Mapeia as previsões para os tokens
    predicted_entities = []
    
    word_ids = inputs.word_ids()
    previous_word_idx = None
    entity = {"type": None, "text": ""}
    
    # Carrega o mapeamento de ID para etiqueta
    id2label = model.config.id2label
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        
        if word_idx != previous_word_idx:
            if entity["type"] is not None:
                predicted_entities.append(entity.copy())
                entity = {"type": None, "text": ""}
            
            predicted_class_id = predicted_token_class_ids[idx]
            
            if predicted_class_id > 0:  # 0 é "O" (Outside)
                entity_label = id2label[predicted_class_id]
                
                if entity_label.startswith("B-"):
                    entity_type = entity_label[2:]
                    entity["type"] = entity_type
                    entity["text"] = tokens[word_idx]
            
        elif entity["type"] is not None:
            predicted_class_id = predicted_token_class_ids[idx]
            entity_label = id2label[predicted_class_id]
            
            if entity_label.startswith("I-") and entity_label[2:] == entity["type"]:
                entity["text"] += " " + tokens[word_idx]
        
        previous_word_idx = word_idx
    
    if entity["type"] is not None:
        predicted_entities.append(entity)
    
    # Formatação do resultado no JSON específico solicitado
    result = {
        "tipo": "",
        "cnpj": "",
        "chave_acesso": "",
        "data_emissao": "",
        "valor_total_pago": "",
        "numero_documento": "",
        "serie": ""
    }
    
    # Preenche o resultado com as entidades extraídas
    for entity in predicted_entities:
        entity_type = entity["type"]
        entity_text = entity["text"].strip()
        
        if entity_type == "TIPO_DOCUMENTO":
            result["tipo"] = entity_text
        elif entity_type == "CNPJ":
            # Limpa o CNPJ para conter apenas dígitos
            cnpj = re.sub(r'[^0-9]', '', entity_text)
            result["cnpj"] = cnpj
        elif entity_type == "CHAVE_ACESSO":
            # Limpa a chave para conter apenas dígitos
            chave = re.sub(r'[^0-9]', '', entity_text)
            result["chave_acesso"] = chave
        elif entity_type == "DATA_EMISSAO":
            result["data_emissao"] = entity_text
        elif entity_type == "VALOR_TOTAL":
            # Padroniza o formato do valor
            valor = entity_text.replace(',', '.')
            if not valor.startswith('R$') and not re.match(r'^\d+\.\d{2}$', valor):
                # Se não tiver o formato correto, tenta extrair o número
                match = re.search(r'(\d+[,.]\d{2})', valor)
                if match:
                    valor = match.group(1).replace(',', '.')
            result["valor_total_pago"] = valor
        elif entity_type == "NUMERO_DOCUMENTO":
            result["numero_documento"] = entity_text
        elif entity_type == "SERIE":
            result["serie"] = entity_text
    
    return result

def main():
    # Exemplo de texto de nota fiscal
    exemplo_texto = """Coloque aqui o texto da nota fiscal para testar"""
    
    # Extrai informações
    resultado = extract_info_from_receipt(exemplo_texto)
    
    # Exibe o resultado formatado
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
