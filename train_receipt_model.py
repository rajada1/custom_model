#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para treinar um modelo customizado de IA para extração de informações de notas fiscais
"""

import os
import json
import re
import random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from seqeval.metrics import classification_report, f1_score

# Configurações
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # Modelo pré-treinado em português
OUTPUT_DIR = "receipt_ner_model"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 25
SEED = 42

# Defina as etiquetas para extrair das notas fiscais
LABELS = [
    "O",  # Outside (não é uma entidade)
    "B-TIPO_DOCUMENTO", "I-TIPO_DOCUMENTO",  # Tipo de documento (NFCe, NFe, SAT, CTe...)
    "B-CNPJ", "I-CNPJ",  # CNPJ
    "B-CHAVE_ACESSO", "I-CHAVE_ACESSO",  # Chave de acesso
    "B-DATA_EMISSAO", "I-DATA_EMISSAO",  # Data de emissão
    "B-VALOR_TOTAL", "I-VALOR_TOTAL",  # Valor total
    "B-NUMERO_DOCUMENTO", "I-NUMERO_DOCUMENTO",  # Número do documento
    "B-SERIE", "I-SERIE"  # Série do documento
]

# Mapeamento de ID para etiqueta e vice-versa
id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}

def extract_dart_strings():
    """
    Extrai as strings de texto do arquivo dataset.dart
    """
    script_dir = Path(__file__).parent
    dart_file = script_dir / 'dataset.dart'
    
    with open(dart_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regex para encontrar strings multilinhas contidas entre r'''...''' ou '''...'''
    pattern = r"(?:r?'''([\s\S]*?)''')"
    matches = re.findall(pattern, content)
    
    return matches

def annotate_data_manually(texts):
    """
    Esta função anota os textos para reconhecimento de entidades específicas:
    - tipo de documento
    - CNPJ
    - chave de acesso
    - data de emissão
    - valor total
    - número do documento
    - série
    
    Em um cenário real, você usaria anotações manuais mais precisas ou
    uma ferramenta específica como Prodigy ou Doccano.
    """
    annotated_data = []
    
    for text in tqdm(texts, desc="Gerando anotações"):
        tokens = []
        ner_tags = []
        
        # Divide o texto em linhas
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Processa cada linha para extrair entidades
            
            # Tipo de documento
            tipo_doc_matches = False
            if "NFC-e" in line or "NOTA FISCAL DE CONSUMIDOR ELETRONICA" in line or "NFCe" in line:
                tipo_doc = "NFCe"
                tipo_doc_matches = True
            elif "NF-e" in line or "NOTA FISCAL ELETRONICA" in line or "NFe" in line or "DANFE" in line:
                tipo_doc = "NFe"
                tipo_doc_matches = True
            elif "SAT" in line or "CUPOM FISCAL ELETRÔNICO - SAT" in line:
                tipo_doc = "SAT"
                tipo_doc_matches = True
            elif "CT-e" in line or "CTE" in line or "CONHECIMENTO DE TRANSPORTE" in line:
                tipo_doc = "CTe"
                tipo_doc_matches = True
            elif "CUPOM FISCAL" in line:
                tipo_doc = "CF"
                tipo_doc_matches = True
                
            if tipo_doc_matches:
                words = line.split()
                for i, word in enumerate(words):
                    tokens.append(word)
                    if tipo_doc in word:
                        ner_tags.append("B-TIPO_DOCUMENTO")
                    elif i > 0 and ner_tags[-1] == "B-TIPO_DOCUMENTO":
                        ner_tags.append("I-TIPO_DOCUMENTO")
                    else:
                        ner_tags.append("O")
                continue
            
            # CNPJ
            cnpj_match = re.search(r'CNPJ[:\s]*([\d\.\/-]+)', line)
            if cnpj_match:
                words = line.split()
                for word in words:
                    tokens.append(word)
                    if cnpj_match.group(1) in word:
                        ner_tags.append("B-CNPJ")
                    else:
                        ner_tags.append("O")
                continue
            
            # Chave de acesso (44 dígitos)
            chave_matches = re.search(r'(\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}\s\d{4}|\d{44})', line)
            if chave_matches or re.search(r'chave\s+de\s+acesso', line.lower()):
                words = line.split()
                for word in words:
                    tokens.append(word)
                    if re.search(r'\d{44}', word.replace(" ", "")) or re.search(r'\d{4}\s\d{4}', word):
                        ner_tags.append("B-CHAVE_ACESSO")
                    elif "chave" in word.lower() or "acesso" in word.lower():
                        ner_tags.append("O")
                    else:
                        ner_tags.append("O")
                continue
            
            # Data de emissão
            data_emissao_patterns = [
                r'Data\s+de\s+[Ee]miss[aã]o[:\s]*(\d{2}/\d{2}/\d{4})',
                r'[Ee]miss[ãa]o[:\s]*(\d{2}/\d{2}/\d{4})',
                r'DATA\s+DE\s+EMISSÃO[:\s]*(\d{2}/\d{2}/\d{4})'
            ]
            
            data_match = None
            for pattern in data_emissao_patterns:
                match = re.search(pattern, line)
                if match:
                    data_match = match
                    break
                    
            if data_match or "emissão" in line.lower() or "emissao" in line.lower():
                words = line.split()
                for word in words:
                    tokens.append(word)
                    if data_match and data_match.group(1) in word:
                        ner_tags.append("B-DATA_EMISSAO")
                    elif re.search(r'\d{2}/\d{2}/\d{4}', word):
                        ner_tags.append("B-DATA_EMISSAO")
                    else:
                        ner_tags.append("O")
                continue
            
            # Valor total
            valor_patterns = [
                r'[Vv]alor\s+[Tt]otal\s*(?:da\s*[Nn]ota)?(?:\s*R\$)?[\s:]*(\d+[,.]\d{2})',
                r'TOTAL\s*R\$\s*(\d+[,.]\d{2})',
                r'[Vv]alor\s+[aA]\s+[Pp]agar\s*R\$\s*(\d+[,.]\d{2})',
                r'[Tt]otal\s+R\$\s*(\d+[,.]\d{2})'
            ]
            
            valor_match = None
            for pattern in valor_patterns:
                match = re.search(pattern, line)
                if match:
                    valor_match = match
                    break
                    
            if valor_match or "total" in line.lower() or "valor" in line.lower():
                words = line.split()
                for word in words:
                    tokens.append(word)
                    if valor_match and (valor_match.group(1) in word or "R$" + valor_match.group(1) in word):
                        ner_tags.append("B-VALOR_TOTAL")
                    elif re.search(r'\d+[,.]\d{2}', word) and ("total" in line.lower() or "valor" in line.lower()):
                        ner_tags.append("B-VALOR_TOTAL")
                    else:
                        ner_tags.append("O")
                continue
            
            # Número do documento
            numero_patterns = [
                r'N[°º\.]?[:\s]*(?:0*)(\d+)',
                r'Nº[:\s]*(?:0*)(\d+)',
                r'N[°\.]?[:\s]*(?:0*)(\d+)',
                r'n°[:\s]*(?:0*)(\d+)',
                r'(?:NF(?:C|E)?-e|SAT)[:\s]*(?:n[°\.]?)?[:\s]*(?:0*)(\d+)',
                r'Extrato\s+(?:N[°º\.]?)?[:\s]*(?:0*)(\d+)'
            ]
            
            numero_match = None
            for pattern in numero_patterns:
                match = re.search(pattern, line)
                if match:
                    numero_match = match
                    break
                    
            if numero_match:
                words = line.split()
                for word in words:
                    tokens.append(word)
                    if numero_match.group(1) in word:
                        ner_tags.append("B-NUMERO_DOCUMENTO")
                    else:
                        ner_tags.append("O")
                continue
            
            # Série do documento
            serie_patterns = [
                r'[Ss][ée]rie[:\s]*(\d+)',
                r'[Ss]erie[:\s]*(\d+)'
            ]
            
            serie_match = None
            for pattern in serie_patterns:
                match = re.search(pattern, line)
                if match:
                    serie_match = match
                    break
                    
            if serie_match or "serie" in line.lower() or "série" in line.lower():
                words = line.split()
                for word in words:
                    tokens.append(word)
                    if serie_match and serie_match.group(1) in word:
                        ner_tags.append("B-SERIE")
                    elif word.isdigit() and len(word) <= 3 and ("serie" in line.lower() or "série" in line.lower()):
                        ner_tags.append("B-SERIE")
                    else:
                        ner_tags.append("O")
                continue
            
            # Para outras linhas não identificadas
            words = line.split()
            for word in words:
                tokens.append(word)
                ner_tags.append("O")
        
        assert len(tokens) == len(ner_tags), f"Tokens e tags devem ter o mesmo tamanho: {len(tokens)} vs {len(ner_tags)}"
        
        annotated_data.append({"tokens": tokens, "ner_tags": ner_tags})
    
    return annotated_data

def prepare_dataset(annotated_data):
    """
    Prepara o conjunto de dados para treinamento, validação e teste
    """
    # Converte as tags NER de strings para IDs
    processed_data = []
    
    for item in annotated_data:
        # Converte as tags para IDs
        ner_ids = [label2id[tag] for tag in item["ner_tags"]]
        
        processed_data.append({
            "tokens": item["tokens"],
            "ner_tags": ner_ids
        })
    
    # Divide os dados em conjuntos de treinamento, validação e teste
    train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)
    
    # Cria objetos Dataset do Hugging Face
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Cria um DatasetDict com os três conjuntos
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    return dataset_dict

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokeniza os exemplos e alinha as etiquetas aos tokens produzidos pelo tokenizador
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding="max_length"  # Garantir padding consistente
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Corrigir alinhamento de subpalavras
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignorar subpalavras
                
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train_model(dataset_dict):
    """
    Treina o modelo usando o conjunto de dados processado
    """
    # Carrega o tokenizador e o modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokeniza e alinha as etiquetas
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True
    )
    
    # Prepara o data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Define os argumentos de treinamento
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",  # Nome corrigido
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none"
    )
    
    # Cria o treinador
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Treina o modelo
    trainer.train()
    
    # Salva o modelo e o tokenizador
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return model, tokenizer, trainer, tokenized_datasets

def evaluate_model(model, tokenizer, test_dataset):
    """
    Avalia o modelo no conjunto de teste
    """
    # Configura o trainer para avaliação
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    
    # Faz previsões no conjunto de teste
    predictions, labels, _ = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)
    
    # Converte IDs de volta para etiquetas
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Calcula as métricas
    report = classification_report(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    
    print("Relatório de Classificação:")
    print(report)
    print(f"F1-Score: {f1}")
    
    return true_labels, true_predictions

def extract_entities_from_text(text, model, tokenizer):
    """
    Extrai entidades de um texto usando o modelo treinado e formata o resultado
    no formato JSON específico solicitado
    """

    device = torch.device("cpu")
    model.to(device)
    # Tokeniza o texto
    tokens = text.split()
    inputs = tokenizer(tokens, truncation=True, is_split_into_words=True, return_tensors="pt", padding=True).to(device)
    
    # Faz a previsão
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtém a sequência mais provável
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_token_class_ids = predictions[0].tolist()
    
    # Mapeia as previsões para os tokens
    predicted_entities = []
    
    word_ids = inputs.word_ids()
    previous_word_idx = None
    entity = {"type": None, "text": ""}
    
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
    """
    Função principal que executa o fluxo completo
    """
    # Verifica se já existem dados processados
    dataset_dir = Path(__file__).parent / 'dataset_prepared'
    
    if dataset_dir.exists() and (dataset_dir / 'extracted_data.json').exists():
        print("Usando os dados já extraídos pelo script prepare_dataset.py...")
        
        # Carrega os dados anotados do arquivo JSON
        with open(dataset_dir / 'annotations.json', 'r', encoding='utf-8') as f:
            annotations_with_text = json.load(f)
        
        print(f"Foram carregados {len(annotations_with_text)} exemplos da anotação existente.")
        
        # Extrai os textos dos dados anotados
        texts = [annotation['text'] for annotation in annotations_with_text]
    else:
        print("Extraindo dados do arquivo dataset.dart...")
        texts = extract_dart_strings()
        print(f"Foram extraídos {len(texts)} textos.")
    
    print("\nAnotando dados...")
    annotated_data = annotate_data_manually(texts)
    print(f"Foram anotados {len(annotated_data)} exemplos.")
    
    print("\nPreparando conjunto de dados...")
    dataset_dict = prepare_dataset(annotated_data)
    print(f"Conjunto de treinamento: {len(dataset_dict['train'])} exemplos")
    print(f"Conjunto de validação: {len(dataset_dict['validation'])} exemplos")
    print(f"Conjunto de teste: {len(dataset_dict['test'])} exemplos")
    
    print("\nTreinando o modelo...")
    model, tokenizer, trainer, tokenized_datasets = train_model(dataset_dict)
    print(f"Modelo treinado e salvo em {OUTPUT_DIR}")
    
    print("\nAvaliando o modelo...")
    evaluate_model(model, tokenizer, tokenized_datasets["test"])
    
    print("\nExemplo de extração de entidades:")
    sample_text = texts[0]
    json_result = extract_entities_from_text(sample_text, model, tokenizer)
    
    print("Texto de exemplo:")
    print(sample_text[:300] + "...")
    print("\nDados extraídos (formato JSON):")
    print(json.dumps(json_result, indent=2, ensure_ascii=False))
    
    # Testa o modelo em alguns exemplos adicionais
    print("\nTestando o modelo em mais alguns exemplos:")
    for i in range(1, min(4, len(texts))):
        print(f"\nExemplo {i+1}:")
        text_excerpt = texts[i][:150] + "..."
        print(text_excerpt)
        
        json_result = extract_entities_from_text(texts[i], model, tokenizer)
        print(json.dumps(json_result, indent=2, ensure_ascii=False))
    
    # Salva um script de exemplo para uso do modelo em produção
    example_script_path = Path(__file__).parent / 'use_receipt_model.py'
    with open(example_script_path, 'w', encoding='utf-8') as f:
        f.write('''
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
''')
    
    print(f"\nScript de exemplo para usar o modelo foi criado em {example_script_path}")
    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()
