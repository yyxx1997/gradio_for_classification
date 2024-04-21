import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import BertTokenizerFast as BertTokenizer, BertForSequenceClassification
import os
import glob


RANDOM_SEED = 42
pd.RANDOM_SEED = 42
LABEL_COLUMNS = ["Assertive Tone", "Conversational Tone", "Emotional Tone", "Informative Tone", "None"]
  

@torch.no_grad()
def predict_csv(data, text_col, tokenizer, model, device, text_bs=16, max_token_len=128):
    predictions = []
    post = data[text_col]
    num_text = len(post)
    generator = range(0, num_text, text_bs)
    for i in tqdm(generator, total=len(generator), desc="Processing..."):
      texts = post[i: min(num_text, i+text_bs)].tolist()
      encoding = tokenizer(
          texts,
          add_special_tokens=True,
          max_length=max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
      logits = model(
          encoding["input_ids"].to(device),
          encoding["attention_mask"].to(device),
          return_dict=True
      ).logits
      prediction = torch.softmax(logits, dim=1)
      predictions.append(prediction.detach().cpu())

    final_pred = torch.cat(predictions, dim=0)
    y_inten = final_pred.numpy().T

    for i in range(len(LABEL_COLUMNS)):
      data[LABEL_COLUMNS[i]] = [round(i, 8) for i in y_inten[i].tolist()]
    return data

@torch.no_grad()
def predict_single(sentence, tokenizer, model, device, max_token_len=128):
    encoding = tokenizer(
        sentence,
        add_special_tokens=True,
        max_length=max_token_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
      )
    logits = model(
        encoding["input_ids"].to(device),
        encoding["attention_mask"].to(device),
        return_dict=True
    ).logits
    prediction = torch.softmax(logits, dim=1)
    y_inten = prediction.flatten().cpu().numpy().T.tolist()
    y_inten = [round(i, 8) for i in y_inten]
    return y_inten



if __name__ == "__main__":

  Data = pd.read_csv("assets/Kickstarter_sentence_level_5000.csv")
  Data = Data[:20]
  device = torch.device('cpu')

  # Load model directly
  tokenizer = BertTokenizer.from_pretrained("Oliver12315/Brand_Tone_of_Voice")
  model = BertForSequenceClassification.from_pretrained("Oliver12315/Brand_Tone_of_Voice")
  model = model.to(device)
  fk_doc_result = predict_csv(Data,"content", tokenizer, model, device)
  single_response = predict_single("Games of the imagination teach us actions have consequences in a realm that can be reset.", tokenizer, model, device)
  fk_doc_result.to_csv(f"output/prediction_Brand_Tone_of_Voice.csv")