import gradio as gr
import pandas as pd
from Prediction import *
import os
from datetime import datetime


examples = []
if os.path.exists("assets/examples.txt"):
    with open("assets/examples.txt", "r", encoding="utf8") as file:
        for sentence in file:
            sentence = sentence.strip()
            examples.append(sentence)
else:
    examples = [
        "Games of the imagination teach us actions have consequences in a realm that can be reset.",
        "But New Jersey farmers are retiring and all over the state, development continues to push out dwindling farmland.",
        "He also is the Head Designer of The Design Trust so-to-speak, besides his regular job ..."
        ]

device = torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained("Oliver12315/Brand_Tone_of_Voice")
model = BertForSequenceClassification.from_pretrained("Oliver12315/Brand_Tone_of_Voice")
model = model.to(device)


def single_sentence(sentence):
    predictions = predict_single(sentence, tokenizer, model, device)
    return sorted(zip(LABEL_COLUMNS, predictions), key=lambda x:x[-1], reverse=True)

def csv_process(csv_file, attr="content"):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    data = pd.read_csv(csv_file.name)
    data = data.reset_index()
    os.makedirs('output', exist_ok=True)
    outputs = []
    predictions = predict_csv(data, attr, tokenizer, model, device)
    output_path = f"output/prediction_Brand_Tone_of_Voice_{formatted_time}.csv"
    predictions.to_csv(output_path)
    outputs.append(output_path)
    return outputs


my_theme = gr.Theme.from_hub("JohnSmith9982/small_and_pretty")
with gr.Blocks(theme=my_theme, title='Brand_Tone_of_Voice_demo') as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href="https://github.com/xxx" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        </a>
        <div>
            <h1 >Place the title of the paper here</h1>
            <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h5>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;>
                <a href="https://arxiv.org/abs/xx.xx"><img src="https://img.shields.io/badge/Arxiv-xx.xx-red"></a>
                <a href='https://huggingface.co/spaces/Oliver12315/Brand_Tone_of_Voice_demo'><img src='https://img.shields.io/badge/Project_Page-Oliver12315/Brand_Tone_of_Voice_demo' alt='Project Page'></a>
                <a href='https://github.com'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
            </div>
        </div>
        </div>
        """)
    
    with gr.Tab("Readme"):
        gr.Markdown("""
            # Detailed information about our model:
                    
            The example model here is a tone classification model suitable for financial field texts.

            # Paper Name
                    
            # Authors

            + First author
            + Corresponding author

            # How to use?
                    
            Please refer to the other two tab card for predictions.
                    
            + The `Single Sentence` for the tone classification of individual sentence.
            + The `CSV File` for inputting CSV file for batch prediction and return.
            ...
            """)

    with gr.Tab("Single Sentence"):
        tbox_input = gr.Textbox(label="Input",
                                    info="Please input a sentence here:")

        tab_output = gr.DataFrame(label='Predictions:', 
                                  headers=["Label", "Probability"],
                                  datatype=["str", "number"],
                                  interactive=False)
        with gr.Row():
            button_ss = gr.Button("Submit", variant="primary")
            button_ss.click(fn=single_sentence, inputs=[tbox_input], outputs=[tab_output])
            gr.ClearButton([tbox_input, tab_output])

        gr.Examples(
            examples=examples,
            inputs=tbox_input,
            examples_per_page=len(examples)
        )

    with gr.Tab("Csv File"):
        with gr.Row():
            csv_input = gr.File(label="CSV File:",
                                file_types=['.csv'],
                                file_count="single"
                                )
            csv_output = gr.File(label="Predictions:")

        with gr.Row():
            button = gr.Button("Submit", variant="primary")
            button.click(fn=csv_process, inputs=[csv_input], outputs=[csv_output])
            gr.ClearButton([csv_input, csv_output])

        gr.Markdown("## Examples \n The incoming CSV must include the ``content`` field, which represents the text that needs to be predicted!")
        gr.DataFrame(label='Csv input format:',
                    value=[[i, examples[i]] for i in range(len(examples))],
                    headers=["index", "content"],
                    datatype=["number","str"],
                    interactive=False
                    )
demo.launch()