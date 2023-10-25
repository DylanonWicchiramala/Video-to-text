import gradio as gr
import main
from config import config

example_link = [
    "https://youtu.be/veCkQ_bPR3k?si=UsKqVpOArmcawwXx",
    "https://youtu.be/TksaY_FDgnk?si=OKtpJF_WzoKXhmWc",
    "https://youtu.be/aircAruvnKk?si=3Si3Are9yxeYNX1d",
]

main.pre_load()

with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    
    with gr.Row():
        with gr.Column():
            inp_file = gr.Video()
            inp_link = gr.Textbox(label="YouTube URL", placeholder="example: https://youtu.be/dQw4w9WgXcQ?si=MhbzIJK5swPr_txf")
            btn_complie = gr.Button("Complie")
            example_box = gr.Examples(
                    examples=example_link,
                    label="Example youtube link",
                    inputs=inp_link
                )
        
        with gr.Column():
            with gr.Row():
                btn_transcribe = gr.Button("Transcribe")
                btn_summarize = gr.Button("Summarize")
            
            out = gr.Text(label="Output", lines=15, max_lines=40)
    
        btn_complie.click(main.complie, inputs=[inp_file, inp_link, btn_transcribe], outputs=out)
        
        btn_transcribe.click(main.show_output, inputs=[btn_transcribe], outputs=out)
        btn_summarize.click(main.show_output, inputs=[btn_summarize], outputs=out)


demo.launch(share=config['gradio']['share'])