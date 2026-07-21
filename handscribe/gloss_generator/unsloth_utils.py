import os
import pandas as pd
from jiwer import wer, cer
from unsloth import FastModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template


PROJ_DIR = os.getcwd()
TUNED_MODELS_DIR = os.path.join(PROJ_DIR, 'ft_output')
DATASET_DIR = os.path.join(PROJ_DIR, 'GFSlowFastSign/dataset/PHOENIX-2014-T/annotations/manual')
GER_TRAIN_CSV = os.path.join(DATASET_DIR, 'PHOENIX-2014-T.train.corpus.csv')
GER_TEST_CSV = os.path.join(DATASET_DIR, 'PHOENIX-2014-T.test.corpus.csv')

DEFAULT_LANG = "German"
DEF_MAX_SEQ_LENGTH = 2048

# More models at https://huggingface.co/unsloth
AVAILABLE_MODELS = [
    "unsloth/Llama-3.1-8B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct", 
    "unsloth/Qwen2.5-14B-Instruct",
    "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
]

TUNED_MODELS = [tuned_model for tuned_model in os.listdir(TUNED_MODELS_DIR) if 'checkpoint' not in tuned_model]

SYS_PROMPT = """
    You are a sign language expert. Your task is to translate the sign language gloss for the given sentence.
    You will be given a sentence in {language} and you need to translate it into sign language gloss.
"""

USER_PROMPT = """
    Here is the sentence:
    {sentence}
    Please translate it into sign language gloss.
"""

OUTPUT_PROMPT = """
    The gloss is: {gloss}
"""


def format_sys_prompt(prompt=SYS_PROMPT, lang=DEFAULT_LANG):
    return prompt.format(language=lang)


def load_unsloth_model(model_id="unsloth/gemma-3-4B-it", seq_len=2048, load_in_4bit=True, full_FT=False):
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_id,
        max_seq_length = seq_len,
        load_in_4bit = load_in_4bit,
        load_in_8bit = not load_in_4bit,
        full_finetuning = full_FT,
        # token = "hf_...", # use one if using gated models
    )
    return model, tokenizer


def store_tuned_llm(model, tokenizer, out_dir=None):
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def extract_model_name(model_id):
    if 'llama' in model_id.lower():
        return "llama"
    elif 'gemma' in model_id.lower():
        return "gemma"
    elif 'qwen' in model_id.lower():
        return "qwen"
    else:
        raise ValueError(f"Unknown model name: {model_id}. Please check the model name.")


def load_data(file_path=GER_TRAIN_CSV, sys_prompt_lang='German', sep_char="|"):
    # Load data
    df = pd.read_csv(file_path, sep=sep_char, header=0)
    print(f'Read {len(df)} rows from {file_path}')
    
    # Rename Gloss and Translation cols
    if 'phoenix' in file_path.lower():
        df = df.rename(columns={"orth": "Output", "translation": "User"})
    else:
        df = df.rename(columns={"gloss": "Output", "text": "User"})

    # Extract only Gloss and Translation
    df = df[["Output", "User"]]

    df["User"] = df["User"].apply(
        lambda x: USER_PROMPT.format(sentence=x).strip()
    )

    df["Output"] = df["Output"].apply(
        lambda x: OUTPUT_PROMPT.format(gloss=x).strip()
    )
    
    sys_prompt = format_sys_prompt(SYS_PROMPT, sys_prompt_lang)
    df["System"] = sys_prompt
    
    return df[["System", "User", "Output"]]


def prepare_msg(model_tokenizer, msg="Translate the text.", llm_name="llama"):
    
    if llm_name == "gemma":
        msg = msg + " [IMPORTANT] Your response must follow the format \" The gloss is: \" "
        messages = [{
            "role": "user",
            "content": [{
                "type" : "text", "text" : f"{msg}",
            }]
        }]
        
    else:
        template_to_use = "llama-3.1" if llm_name == "llama" else "qwen-2.5"
        
        model_tokenizer = get_chat_template(
            model_tokenizer, chat_template = template_to_use,
        )

        messages = [
            {"role": "user", "content": f"{msg}"},
        ]
    
    return model_tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")


def get_text_streamer(model_tokenizer):
    return TextStreamer(model_tokenizer, skip_prompt=True)


def gen_response_textstream(input_ids, tuned_model, max_new_tokens=128, gen_kwargs=None,
                            text_streamer=None, pad_token_id=None, 
                            llm_tokenizer=None, use_cache=True, use_streamer=False):
    
    if use_streamer:
        if not text_streamer and llm_tokenizer:
            text_streamer = text_streamer = TextStreamer(llm_tokenizer, skip_prompt=True)
        gen_kwargs["streamer"] = text_streamer
    
    return tuned_model.generate(
        input_ids,
        **gen_kwargs,
        max_new_tokens = max_new_tokens,
        use_cache=use_cache,
        do_sample = True,
        pad_token_id = pad_token_id,
    )
    

def compute_wer(og_sent, gen_sent):
    return wer(og_sent, gen_sent)


def compute_cer(og_sent, gen_sent):
    return cer(og_sent, gen_sent)