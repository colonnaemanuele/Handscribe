import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    MBartForConditionalGeneration,
    MBartTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoTokenizer,
    BitsAndBytesConfig
)

MBART_MODEL_ID = "large-mtm"


def get_mbart_model_dict():
    return {
        "large": "facebook/mbart-large-50",
        "en-ro": "facebook/mbart-large-en-ro",
        "large-cc": "facebook/mbart-large-cc25",
        "large-otm": "facebook/mbart-large-50-one-to-many-mmt",
        "large-mto": "facebook/mbart-large-50-many-to-one-mmt",
        "large-mtm": "facebook/mbart-large-50-many-to-many-mmt",
    }

def get_mbart_langs_dict():
    return {
        "Arabic": "ar_AR",
        "Czech": "cs_CZ",
        "German": "de_DE",
        "English": "en_XX",
        "Spanish": "es_XX",
        "Estonian": "et_EE",
        "Finnish": "fi_FI",
        "French": "fr_XX",
        "Gujarati": "gu_IN",
        "Hindi": "hi_IN",
        "Italian": "it_IT",
        "Japanese": "ja_XX",
        "Kazakh": "kk_KZ",
        "Korean": "ko_KR",
        "Lithuanian": "lt_LT",
        "Latvian": "lv_LV",
        "Burmese": "my_MM",
        "Nepali": "ne_NP",
        "Dutch": "nl_XX",
        "Romanian": "ro_RO",
        "Russian": "ru_RU",
        "Sinhala": "si_LK",
        "Turkish": "tr_TR",
        "Vietnamese": "vi_VN",
        "Chinese": "zh_CN",
        "Afrikaans": "af_ZA",
        "Azerbaijani": "az_AZ",
        "Bengali": "bn_IN",
        "Persian": "fa_IR",
        "Hebrew": "he_IL",
        "Croatian": "hr_HR",
        "Indonesian": "id_ID",
        "Georgian": "ka_GE",
        "Khmer": "km_KH",
        "Macedonian": "mk_MK",
        "Malayalam": "ml_IN",
        "Mongolian": "mn_MN",
        "Marathi": "mr_IN",
        "Polish": "pl_PL",
        "Pashto": "ps_AF",
        "Portuguese": "pt_XX",
        "Swedish": "sv_SE",
        "Swahili": "sw_KE",
        "Tamil": "ta_IN",
        "Telugu": "te_IN",
        "Thai": "th_TH",
        "Tagalog": "tl_XX",
        "Ukrainian": "uk_UA",
        "Urdu": "ur_PK",
        "Xhosa": "xh_ZA",
        "Galician": "gl_ES",
        "Slovene": "sl_SI"
    }


def get_mbart_details(mbart_id, source_lang=None, target_lang=None):
    mbart_lang_dict = get_mbart_model_dict()
    model_name = mbart_lang_dict[mbart_id] if mbart_id in mbart_lang_dict else "facebook/mbart-large-mtm"
    languages = get_mbart_languages_ids([source_lang, target_lang])
    return model_name, languages


def get_mbart_languages_ids(languages_to_use):
    langs_dict = get_mbart_langs_dict()
    return [langs_dict[lang] for lang in languages_to_use if lang in langs_dict]


def load_mbart(mbart_id="large-mtm", language_model=None):
    source_lang = target_lang = language_model if language_model else "German"
    mbart_model_name, mbart_languages = get_mbart_details(mbart_id, source_lang, target_lang)
    mbart_tokenizer = MBartTokenizer.from_pretrained(mbart_model_name, src_lang=mbart_languages[0], tgt_lang=mbart_languages[1])
    mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)
    return mbart_model, mbart_tokenizer, mbart_languages


def get_quantization_config(use_4bit: bool = False):

    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16 
        )

    else:
        return BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=True
        )


def load_llm(llm_model_id=None, torch_dtype="auto", load_pipe=False, quantize=False, quantize_4bit=False):
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    
    if quantize:
        model_kwargs['quantization_config'] = get_quantization_config(use_4bit=quantize_4bit)
    
    if not load_pipe:
    
        llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            **model_kwargs,
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
        return llm, llm_tokenizer
    
    else:
        
        pipe = pipeline(
            "text-generation",
            model=llm_model_id,
            torch_dtype=torch.bfloat16, # or torch_dtype="auto"
            device_map="auto",
        )
        
        return pipe
