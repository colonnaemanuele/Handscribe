import os
# Set this here, while in notebooks just before the trainer.train(): https://github.com/unslothai/unsloth/issues/1530
# Environmental flags: https://docs.unsloth.ai/basics/errors-troubleshooting/unsloth-environment-flags
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"]  = "1"  

import unsloth
import argparse
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from transformers import DataCollatorForSeq2Seq # type: ignore
from unsloth import FastModel, FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt, standardize_data_formats, train_on_responses_only
from unsloth_utils import (
    GER_TRAIN_CSV,
    AVAILABLE_MODELS,
    DEF_MAX_SEQ_LENGTH,
    extract_model_name,    
    load_unsloth_model,
    store_tuned_llm,
    load_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a sign language recognition dataset.")
    parser.add_argument(
        "--max_seq_length", type=int, default=DEF_MAX_SEQ_LENGTH, help="Maximum sequence length for the model."
        )
    parser.add_argument(
        "--llm_to_tune", type=str, default=AVAILABLE_MODELS[1], help="The model to fine-tune.", choices=AVAILABLE_MODELS
        )
    parser.add_argument(
        "--data_csv_path", type=str, default=GER_TRAIN_CSV, help="Path to the training CSV file."
        )
    parser.add_argument(
        "--data_language", type=str, default="German", help="Language of the training data."
        )
    parser.add_argument(
        "--use_data_subset", action="store_true", help="Use a subset of the data for training."
        )
    parser.add_argument(
        "--subset_amount", type=int, default=1000, help="Number of samples to use from the dataset."
        )
    parser.add_argument(
        "--out_dir", type=str, default="./ft_output", help="Output directory for the fine-tuned model."
        )
    parser.add_argument(
        "--load_4bit", action="store_true", help="Load the model in 4-bit precision."
        )
    
    args, _ = parser.parse_known_args()
    args.data_language = "German" if args.data_csv_path == GER_TRAIN_CSV else "English"
    return vars(args)

  
def setup_fastllm(model, llora_rank=16, max_seq_length=2048):
    # Settings only for Gemma
    if model.config.model_type == "gemma":
        return FastModel.get_peft_model(
                model,
                finetune_vision_layers     = False, # Turn off for just text!
                finetune_language_layers   = True,
                finetune_attention_modules = True,
                finetune_mlp_modules       = True,

                r = 8,
                lora_alpha = 8,
                lora_dropout = 0,
                bias = "none",
                random_state = 3407,
            )
    # Settings for LLaMa and Qwen
    else:
        return FastLanguageModel.get_peft_model(
            model,
            r = llora_rank, # Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = llora_rank, # Suggested to be equal to the rank r, or double it.
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context # type: ignore
            random_state = 3407,
            max_seq_length = max_seq_length,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

    
def preprocess_data(dataframe, llm_tokenizer, use_susbet=False, subset_amount=1000, llm_name="llama"):

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        # For all Qwen, LLaMa and Gemma
        texts = [llm_tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        
        if llm_name == "gemma":
            texts = [t.removeprefix('<bos>') for t in texts]
        
        return { "text" : texts, }

    converted = []
    for _, row in dataframe.iterrows():
        messages = [
            {"from": "system", "value": row["System"].strip()},
            {"from": "human", "value": row["User"].strip()},
            {"from": "gpt", "value": row["Output"].strip()},
        ]
        converted.append({"conversations": messages})

    print(f'Converted {len(converted)} rows to chat format.')

    data = Dataset.from_list(converted)
    
    data = standardize_sharegpt(data) if llm_name != "gemma" else standardize_data_formats(data)
    
    data = data.map(formatting_prompts_func, batched=True)

    if use_susbet:
        return data.shuffle(seed=42).select(range(subset_amount))
    else:
        return data.shuffle(seed=42)


def get_stft_config(out_dir="./ft_output", seq_len=DEF_MAX_SEQ_LENGTH):
    os.makedirs(out_dir, exist_ok=True)
    
    return SFTConfig(
        dataset_text_field = "text",
        max_seq_length = seq_len,
        dataset_num_proc = 4,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 60, # Comment this to set num_train_epochs (longer training)
        num_train_epochs = 3, # 3 Suggested as best number of epochs https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-11.-inference-running-the-model
        learning_rate=2e-4, # Suggested values: 2e-4, 1e-4, 5e-5, 2e-5
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim="adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = out_dir,
    )


def setup_trainer(model, tokenizer, train_data, ft_config, llm_name="llama"):
    
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_data,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        args = ft_config
    )


    if llm_name == "llama":
        return train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        
    elif llm_name == "qwen":
        return train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )
        
    elif llm_name == "gemma":
        return train_on_responses_only(
            trainer,
            instruction_part = "<start_of_turn>user\n",
            response_part = "<start_of_turn>model\n",
        )
                
    return trainer


def tune_llm():
    
    args = parse_args()
    
    # Load data and retrieve test sentences
    if 'phoenix' in args['data_csv_path'].lower():
        df = load_data(args['data_csv_path'], "German")
    else:
        df = load_data(args['data_csv_path'], "Multilingual", '___')

    chosen_llm = args['llm_to_tune']
    print(f'Using model: {chosen_llm}')

    ft_config = get_stft_config()
    
    model, tokenizer = load_unsloth_model(
        chosen_llm, seq_len=2048, load_in_4bit=False, full_FT=False
    )

    model = setup_fastllm(model, llora_rank=16, max_seq_length=args['max_seq_length'])

    train_data = preprocess_data(df, tokenizer, 
                                 args['use_data_subset'], args['subset_amount'],
                                 llm_name=extract_model_name(args['llm_to_tune']))

    trainer = setup_trainer(model, tokenizer, train_data, ft_config, llm_name=extract_model_name(args['llm_to_tune']))

    trainer.train() # type: ignore

    # Save the model and tokenizer
    store_tuned_llm(model, tokenizer, out_dir=args['out_dir'])


if __name__ == "__main__":
    tune_llm()