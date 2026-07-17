import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
# os.environ["UNSLOTH_COMPILE_DISABLE"]  = "1"
import argparse
from tqdm import tqdm
from torch import inference_mode
from unsloth import FastLanguageModel
from unsloth_utils import (
    TUNED_MODELS,
    GER_TEST_CSV,
    TUNED_MODELS_DIR,
    DEF_MAX_SEQ_LENGTH,
    load_data,
    prepare_msg,
    compute_wer,
    compute_cer,
    get_text_streamer,
    extract_model_name,
    load_unsloth_model,
    gen_response_textstream,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a sign language recognition dataset.")
    parser.add_argument(
        "--max_seq_length", type=int, default=DEF_MAX_SEQ_LENGTH, help="Maximum sequence length for the model."
        )
    parser.add_argument(
        "--llm_to_use", type=str, help="The model to fine-tune.", choices=TUNED_MODELS
        )
    parser.add_argument(
        "--data_csv_path", type=str, default=GER_TEST_CSV, help="Path to the training CSV file."
        )
    parser.add_argument(
        "--data_language", type=str, default="German", help="Language of the training data."
        )
    parser.add_argument(
        "--sentences_to_test", type=int, default=0, help="Number of sentences to test."
    )
    parser.add_argument(
        "--use_streamer", action='store_true', help="Number of sentences to test."
    )
    
    args, _ = parser.parse_known_args()
    args.data_language = "German" if args.data_csv_path == GER_TEST_CSV else "English"
    return vars(args)


def get_gen_kwargs(llm_name="llama"):
    if llm_name == "gemma":
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    
    elif llm_name == "llama":
        return {"temperature": 0.7, "top_k": 50, "top_p": 0.95}
    
    elif llm_name == "qwen":
        return {"temperature": 1.5, "min_p": 0.1}
    
    else:
        return {"temperature": 0.5, "top_k": 50, "top_p": 0.95}


def run_inference():
    
    args = parse_args()

    print(f'Using the tuned model called: {args['llm_to_use']}')
    llm_name = extract_model_name(args['llm_to_use'])
    gen_kwargs = get_gen_kwargs(llm_name)
    
    # Load data and retrieve test sentences
    if 'phoenix' in args['data_csv_path'].lower():
        df = load_data(args['data_csv_path'], args['data_language'])
    else:
        df = load_data(args['data_csv_path'], args['data_language'], "___")
    
    # Retrieve GT glossess and remove "The gloss is:" prefix
    if args['sentences_to_test'] != 0:
        sents = df.tail(args['sentences_to_test']).iloc[:, 1].tolist()
        ground_truths = df.tail(args['sentences_to_test']).iloc[:, 2].tolist()
    else:
        sents = df.iloc[:, 1].tolist()
        ground_truths = df.iloc[:, 2].tolist()
    
    ground_truths = [gt.replace("The gloss is: ", "") for gt in ground_truths]
    print(f'Will try to translate {len(sents)} sentences!')
    
    # Load model, tokenizer and prepare text streamer
    model_path = os.path.join(TUNED_MODELS_DIR, args['llm_to_use'])
    print(f'Loading model from {model_path}')
    model, tokenizer = load_unsloth_model(model_id=model_path)
    model = FastLanguageModel.for_inference(model)
    
    if args['use_streamer']:
        text_streamer = get_text_streamer(tokenizer)
    else:
        text_streamer = None
    
    gen_sents = []
    
    with inference_mode():
        for sen in tqdm(sents, desc="Glossing sentences", total=len(sents)):
            inputs = prepare_msg(tokenizer, sen, llm_name)
            
            gen_sent = gen_response_textstream(
                inputs, model, 
                max_new_tokens=128,
                gen_kwargs=gen_kwargs,
                text_streamer=text_streamer, 
                pad_token_id=tokenizer.eos_token_id, llm_tokenizer=tokenizer, 
                use_cache=False,
                use_streamer=args['use_streamer'],
            )
        
            gen_sent = tokenizer.decode(gen_sent[0], skip_special_tokens=True)
            gen_sents.append(gen_sent.split('The gloss is:')[1].strip())
    
    avg_wer = compute_wer(ground_truths, gen_sents)
    avg_cer = compute_cer(ground_truths, gen_sents)
    
    print(f'=================\nAverage WER: {avg_wer}'
          f'\nAverage CER: {avg_cer}\n=================')  


if __name__ == "__main__":
    run_inference()