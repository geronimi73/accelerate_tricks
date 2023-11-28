from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json

accelerator = Accelerator()

def write_pretty_json(file_path, data):
    import json
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)

# 10*10 Prompts. Source: https://www.penguin.co.uk/articles/2022/04/best-first-lines-in-books
prompts_all=[
    "The King is dead. Long live the Queen.",
    "Once there were four children whose names were Peter, Susan, Edmund, and Lucy.",
    "The story so far: in the beginning, the universe was created.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The sweat wis lashing oafay Sick Boy; he wis trembling.",
    "124 was spiteful. Full of Baby's venom.",
    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",
    "I write this sitting in the kitchen sink.",
    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",
] * 10

# load a base model and tokenizer
model_path="models/llama2-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)   
tokenizer.pad_token = tokenizer.eos_token

# batch, left pad (for inference), and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

# sync GPUs and start the timer
accelerator.wait_for_everyone()    
start=time.time()

# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference in batches
    prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=16)

    for prompts_tokenized in prompt_batches:
        outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=100)

        # remove prompt from gen. tokens
        outputs_tokenized=[ tok_out[len(tok_in):] 
            for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

        # count and decode gen. tokens 
        num_tokens=sum([ len(t) for t in outputs_tokenized ])
        outputs=tokenizer.batch_decode(outputs_tokenized)

        # store in results{} to be gathered by accelerate
        results["outputs"].extend(outputs)
        results["num_tokens"] += num_tokens

    timediff=time.time()-start
    print("GPU {}: {} prompts received, generated {} tokens in {} seconds, {} t/s".format(
        accelerator.process_index,
        len(prompts),
        results["num_tokens"],
        timediff,
        results["num_tokens"]//timediff,
        ))

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

results_gathered=gather_object(results)

if accelerator.is_main_process:
    timediff=time.time()-start
    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")

write_pretty_json("inference_batched.json",results_gathered)

### BS 8, 100 prompts, 100 tokens per prompt, 290 Watts per GPU

# GPU 0: 100 prompts received, generated 10000 tokens in 31.271002769470215 seconds, 319.0 t/s
# tokens/sec: 319.0, time elapsed: 31.271076679229736, num_tokens 10000

# GPU 1: 50 prompts received, generated 5000 tokens in 33.125919342041016 seconds, 150.0 t/s
# GPU 0: 50 prompts received, generated 5000 tokens in 36.581854581832886 seconds, 136.0 t/s
# tokens/sec: 273.0, time elapsed: 36.582783222198486, num_tokens 10000

# GPU 2: 32 prompts received, generated 3200 tokens in 18.911545515060425 seconds, 169.0 t/s
# GPU 1: 34 prompts received, generated 3400 tokens in 27.072452545166016 seconds, 125.0 t/s
# GPU 0: 34 prompts received, generated 3400 tokens in 28.82193922996521 seconds, 117.0 t/s
# tokens/sec: 346.0, time elapsed: 28.823057413101196, num_tokens 10000

# GPU 3: 25 prompts received, generated 2500 tokens in 18.729575157165527 seconds, 133.0 t/s
# GPU 1: 25 prompts received, generated 2500 tokens in 18.787354946136475 seconds, 133.0 t/s
# GPU 2: 25 prompts received, generated 2500 tokens in 19.151126623153687 seconds, 130.0 t/s
# GPU 0: 25 prompts received, generated 2500 tokens in 23.442680597305298 seconds, 106.0 t/s
# tokens/sec: 426.0, time elapsed: 23.444172859191895, num_tokens 10000

# GPU 1: 20 prompts received, generated 2000 tokens in 14.69238829612732 seconds, 136.0 t/s
# GPU 4: 20 prompts received, generated 2000 tokens in 14.714111089706421 seconds, 135.0 t/s
# GPU 2: 20 prompts received, generated 2000 tokens in 14.749407529830933 seconds, 135.0 t/s
# GPU 3: 20 prompts received, generated 2000 tokens in 14.98465371131897 seconds, 133.0 t/s
# GPU 0: 20 prompts received, generated 2000 tokens in 16.738210439682007 seconds, 119.0 t/s
# tokens/sec: 597.0, time elapsed: 16.74017906188965, num_tokens 10000

### BS 16, 100 prompts, 100 tokens per prompt, 290 Watts per GPU

# GPU 0: 100 prompts received, generated 10000 tokens in 19.313682556152344 seconds, 517.0 t/s
# tokens/sec: 517.0, time elapsed: 19.31375765800476, num_tokens 10000

# GPU 1: 50 prompts received, generated 5000 tokens in 19.065516233444214 seconds, 262.0 t/s
# GPU 0: 50 prompts received, generated 5000 tokens in 24.194597244262695 seconds, 206.0 t/s
# tokens/sec: 413.0, time elapsed: 24.195536136627197, num_tokens 10000

# GPU 2: 32 prompts received, generated 3200 tokens in 10.65116286277771 seconds, 300.0 t/s
# GPU 1: 34 prompts received, generated 3400 tokens in 19.152705907821655 seconds, 177.0 t/s
# GPU 0: 34 prompts received, generated 3400 tokens in 20.078866243362427 seconds, 169.0 t/s
# tokens/sec: 498.0, time elapsed: 20.07996392250061, num_tokens 10000

# GPU 1: 25 prompts received, generated 2500 tokens in 11.055442094802856 seconds, 226.0 t/s
# GPU 3: 25 prompts received, generated 2500 tokens in 11.070488214492798 seconds, 225.0 t/s
# GPU 2: 25 prompts received, generated 2500 tokens in 11.285370349884033 seconds, 221.0 t/s
# GPU 0: 25 prompts received, generated 2500 tokens in 14.004549026489258 seconds, 178.0 t/s
# tokens/sec: 713.0, time elapsed: 14.00615406036377, num_tokens 10000

# GPU 3: 20 prompts received, generated 2000 tokens in 10.628915071487427 seconds, 188.0 t/s
# GPU 1: 20 prompts received, generated 2000 tokens in 10.795271158218384 seconds, 185.0 t/s
# GPU 4: 20 prompts received, generated 2000 tokens in 11.019695520401001 seconds, 181.0 t/s
# GPU 2: 20 prompts received, generated 2000 tokens in 11.066817045211792 seconds, 180.0 t/s
# GPU 0: 20 prompts received, generated 2000 tokens in 13.865175485610962 seconds, 144.0 t/s
# tokens/sec: 721.0, time elapsed: 13.867400646209717, num_tokens 10000

##### FIXED DEVICE_MAP


# GPU 0: 100 prompts received, generated 10000 tokens in 19.21204710006714 seconds, 520.0 t/s
# tokens/sec: 520.0, time elapsed: 19.21213436126709, num_tokens 10000

# GPU 1: 50 prompts received, generated 5000 tokens in 11.014115571975708 seconds, 453.0 t/s
# GPU 0: 50 prompts received, generated 5000 tokens in 11.108545303344727 seconds, 450.0 t/s
# tokens/sec: 900.0, time elapsed: 11.109480381011963, num_tokens 10000

# GPU 2: 32 prompts received, generated 3200 tokens in 6.120448350906372 seconds, 522.0 t/s
# GPU 1: 34 prompts received, generated 3400 tokens in 8.22350263595581 seconds, 413.0 t/s
# GPU 0: 34 prompts received, generated 3400 tokens in 8.295660495758057 seconds, 409.0 t/s
# tokens/sec: 1205.0, time elapsed: 8.296635866165161, num_tokens 10000

# GPU 0: 25 prompts received, generated 2500 tokens in 5.953023910522461 seconds, 419.0 t/s
# GPU 2: 25 prompts received, generated 2500 tokens in 6.007809162139893 seconds, 416.0 t/s
# GPU 3: 25 prompts received, generated 2500 tokens in 6.008904457092285 seconds, 416.0 t/s
# GPU 1: 25 prompts received, generated 2500 tokens in 6.0392467975616455 seconds, 413.0 t/s
# tokens/sec: 1655.0, time elapsed: 6.040315628051758, num_tokens 10000

# GPU 1: 20 prompts received, generated 2000 tokens in 5.829110622406006 seconds, 343.0 t/s
# GPU 4: 20 prompts received, generated 2000 tokens in 5.915517091751099 seconds, 338.0 t/s
# GPU 0: 20 prompts received, generated 2000 tokens in 5.936866998672485 seconds, 336.0 t/s
# GPU 3: 20 prompts received, generated 2000 tokens in 5.976394176483154 seconds, 334.0 t/s
# GPU 2: 20 prompts received, generated 2000 tokens in 6.028886318206787 seconds, 331.0 t/s
# tokens/sec: 1658.0, time elapsed: 6.030542612075806, num_tokens 10000



