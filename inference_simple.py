from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json

accelerator = Accelerator()

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

# sync GPUs and start the timer
accelerator.wait_for_everyone()
start=time.time()

# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:
    # store output of generations in dict
    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference, prompt by prompt
    for prompt in prompts:
        prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

        # remove prompt from output 
        output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        # store outputs and number of tokens in result{}
        results["outputs"].append( tokenizer.decode(output_tokenized) )
        results["num_tokens"] += len(output_tokenized)

    timediff=time.time()-start
    print("GPU {}: {} prompts received, generated {} tokens in {} seconds, {} t/s".format(
        accelerator.process_index,
        len(prompts),
        results["num_tokens"],
        timediff,
        results["num_tokens"]//timediff,
        ))

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs
results_gathered=gather_object(results)

if accelerator.is_main_process:
    timediff=time.time()-start
    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")


## RESULTS
# GPU 0: 65 prompts received, generated 3250 tokens in 73.38190412521362 seconds, 44.0 t/s
# tokens/sec: 44.0, time 73.38197040557861, total tokens 3250, total prompts 65

# GPU 1: 32 prompts received, generated 1600 tokens in 69.78886079788208 seconds, 22.0 t/s
# GPU 0: 33 prompts received, generated 1650 tokens in 74.3731198310852 seconds, 22.0 t/s
# tokens/sec: 43.0, time 74.37401580810547, total tokens 3250, total prompts 65

# GPU 2: 21 prompts received, generated 1050 tokens in 43.56167006492615 seconds, 24.0 t/s
# GPU 0: 22 prompts received, generated 1100 tokens in 47.51334881782532 seconds, 23.0 t/s
# GPU 1: 22 prompts received, generated 1100 tokens in 47.575270652770996 seconds, 23.0 t/s
# tokens/sec: 68.0, time 47.57618808746338, total tokens 3250, total prompts 65

# GPU 3: 14 prompts received, generated 700 tokens in 28.84010410308838 seconds, 24.0 t/s
# GPU 0: 17 prompts received, generated 850 tokens in 38.90381646156311 seconds, 21.0 t/s
# GPU 1: 17 prompts received, generated 850 tokens in 39.05107378959656 seconds, 21.0 t/s
# GPU 2: 17 prompts received, generated 850 tokens in 39.11865496635437 seconds, 21.0 t/s
# tokens/sec: 83.0, time 39.120097637176514, total tokens 3250, total prompts 65

# GPU 0: 13 prompts received, generated 650 tokens in 30.16798162460327 seconds, 21.0 t/s
# GPU 3: 13 prompts received, generated 650 tokens in 30.245014429092407 seconds, 21.0 t/s
# GPU 4: 13 prompts received, generated 650 tokens in 30.246111392974854 seconds, 21.0 t/s
# GPU 1: 13 prompts received, generated 650 tokens in 30.761144399642944 seconds, 21.0 t/s
# GPU 2: 13 prompts received, generated 650 tokens in 30.777416944503784 seconds, 21.0 t/s
# tokens/sec: 105.0, time 30.77891182899475, total tokens 3250, total prompts 65

### 100 prompts, 100 tokens per prompt, 290 Watts per GPU

# GPU 0: 100 prompts received, generated 10000 tokens in 225.55214309692383 seconds, 44.0 t/s
# tokens/sec: 44.0, time 225.55221390724182, total tokens 10000, total prompts 100

# GPU 0: 50 prompts received, generated 5000 tokens in 215.75409865379333 seconds, 23.0 t/s
# GPU 1: 50 prompts received, generated 5000 tokens in 215.89760065078735 seconds, 23.0 t/s
# tokens/sec: 46.0, time 215.89851832389832, total tokens 10000, total prompts 100

# GPU 2: 32 prompts received, generated 3200 tokens in 121.804114818573 seconds, 26.0 t/s
# GPU 1: 34 prompts received, generated 3400 tokens in 136.55819630622864 seconds, 24.0 t/s
# GPU 0: 34 prompts received, generated 3400 tokens in 136.8167154788971 seconds, 24.0 t/s
# tokens/sec: 73.0, time 136.8177628517151, total tokens 10000, total prompts 100

# GPU 0: 25 prompts received, generated 2500 tokens in 95.79093980789185 seconds, 26.0 t/s
# GPU 3: 25 prompts received, generated 2500 tokens in 95.95476984977722 seconds, 26.0 t/s
# GPU 1: 25 prompts received, generated 2500 tokens in 96.01553773880005 seconds, 26.0 t/s
# GPU 2: 25 prompts received, generated 2500 tokens in 96.20409512519836 seconds, 25.0 t/s
# tokens/sec: 103.0, time 96.20520401000977, total tokens 10000, total prompts 100

# GPU 0: 20 prompts received, generated 2000 tokens in 84.0069522857666 seconds, 23.0 t/s
# GPU 4: 20 prompts received, generated 2000 tokens in 84.19389176368713 seconds, 23.0 t/s
# GPU 3: 20 prompts received, generated 2000 tokens in 84.29939103126526 seconds, 23.0 t/s
# GPU 1: 20 prompts received, generated 2000 tokens in 84.36542701721191 seconds, 23.0 t/s
# GPU 2: 20 prompts received, generated 2000 tokens in 84.58180141448975 seconds, 23.0 t/s
# tokens/sec: 118.0, time 84.58379983901978, total tokens 10000, total prompts 100

## DEVICE MAP FIX!!!!

# tokens/sec: 44.0, time 225.03824400901794, total tokens 10000, total prompts 100

# GPU 1: 50 prompts received, generated 5000 tokens in 112.392493724823 seconds, 44.0 t/s
# GPU 0: 50 prompts received, generated 5000 tokens in 112.88041234016418 seconds, 44.0 t/s
# tokens/sec: 88.0, time 112.88138389587402, total tokens 10000, total prompts 100

# GPU 2: 32 prompts received, generated 3200 tokens in 72.51620578765869 seconds, 44.0 t/s
# GPU 1: 34 prompts received, generated 3400 tokens in 76.89501714706421 seconds, 44.0 t/s
# GPU 0: 34 prompts received, generated 3400 tokens in 77.64040040969849 seconds, 43.0 t/s
# tokens/sec: 128.0, time 77.64143228530884, total tokens 10000, total prompts 100

# GPU 3: 25 prompts received, generated 2500 tokens in 71.47983646392822 seconds, 34.0 t/s
# GPU 2: 25 prompts received, generated 2500 tokens in 72.08399438858032 seconds, 34.0 t/s
# GPU 1: 25 prompts received, generated 2500 tokens in 72.38916826248169 seconds, 34.0 t/s
# GPU 0: 25 prompts received, generated 2500 tokens in 72.74923491477966 seconds, 34.0 t/s
# tokens/sec: 137.0, time 72.75045394897461, total tokens 10000, total prompts 100

# GPU 4: 20 prompts received, generated 2000 tokens in 82.97140645980835 seconds, 24.0 t/s
# GPU 0: 20 prompts received, generated 2000 tokens in 83.48482155799866 seconds, 23.0 t/s
# GPU 2: 20 prompts received, generated 2000 tokens in 83.52051067352295 seconds, 23.0 t/s
# GPU 3: 20 prompts received, generated 2000 tokens in 83.74173140525818 seconds, 23.0 t/s
# GPU 1: 20 prompts received, generated 2000 tokens in 83.78473687171936 seconds, 23.0 t/s
# tokens/sec: 119.0, time 83.78653645515442, total tokens 10000, total prompts 100


