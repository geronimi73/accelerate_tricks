# note: doing it this way is stupid. https://github.com/huggingface/accelerate/issues/2029#issuecomment-1832339828

from accelerate import Accelerator
from accelerate.utils import gather_object, tqdm
import time
import random

accelerator = Accelerator()

data=[ random.randint(0,100) for _ in range(100) ]
pbar=tqdm(total=len(data))    

with accelerator.split_between_processes(data) as batch:
	for sample in batch:
		# do something with data
		time.sleep(0.1)

		accelerator.wait_for_everyone()

		pbar.update( accelerator.num_processes )