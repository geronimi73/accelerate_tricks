from accelerate import Accelerator
from accelerate.utils import gather_object
import json

accelerator = Accelerator()

# each GPU creates a string
message=[{
	"text": f"Hello this is GPU {accelerator.process_index}",
	"num": 3
	}]

# collect the messages from all GPUs
messages=gather_object(message)

# output the messages only on the main process with accelerator.print()
accelerator.print(json.dumps(messages, indent=2))
