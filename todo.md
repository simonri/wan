1. make sure to autocast if dtype != fp32
for example in text encoder

2. lora loading doesnt change the graph. that means we can
compile the graph even if we need to change loras later.
note: the lora rank needs to stay the same (32).