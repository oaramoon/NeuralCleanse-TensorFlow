# NeuralCleanse-TensorFlow
Recreating the results of paper titled as "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks"

The uploaded model trained for GTSRB dataset has a backdoor from all class to the "Stop Sign" class (class id =1), and the corresponding backdoor trigger is a yellow sticker on the traffic signs. 

As you can see in the image below,
1. The trigger for the backldoored class has smaller L1 norm compared to all the othert extracted triggers. 
2. With Neural Cleanse we could succesfully recover the original trigger.

<img src="https://github.com/oaramoon/NeuralCleanse-TensorFlow/blob/main/all_triggers.png" alt="MarineGEO circle logo" style="height: 1000px; width:1000px;"/>
