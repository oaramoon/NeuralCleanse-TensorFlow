# NeuralCleanse-TensorFlow
Recreating the results of paper titled as "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks"

The uploaded model 'badnet_model.hdf5' for GTSRB dataset has a backdoor from all class to the "Stop Sign" class (class id =1), and the corresponding backdoor trigger is a yellow sticker on the traffic signs. 

As you can see in the diagram below,
1. The reverse engineered trigger for the backdoored class has the smallest L1 norm compared to all the other extracted triggers. 
2. Neural Cleanse could succesfully recover the original trigger.

<img src="https://github.com/oaramoon/NeuralCleanse-TensorFlow/blob/main/all_triggers.png" alt="MarineGEO circle logo" style="height: 1000px; width:1000px;"/>
