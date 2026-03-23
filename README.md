# Machine-Learning-Research
Collection of some of the machine learning research I have done, primarily during my undergrad research

## What each of the files contain
"early_custom_models" contains several custom PyTorch models that I made when I was first learning ML and PyTorch, and following a lot of ablation testing I was able to create the "Chaos Unit", which given certain parameters has chaotic dynamics.

"newer_custom_models" contains later custom PyTorch models as I gained more experience and developed my research. As well as running Mamba and S5 in PyTorch to get my own baselines for LRA, I created models meant to mimick the behaviour of S5/Mamba while restricting them to the form of a basic RNN. I then simulated the MurmurHash3 algorithm as an RNN in similar fashion, with the idea that just as the Hash encapsulates the entire input when generating an output, perhaps the RNN can encapsulate the entire input and so handle long sequences of inputs more effectively.
I included "example_lr_train_pipeline" to show some of the pipeline I developed for running these models on the LRA (Long Range Arena) dataset. I could not share everything, most notably my output handling and custom config files, but I added this simply to demonstrate that I have experience handling the whole ML process and not just model development.

"drug_resistant_clustering_pipeline" contains an unsupervised pipeline for learning about drug resistant bacteria. Unfortunately this research is still ongoing and I cannot share much of it, but this was included as my first attempt at creating a pipeline for this.
