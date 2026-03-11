# Notes from Run 4.
Run 4 is our longest running GRPO training. Since training is very long and the fact that we will run into problems, we made this document to note some of the things we do during training which might have an effect on the training run.

1. Training stopped on step ~ 700, with out latest checkpoint being at step 500.
    * This cause issues with the graph, since we could not override the old logged steps, the two graphs are overlapping between step 500-700
    * We did not have the training state saved at this point, this was implemented, and we created a manual training state with global step = 500. We could not save the optim state. 
2. Training was stopped again at step ~ 1300. This time we stopped right at a good evaluation where we got a checkpoint for a new best result.
    * However, we had only implemented training state checkpoint for the save frequencies, and not for the best eval. So we lost training state again, and had to manually create a training state with global step 1300, but no optim again.
3. Started training again now, but at 764 max response length. Because our response lengths have stabilized more, and we noticed that some valid think traces were cut off.
    * Also, started seeing quite a few full sweeps, where the model gets 1.0 rewards across all completions, all the advantages will be 0, so nothing to learn. We filtered them out from doing the _training_step. Should make it a bit faster.
    