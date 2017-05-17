#Online CCA Todo List

##Engineering
* Reimplement old algorithm (or some variation/extension of it?) with current refactoring of my research code
* Use Adam
* Implement the paper that Al sent me.

##Algorithms
* Think about how to formalize the active learning component. What do I want to accomplish? Probably sensor fusion, so something like a CCA graph across a bunch of sensors.
    * Would it be helpful to put a probabilistic model on it and turn it into something like a Kalman filter?
    * Is there any way I can formulate it as a regret minimization problem? I have a loss function, and there's an optimal loss in hind-sight, so yeah. Do I really need to do SGD, or is there a better way? Think decentralized Coh.Lin.
* Is there any way I can use some tricks from Matrix Cookbook to decrease the operation count?
* Can I directly address the GEP formulation with SGD or an MDP instead of using the CCALin thing?
* Try doing the exponential increase aggregation schedule with the Matrix Cookbook linear system solver updates for genelink/ccalin. First, I will have to derive a decentralized genelink/ccalin. Convergence on that might be kinda tricky.

##Writing
* Go back and read and revise the old write-up.
* Start a conference-style write-up with an outline so I can bring it to Al. This is probably the first thing I should do.
* Probably go look at other papers submitted to IEEE Big Data for examples of how to structure the paper.

##Experiments
* Is this even comparable to the paper that Al sent me? Are we trying to do the same thing? Seems like they might just be trying to scale up.