#Online CCA Todo List

##Presentation
* Introduce notation for optimization problem formulation, pseudocode, etc.

* Think of ways to clarify what exactly the plots are showing (e.g. clarify that the filtering plots are showing X times Phi).

* Think of best way to juxtapose filter with random noise input to filter with E4 input to emphasize patterns in the E4.

##Experiments
* Run some experiments to figure out why AppGrad filter takes so long to stabilize.
    * Use synthesized data for sanity check on an easy case.
    * Try different step sizes. This will take some software engineering, so get the easier one out of the way first.

* Run some experiments on the new periodic data loader you created.

##Plots
* Plot the application of the basis for each of the periods in the periodic metalearner to the full dataset.

##Implementation

###Metalearners
* Make it possible to pass more args to the submodels in the periodic metalearners.
    * Maybe need to abstract away the call to 'fit' so that I can arbitrarily pass arguments to it. Should pass in a 'trainer' method?

###Gram Variations
* FOR CCALin ONLY: is it problematic that my cross-Gram matrices are not stateful like the Gram matrices? Maybe I should implement the algorithms to run the stateful updates on the cross Grams too. Should brainstorm with Sijia or John about this at some point.

###Numerical Linear Algebra
* See if block-diagonal structure can be leveraged in CCALin (and GenELinK) implementation(s). Judging from what I have read so far in the Matrix Computations book, it probably can.

##Model Serialization
* Need to implement save and load functions for the CCA objects.
