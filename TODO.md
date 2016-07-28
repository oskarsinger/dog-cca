#Online CCA Todo List

##Presentation
* Introduce notation for optimization problem formulation, pseudocode, etc.

* Think of ways to clarify what exactly the plots are showing (e.g. clarify that the filtering plots are showing X times Phi).

##Experiments
* Run on some synthetic data to figure out why the AppGrad filter takes so long to stabilize.

##Plots
* Plot the application of the basis for each of the periods in the periodic metalearner to the full dataset.

* Zoom in on the filtering plots to make the subtleties and periodicity more apparent.

##Implementation

###Gram Variations
* FOR CCALin ONLY: is it problematic that my cross-Gram matrices are not stateful like the Gram matrices? Maybe I should implement the algorithms to run the stateful updates on the cross Grams too. Should brainstorm with Sijia or John about this at some point.

###Feature Representations

###Optimization
* Try the cubic regularization technique from Rong Ge's paper maybe.

###Numerical Linear Algebra
* See if block-diagonal structure can be leveraged in CCALin (and GenELinK) implementation(s).

##Model Serialization
* Need to implement save and load functions for the CCA objects.

##Collaboration

##New Ideas
* Try generalizing the current formulation to a broader class of dependency structure than just the pinwheel. How does this relate back to the auxiliary variable you used in the DD work? Can I use ADMM?
