#Online CCA Todo List

##Experiments

###Random
* Figure out why your objective is bouncing all over for the experiment you tried with non-zero column means.

###E4
* Clean up the data to get more 'regular' parts of the time series.

##Plots
* Simple line plots of the different data streams. Just time step and value.

* Plots of the canonical bases under certain arrangements of tuning parameters and optional add-ons.

##Implementation

###Features
* Histograms for fixed time intervals. Add a data transformer for this.

###Corrections and Regularization
* Try the cubic regularization technique maybe.

###Numerical Linear Algebra
* See if block-diagonal structure can be leveraged in CCALin (and GenELinK) implementation(s).

##Model Serialization
* Need to implement save and load functions for the CCA objects.

##Collaboration

##New Ideas
* Try generalizing the current formulation to a broader class of dependency structure than just the pinwheel. How does this relate back to the auxiliary variable you used in the DD work? Can I use ADMM?
