#Online CCA Todo List

##Experiments

###Random
* Figure out why your objective is bouncing all over for the experiment you tried with non-zero column means.

###E4
* Clean up the data to get more 'regular' parts of the time series.

##Implementation

###Step Size
* Generalize my code to allow for an arbitrary step-size scheduler.
    * What kind of arguments will they need to take?
* Implement that one from Yann LeCun's student.
* Implement the probabilistic one.
* Implement the one that was mentioned in the optimization-online update.

###Corrections and Regularization
* Try the cubic regularization technique maybe.

###Numerical Linear Algebra
* See if block-diagonal structure can be leveraged in CCALin (and GenELinK) implementation(s).

###Misc.
* Allow for averaging across windows of coordinates in the minibatch server.

##Collaboration

##New Ideas
* Try generalizing the current formulation to a broader class of dependency structure than just the pinwheel. How does this relate back to the auxiliary variable you used in the DD work? Can I use ADMM?
