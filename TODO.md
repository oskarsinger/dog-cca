#Online CCA Todo List

##Experiments

###Random
* Figure out why your objective is bouncing all over for the experiment you tried with non-zero column means.

###E4
* Clean up the data to get more 'regular' parts of the time series.
* Figure out how to use the HD5 file type and the Python library.

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
* Allow maximum number of iterations for gradient-based algorithms.

##Collaboration
* Mention your block diagonal observation to Sijian and Brandon.
* Ask Brandon if he's looked at your codebase yet.
