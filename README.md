---
<font face="ITC Berkeley Oldstyle" size="3">Final Project for the courses <i>Effective Programming Practices</i> and <i>Scientific Computing</i> | Winter 20/21, M.Sc. Economics, Bonn University | [Arbi Kodraj](https://github.com/ArbiKodraj) </font><br/>

# Function Approximation via Machine Learning Methods

[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="justify">
The notebook FunctionApproximation.ipynb contains my work for the final project for EPP and Scientific Computing. It deals with different approximation methods for functions that vary in their complexity. The core methods originate from Machine Learning. This project aims to convince the audience that modern Machine Learning methods are as good as conventional methods for approximating mathematical functions (for some functions even better). Therefore, the audience should consider using them for approximation-related problems.  
</p>

<p align="justify">
At this point, I would like to mention that I use both object-oriented and functional programming. This is because it was only in the context of this work that I acquired project-oriented coding on my own and was able to improve it through the book "Fluent Python" written by Ramalho (2015). Also, I apply the acquired knowledge from the Effective Programming Practices and Scientific Computing courses to some extent, which is why I think there is a significant difference in my coding quality. To show my learning effect, I still decided to include the worse implementations from the code point of view. This should only serve as an explanation for the different coding styles. 
</p>

The best way to access this notebook is by downloading it [here](https://github.com/ArbiKodraj/Final_Project) and open it locally via jupyter notebook. Alternatively, it can be viewed [here](https://github.com/ArbiKodraj/Final_Project/blob/master/FunctionApproximation.ipynb), online on GitHub.  

For my code's documentation, I used <a href="https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html">Google-style docstrings</a> and created a HTML file via Sphinx, which requires the <a href="https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html">napoleon extension</a>. Since the repository is private, I could neither access <a href="https://pages.github.com">GitHub pages</a> nor <a href="https://readthedocs.com/dashboard/">readthedocs</a> for hosting the HTML file. Therefore, the documentation has to be opened locally. For this and for replication purposes, I recommend cloning the repository after the git initialization as follows:

```
$ git clone https://github.com/ArbiKodraj/Final_Project.git
```

As soon as the repository has been localized, the corresponding file can be found in the following directory:

> <mark>./docs/build/html/index.html</mark>

After the project's release, I will set up a <a href="https://readthedocs.com/dashboard/">readthedocs</a> website and make the code documentation available here.

Furthermore, I have prepared tests for the last section using the <a href="https://docs.python.org/3/library/unittest.html#unittest
">Unittest library</a> involved in Python. I decided to use Unittest because it seems to be very usable for object-oriented programming. The tests can be found in the test folder. For the execution of the tests, I recommend using Pycharm or Visual Studio Code.

> **Course Instructor Effective Programming Practices** : [Hans-Martin Gaudecker](https://github.com/hmgaudecker)

> **Course Instructor Scientific Computing** : [Philipp Eisenhauer](https://github.com/peisenha)

## Reproducibility

In order to ensure full reproducibility, I have set up a [continous integration](https://github.com/ArbiKodraj/Final_Project/blob/master/material/travis-ci.png) environment using [Travis Ci](https://travis-ci.com) which can be checked here: [![Build Status](https://travis-ci.com/ArbiKodraj/Final_Project.svg?token=FjHb3G3wqwrNzub1KhJT&branch=master)](https://travis-ci.com/ArbiKodraj/Final_Project)


## Notebook's Structure

- **1. Introduction:** Introduces the paper's objective and structure
- **2. Motivation of Interpolation:** Briefly motivates the use of approximation/interpolation
- **3. Application of Interpolation:** Demonstrates two approximation strategies as convential interpolation tool
- **4. Neural Networks as Modern Machine Learning Method:** Illustrates the use of Neural Networks as an alternative tool for approximating functions
- **5. Further Machine Learning Methods:** Presents additional Machine Learning methods and their usefulness in approximating functions
- **6. Economical Application:** Uses discussed approximation tools to solve economical problems related to function approximation
- **7. Conclusion:** Summarizes the key insights and contrasts the illustrated approximation tools

[![Continuous Integration](https://github.com/ArbiKodraj/Final_Project_EPP/workflows/Continuous%20Integration/badge.svg)](https://github.com/ArbiKodraj/Final_Project/actions)

## References

- <b>Athey, S., 2018.</b> The impact of machine learning on economics. In The economics of artificial intelligence: An agenda (pp. 507-547). University of Chicago Press.

- <b>Brynjolfsson, E., Mitchell, T. and Rock, D., 2018, May.</b> What can machines learn, and what does it mean for occupations and the economy?. *In AEA Papers and Proceedings* (Vol. 108, pp. 43-47).

- <b>Chaboud, A.P., Chiquoine, B., Hjalmarsson, E. and Vega, C., 2014.</b> Rise of the machines: Algorithmic trading in the foreign exchange market. *The Journal of Finance,* 69(5), pp.2045-2084.

- <b>Cybenko, G., 1989.</b> Approximation by superpositions of a sigmoidal function. *Mathematics of control, signals and systems,* 2(4), pp.303-314.

- <b>Gasca, M. and Sauer, T., 2000.</b> Polynomial interpolation in several variables. *Advances in Computational Mathematics,* 12(4), pp.377-410.

- <b>Gasca, M. and Sauer, T., 2001.</b> On the history of multivariate polynomial interpolation. *In Numerical Analysis:* Historical Developments in the 20th Century (pp. 135-147). Elsevier.

- <b>Goodfellow, I., Bengio, Y., Courville, A. and Bengio, Y., 2016.</b>  Deep learning (Vol. 1, No. 2). Cambridge: MIT press.

- <b>Heaton, J.B., Polson, N.G. and Witte, J.H., 2016.</b> Deep learning in finance. arXiv preprint arXiv:1602.06561.

- <b>Hendershott, T., Jones, C.M. and Menkveld, A.J., 2011.</b> Does algorithmic trading improve liquidity?. *The Journal of finance,* 66(1), pp.1-33.

- <b>Hopfield, J.J., 1982.</b> Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the national academy of sciences,* 79(8), pp.2554-2558.

- <b>Ketkar, N. and Santana, E., 2017.</b> Deep learning with python (Vol. 1). *Berkeley,* CA: Apress.

- <b>Lin, H.W., Tegmark, M. and Rolnick, D., 2017.</b> Why does deep and cheap learning work so well?. *Journal of Statistical Physics,* 168(6), pp.1223-1247.

- <b>Miranda, M.J. and Fackler, P.L., 2004.</b> Applied computational economics and finance. *MIT press.*

- <b>Moocarme, M., Abdolahnejad M. and Bhagwat, R., 2020.</b> The Deep Learning with Keras Workshop: An Interactive Approach to Understanding Deep Learning with Keras, 2nd Edition. 

- <b>Murphy, K.P., 2012.</b> Machine learning: a probabilistic perspective. *MIT press.*

- <b>Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J., 2011.</b> Scikit-learn: Machine learning in Python. *the Journal of machine Learning research*, 12, pp.2825-2830.

- <b>Ramalho, L., 2015.</b> Fluent python: Clear, concise, and effective programming. " *O'Reilly Media, Inc.*".

- <b>Tegmark, M., 2017.</b> Life 3.0: Being human in the age of artificial intelligence. *Knopf.*

- <b>Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J. and van der Walt, S.J., 2020.</b> SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature methods,* 17(3), pp.261-272.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ArbiKodraj/Final_Project/blob/master/LICENSE)

