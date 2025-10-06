"""
Below are thoughts about this
1. data pre proc
    - keep 2 varation : 1 , current version; 2,dummy category variable
    no matter what is, the target shall be transform to log once right skewed is confirmed
2. model: simple OLS, Lasso
3. Cross validation for performance evaluation

so:
data v1 + Simple OLS ;
data v1 + LASSO
data v2 + simple OLS
data v2 + LASSO
"""