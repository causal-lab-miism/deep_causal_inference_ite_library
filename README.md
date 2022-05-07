## Review of Deep Learning Methods for ITE Estimation with Automatic Hyperparameter Optimization

Tensorflow implementation of methods presented in:

Andrei Sirazitdinov, Marcus Buchwald, Jürgen Hesser, and Vincent Heuveline _"Review of Deep Learning Methods for Individual Treatment Effect Estimation with Automatic Hyperparameter Optimization"_, 2022. Submitted to IEEE Transactions on Neural Networks and Learning.

Paper link: arxive link?

Contact: andrei.sirazitdinov@medma.uni-heidelberg.de, marcus.buchwald@medma.uni-heidelberg.de

### Requirements:
1. Python 3.9, 
2. Tensorflow 2.8, 
3. Tensorflow Probability 0.16.0, 
4. Numpy

To run the code simply use:
```
python main.py --dataset_name [dataset name] --model_name [model name]
```
Alternatively use Jupyter notebooks.  

See the full list of available models and datasets in main.py.  

The file hyperparameters.py contains hyperparameters such as batch size or learning rate for the presented models.  

Note, that our code performs hyperparameter search at first execution for each method to find the other hyperparameters.

We output PEHE for each sub dataset and after training on all datasets an average PEHE with 95% confidence interval.
