from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

def func(x,y):
    res = x/y
    res2 = res*500 - res

    return -1*(1024000-res2)

def find_params():

    pbounds = {
        "x" : (10000,100000),
        "y" : (10, 1000)
    }

    optimizer = BayesianOptimization(
        f = func,
        pbounds=pbounds,
        random_state=1,
        verbose=2
        )

    logger = JSONLogger(path="log/logsDQN.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=30,
        n_iter=5000,
    )


    print("Best hyperparameters found were: ", optimizer.max)

find_params()
