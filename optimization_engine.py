import numpy as np
import random
from deap import base, creator, tools, algorithms

if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0)) 
    creator.create("Individual", list, fitness=creator.FitnessMulti)

def run_rebalancing(df, total_equity, max_risk, min_yield, target_years, max_concentration):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(df))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        weights = np.array(individual)
        weights = np.maximum(weights, 0)
        if weights.sum() == 0: return -999, 999, 999
        weights /= weights.sum()
        
        port_yield = np.dot(weights, df["Yield"])
        port_risk = np.dot(weights, df["Risk_Vol"])
        port_dur = np.dot(weights, df["Duration"])
        cost = np.dot(weights, df["Spread_bps"]) / 10000
        port_sentiment = np.dot(weights, df["Sentiment"])
        
        alpha = 0.05 
        net_yield = port_yield - cost + (port_sentiment * alpha)
        
        penalty = 0
        if port_risk > max_risk: penalty += 0.5
        if port_yield < min_yield: penalty += 0.2
        if np.max(weights) > max_concentration: penalty += 0.5
        if port_sentiment < 0: penalty += abs(port_sentiment) 
        
        dur_gap = abs(port_dur - target_years)
        return (net_yield - penalty), (port_risk + penalty), (dur_gap + penalty)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=60)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=60, lambda_=120, cxpb=0.7, mutpb=0.2, ngen=50, verbose=False)
    
    best_ind = tools.selBest(pop, k=1)[0]
    weights = np.maximum(np.array(best_ind), 0)
    weights /= weights.sum()
    
    df_res = df.copy()
    df_res["Target_Weight"] = weights
    df_res["Target_Value"] = df_res["Target_Weight"] * total_equity
    df_res["Target_Shares"] = np.floor(df_res["Target_Value"] / df_res["Price"])
    return df_res