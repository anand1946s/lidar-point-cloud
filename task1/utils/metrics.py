def extract_metrics(result):

    return {
        "fitness": result.fitness,
        "rmse": result.inlier_rmse
    }