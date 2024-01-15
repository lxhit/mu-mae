def max_accuracy_from_dict(dataset,accuracy_dict):
    max_dict = {}
    max_accuracy = 0.0
    max_confidence = 0.0
    best_iteration = 6400
    for key,value in accuracy_dict.items():
        accuracy_temp = value[dataset]['accuracy']
        if accuracy_temp > max_accuracy:
            max_accuracy = accuracy_temp
            max_confidence = value[dataset]['confidence']
            best_iteration = key
    max_dict['best_accuracy'] = max_accuracy
    max_dict['confidence'] = max_confidence
    max_dict['best_iteration'] = best_iteration
    return max_dict



