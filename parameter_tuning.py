'''
This module is for tuning parameters of the model, using randomized search and multi-thread programming
'''


import threading
from random import sample
from config import Config
from main import model_training
import json


thread_num = 1
test_size = 1  # the number of parameters set test in each thread

# parameters
parameter_candidates_dict = {
    "inner_learning_rate": [0.1, 0.01, 0.05, 0.2, 0.09, 0.03, 0.001, 0.005],
    "learning_rate": [0.003, 0.005, 0.001, 0.002],
    "dropout": [0.1, 0.3, 0.5, 0.2, 0.4, 0.25, 0.35]
}

total_results = [None] * thread_num

# create txt files for records
for i in range(thread_num):
    open("thread_{}.txt".format(i), "w+")

def train_with_params(thread_id):
    results_in_threads = []

    for j in range(test_size):
        single_set_param = {}

        # randomly choose parameters
        for key, values in parameter_candidates_dict.items():
            sampled_value = sample(values, 1)[0]  # [0] for getting the single value from list
            single_set_param.update({key: sampled_value})

        config = Config(single_set_param)
        returned_values = model_training(config)  # model_training returns "model, (mean_rank, hits10)"

        mean_rank, hits10 = returned_values[1]
        single_set_param.update({"mean_rank": mean_rank, "hits@10": hits10})
        with open("thread_{}.txt".format(thread_id), "a+") as f:
            f.write(json.dumps(single_set_param))
            f.write("\n")

        results_in_threads.append(single_set_param)

    total_results[thread_id] = results_in_threads


if __name__ == "__main__":
    threads = []

    for i in range(thread_num):
        new_thread = threading.Thread(target=train_with_params, args=(i,))
        threads.append(new_thread)
        new_thread.start()

    # wait for all threads to complete
    for i in range(thread_num):
        threads[i].join()

    total_results = [single_result for thread_results in total_results for single_result in thread_results]
    total_results.sort(key=lambda x: x["mean_rank"], reverse=True)  # sort the list based on mean_rank

    with open("total_result.txt", "w+") as f:
        for result in total_results:
            f.write(json.dumps(result))
            f.write("\n")
