import pandas as pd
from collections import defaultdict
import random


class TuplesSampler:
    def __init__(self, origin_file_path, working_path):
        self.working_path = working_path
        self.original_data = pd.read_csv(origin_file_path, sep='\t')

        self.origin_data_size = self.original_data.shape[0]
        self.entities_with_degree = defaultdict(lambda: [0, 0])  # degree representation, 1st: out, 2nd: in
        self.relations = set()
        self.tuples = []
        self.log_frequency = round(self.origin_data_size / 1000)
        self.sample_tuples = []
        self.sample_size = 0
        self.train = []
        self.test = []
        self.validation = []
        self.relation_num = 0

    def extract_tuples(self):
        print("=========Start Processing=========")

        progress = 0

        for index, row in self.original_data.iterrows():  # row: e1 rel e2
            self.entities_with_degree[row[0]][0] += 1  # records out degree
            self.entities_with_degree[row[2]][1] += 1  # records in degree
            self.relations.add(row[1])
            self.tuples.append([row[0], row[1], row[2]])

            if (index + 1) % self.log_frequency == 0:
                print("Processing {}%".format(progress))
                progress += 0.1

        self.relation_num = len(self.relations)
        print("=========Finish Processing=========")

    def analyze_dataset(self, is_write_to_file=True):
        def write_analysis_to_file():
            with open(self.working_path+"dataset_information.txt", 'w+') as f:
                f.write("Entities number = {}\n".format(total_size))
                f.write("Relations number = {}\n".format(relation_num))
                f.write("Average in-degree = {}\n".format(in_degree_sum / total_size))
                f.write("Min in-degree = {}\n".format(in_degree_min))
                f.write("Max in-degree = {}\n".format(in_degree_max))
                f.write("Average out-degree = {}\n".format(out_degree_sum / total_size))
                f.write("Min out-degree = {}\n".format(out_degree_min))
                f.write("Max out-degree = {}\n".format(out_degree_max))
                f.write("Average degree = {}\n".format(degree_sum / total_size))
                f.write("Min degree = {}\n".format(degree_min))
                f.write("Max degree = {}\n".format(degree_max))

        total_size = self.entities_with_degree
        relation_num = self.relation_num
        # initialize
        in_degree_min = float('inf')
        in_degree_max = float('-inf')
        in_degree_sum = 0
        out_degree_min = float('inf')
        out_degree_max = float('-inf')
        out_degree_sum = 0
        degree_min = float('inf')
        degree_max = float('-inf')
        degree_sum = 0

        # iterate over entities_with_degree to records information
        for entity in self.entities_with_degree:
            if entity[1][1] > in_degree_max:
                in_degree_max = entity[1][1]
            if entity[1][1] < in_degree_min:
                in_degree_min = entity[1][1]
            in_degree_sum += entity[1][1]

            if entity[1][0] > out_degree_max:
                out_degree_max = entity[1][0]
            if entity[1][0] < out_degree_min:
                out_degree_min = entity[1][0]
            out_degree_sum += entity[1][0]

            degree = entity[1][0] + entity[1][1]
            if degree > degree_max:
                degree_max = degree
            if degree < degree_min:
                degree_min = degree
            degree_sum += degree

        if is_write_to_file:
            write_analysis_to_file()

    def sort_entities_with_degree(self):
        self.entities_with_degree = sorted(self.entities_with_degree.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)

    def get_ratio_sample(self, sample_ratio=0.5):
        print("==========Start Sampling==========")

        tuples_to_remove = set()
        target_data_size = self.origin_data_size * sample_ratio
        data_size = self.origin_data_size

        for entity_info in self.entities_with_degree:
            entity = entity_info[0]
            for t in self.tuples:
                # remove the entities from dataset
                if t[0] == entity or t[2] == entity:
                    data_size -= 1  # here I neglect the-same-tuple situation temporarily
                    tuples_to_remove.add(tuple(t))  # convert lists to tuples because lists are unhashable
            print("new data size = {}".format(data_size))

            if data_size <= target_data_size:
                break

        progress = 0
        for t in range(len(self.tuples)):
            if tuple(self.tuples[t]) not in tuples_to_remove:
                self.sample_tuples.append(tuples[t])
            if (t + 1) % self.log_frequency == 0:
                print(progress, '%')
                progress += 0.1
        self.sample_size = len(self.sample_tuples)
        print("==========Finish Sampling (new data size = {})==========".format(self.sample_size))

    def split_dataset_for_cross_validate(self, ratio=[10, 1, 1]):
        print("==========Start Splitting for cross validation==========")

        # list 0: train, list 1: test, list 2: validation
        random_sample_group = random.choices([0, 1, 2], weights=ratio, k=self.sample_size)

        for t, group in zip(self.sample_tuples, random_sample_group):
            if group == 0:
                self.train.append(t)
            elif group == 1:
                self.test.append(t)
            elif group == 2:
                self.validation.append(t)

        print("train size = {}\ntest size = {}\nvalidation size = {}".format(len(self.train), len(self.test),
                                                                             len(self.validation)))
        print("==========Finish Splitting for cross validation==========")

    def write_to_files(self):
        for list_name, dataset_list in zip(['train', 'test', 'valid'], [self.train, self.test, self.validation]):
            print("========Start Writing {}.txt========".format(list_name))

            with open(self.working_path+"{}.txt".format(list_name), 'w+') as f:
                for t in self.dataset_list:
                    f.write("{}\t{}\t{}\n".format(t[0], t[1], t[2]))
            print("========Finish Writing {}.txt========".format(list_name))

    def execute(self):
        self.extract_tuples()
        self.sort_entities_with_degree()
        self.analyze_dataset()
        self.get_ratio_sample()
        self.split_dataset_for_cross_validate()
        self.write_to_files()


if __name__ == '__main__':
    sampler = TuplesSampler("data/alicoco/AliCoCo_v0.2.csv", "data/alicoco/")
    sampler.execute()
