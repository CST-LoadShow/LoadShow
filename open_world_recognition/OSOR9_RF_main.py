from open_world_recognition.random_forest import modelTrain

if __name__ == "__main__":

    OSOR9_label = ['unity', 'audition', 'potplayer', 'cloudmusic', 'zoom', 'AliyunNetdisk', 'bandizip', 'lol',
                   'sunlogin']
    train_size = [5, 10, 15]
    FPR_list, TPR_list = [], []
    precision_list, recall_list = [], []
    weighted_precision_list, weighted_recall_list = [], []
    roc_res = []
    precision_recall_f1_res = []
    weighted_precision_recall_f1_res = []
    all_roc = []
    roc_list = []
    FPR_TPR = []
    fpr_list = []
    for v in range(len(train_size)):
        print("==================")
        print('train_size', train_size[v])
        for u in range(len(OSOR9_label)):
            print('====================')
            print('program ', OSOR9_label[u])
            dataset_0 = "./file/feature_%s_%d_0.csv" % (OSOR9_label[u], train_size[v])
            dataset_1 = "./file/feature_%s_%d_1.csv" % (OSOR9_label[u], train_size[v])
            test_dataset = './file/test_%s.csv' % OSOR9_label[u]
            massage, temp_list = modelTrain(dataset_0, dataset_1, test_dataset)
            precision_recall_f1_res.append(temp_list[2:5])
        avg_roc = []

    precision, recall, f1 = (list(map(list, zip(*precision_recall_f1_res))))

    def avg(data: list):
        return sum(data) / len(data)


    def std_calculator(data: list):
        a = avg(data)
        res = 0
        for d in data:
            res += (d - a) ** 2
        return (res / len(data)) ** 0.5


    precision_std = []
    precision_avg = []
    recall_std = []
    recall_avg = []
    f1_avg = []
    f1_std = []

    label_len = len(OSOR9_label)
    arg_num = len(train_size)
    for i in range(0, label_len * arg_num, label_len):
        precision_avg.append(avg(precision[i:i + label_len]))
        precision_std.append(std_calculator(precision[i:i + label_len]))
        recall_avg.append(avg(recall[i:i + label_len]))
        recall_std.append(std_calculator(recall[i:i + label_len]))
        f1_avg.append(avg(f1[i:i + label_len]))
        f1_std.append(std_calculator(f1[i:i + label_len]))

    print("-----------precision-------------")
    print([("%.3f" % (100 * n)) for n in precision_avg])
    print([("%.3f" % (100 * n)) for n in precision_std])
    print("-----------recall-------------")
    print([("%.3f" % (100 * n)) for n in recall_avg])
    print([("%.3f" % (100 * n)) for n in recall_std])
    print("-----------f1-------------")
    print([("%.3f" % (100 * n)) for n in f1_avg])
    print([("%.3f" % (100 * n)) for n in f1_std])
