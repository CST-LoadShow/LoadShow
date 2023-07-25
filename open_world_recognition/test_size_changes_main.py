from open_world_recognition.random_forest import modelTrain

if __name__ == "__main__":

    rot8_label = ['unity', 'audition', 'potplayer', 'cloudmusic', 'zoom', 'AliyunNetdisk', 'bandizip', 'lol', 'sunlogin']
    train_size = [5, 10, 15]
    test_size_time = [3, 6, 9, 12, 15]
    global_precision = []
    global_fpr = []
    for test_size in test_size_time:
        print(f'==================== test size set = {test_size} ====================')
        FPR_list, TPR_list = [], []
        precision_recall_f1_res = []
        FPR_TPR = []
        fpr_list =[]
        for v in range(len(train_size)):
            # print("==================")
            # print('train_size', train_size[v])
            for u in range(len(rot8_label)):
                # print('====================')
                # print('program ', rot8_label[u])
                dataset_0 = "./file/feature_%s_%d_0.csv" % (rot8_label[u], train_size[v])
                dataset_1 = "./file/feature_%s_%d_1.csv" % (rot8_label[u], train_size[v])
                test_dataset = './file/test_%s.csv' % rot8_label[u]
                model_file = "./file/feature_%s_%d.pickle" % (rot8_label[u], train_size[v])
                pic_save = 'predict_val_feature_%s_%d.png' % (rot8_label[u], train_size[v])

                massage, temp_list = modelTrain(dataset_0, dataset_1, test_dataset, test_size_time=test_size)
                FPR_TPR.append(temp_list[:2])
                precision_recall_f1_res.append(temp_list[2:5])


            def avg(data: list):
                return sum(data) / len(data)

        precision, recall, f1 = (list(map(list, zip(*precision_recall_f1_res))))
        fpr, tpr = (list(map(list, zip(*FPR_TPR))))

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

        fpr_mean = []
        tpr_mean = []
        label_len = len(rot8_label)
        arg_num = len(train_size)

        for i in range(0, label_len * arg_num, label_len):
            precision_avg.append(avg(precision[i:i + label_len]))
            precision_std.append(std_calculator(precision[i:i + label_len]))
            recall_avg.append(avg(recall[i:i + label_len]))
            recall_std.append(std_calculator(recall[i:i + label_len]))
            f1_avg.append(avg(f1[i:i + label_len]))
            f1_std.append(std_calculator(f1[i:i + label_len]))
            fpr_mean.append(avg(fpr[i:i + label_len]))
            tpr_mean.append(avg(tpr[i:i + label_len]))

        print("-----------precision-------------")
        print([("%.5f" % n) for n in precision_avg])
        print("---------- fpr -------------")
        print([("%.5f" % n) for n in fpr_mean])

        global_precision.append(precision_avg)
        global_fpr.append(fpr_mean)
