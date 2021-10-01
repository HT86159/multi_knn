import numpy as np
import operator
class pingjia():#
    def __init__(self,y,result1,result2,result3,n):
        self.y = y
        self.result1 = result1
        self.result2 = result2
        self.result3 = result3
        self.n = n
    def one_error(self):
        c = 0
        for i in range(len(self.y)):
            s1 = set(self.y[i])
            s2 = set([self.result2[i]])
            if len(list(s1&s2))>0:
                c+=1
        return (len(self.y)-c)/len(self.y)
    def hamming_loss(self):
        c = 0
        for i in range(len(self.y)):
            d1 = len(self.y[i])
            d2 = 0
            s1 = set(self.y[i])
            s2 = set(self.result1[i])
            c+=len(list(s1-s2))
        return c/(len(self.y))
    def coverage(self):
        cover_sum = 0
        for i in range(len(self.y)):
            label_outputs = []
            for label in self.y[i]:
                label_outputs.append(self.result3[i][label])
            #print(label_outputs)
            #print(self.y[i])
            min_output = min(label_outputs)
            for j in range(self.n):
                #print(self.n)
                if self.result3[i][j] >= min_output:
                    cover_sum += 1

        return ((cover_sum / (len(self.y))) - 1)/self.n
    def ranking_loss(self):
        rloss_sum = 0
        ap_sum = 0
        for sample_index in range(len(self.y)):#对每一条数据进行遍历
            unodered_part = []
            expected_num = len(self.y[sample_index])

            sample_output = self.result3[sample_index]
            output_dic = {}
            for output_index in range(self.n):#建立映射字典
                output_dic[output_index] = sample_output[output_index]

            sorted_output = sorted(output_dic.items(), key=operator.itemgetter(1), reverse=True)#按照第二个关键字进行排序

            temp_count = 0
            times = 0
            for sorted_tuples in sorted_output:#遍历所有的标签
                if times == expected_num:#结束标志就是每个有的标签已经遍历结束
                    break

                if sorted_tuples[0] not in self.y[sample_index]:
                    temp_count += 1
                else:
                    unodered_part.append(temp_count)
                    temp_count = 0
                    times += 1

            if len(unodered_part) != expected_num:
                raise Exception("function error for RankingLoss")

            pairs_num = 0
            fraction_sum = 0
            fraction_divide = 0
            for cal_index in range(expected_num):
                pairs_num += unodered_part[cal_index] * (expected_num - cal_index)
                # prepare for calculating average precision
                fraction_divide += unodered_part[cal_index] + 1
                fraction_sum += (cal_index + 1) / fraction_divide

            rloss_sum += pairs_num / (expected_num * (self.n - expected_num))
            ap_sum += fraction_sum / expected_num

        self.ap = ap_sum / len(self.y)
        self.rl = rloss_sum / len(self.y)
        self.ap_prepared = True
        self.rl_prepared = True

        return self.rl
    def average_precision(self):
        # contained in the ranking_loss function to save running time
        self.ranking_loss()
        return self.ap


