import pandas as pd

def load_label_output(filename):
    data1 = pd.read_csv(filename, sep=',', header=None,
                           names=['text_id','message','target', 'startend','polarity' ],dtype=str)
    return data1['text_id'].tolist(),data1['polarity'].tolist()
def eval_list(glabels, slabels):
    if (len(glabels) != len(slabels)):
        print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
        len(slabels), len(glabels)))
    n = min(len(slabels), len(glabels))
    incorrect_count = 0
    for i in range(0, n):
        if slabels[i]=='-':
            slabels[i]='-1'
    for i in range(0, n):
        if slabels[i] != glabels[i]: incorrect_count += 1
    acc = (n - incorrect_count) / n
    print("\nACCURACY: %.2f" % (acc * 100))


