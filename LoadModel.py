import CRF
import pickle
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from collections import Counter
from sklearn_crfsuite import metrics

loaded_model = pickle.load(open('finalized_model_raw_text.pkl', 'rb'))

labels = loaded_model.classes_
labels.remove('O') #remove O tags- O tags make no differences in the f1 score result

y_pred1 = loaded_model.predict(CRF.X_test)

# get output file from test set with predict tags appended
fo = open('output.txt', mode='w', encoding="utf8")
for i, sent in enumerate(CRF.test_sent):
    for j, ner in enumerate(sent):
        fo.write(CRF.test_sent[i][j][0]+'\t'+CRF.test_sent[i][j][1]
            +'\t'+CRF.test_sent[i][j][3]+'\t'+y_pred1[i][j])
        fo.write('\n')
    fo.write('\n')
fo.close()

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    CRF.y_test, y_pred1, labels=sorted_labels, digits=4
))

# Top transition
def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


print("Top likely transitions:")
print_transitions(Counter(loaded_model.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(loaded_model.transition_features_).most_common()[-20:])


# Top positive
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))
print("Top positive:")
print_state_features(Counter(loaded_model.state_features_).most_common(30))
print("\nTop negative:")
print_state_features(Counter(loaded_model.state_features_).most_common()[-30:])
