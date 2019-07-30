from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
import sklearn
import sklearn_crfsuite
import eli5
import pickle
from sklearn.model_selection import RandomizedSearchCV
import scipy
from sklearn.metrics import make_scorer

def parse_file2(filename):
    raw_text = open(filename).read().split('\n')
    sents = []
    for line in raw_text:
        tokens = line.strip().split()
        if line.strip():
            if len(tokens) == 3:
                sents.append(tokens)
    return sents

def parse_file(filename):     
    raw_text = open(filename, mode='r', encoding='utf8').readlines()
    sents = []
    s = []
    for line in raw_text:
        tokens = line.strip().split('\t')
        if line.strip():
            if len(tokens) == 4:
                token = tokens[0]
                pos = tokens[1]
                chunk = tokens[2]
                label = tokens[3]
                s.append((token, pos, chunk, label))
        else:
            sents.append(s)
            s = []
    return sents

def isName(word):
    tokens = word.split('_')
    for i in range(len(tokens)):
        if tokens[i].islower():
            return False
    return True

def isMixCase(word):
    if len(word) > 2:
        if word[0].islower() and word[1].istitle():
            return True
        return False
    return False

def wordShape(word): 
    shape = ''
    for character in word:
        if character.istitle():
            shape += 'U' 
        elif character.islower():
            shape += 'L'
        elif character.isdigit():
            shape += 'D'
        else:
            shape += character
    return shape

train_sent = parse_file('train_preprocess.txt')
test_sent = parse_file('test_preprocess.txt')
# dev_sent = parse_file('dev.txt')

def word2feature(sent, i):
    word = sent[i][0]
    # pos = sent[i][1]
    # chunk = sent[i][2]
    features = {
        #define the current word, its prefixes and suffixes
        'w(0)': word,                       
        'w(0)[:1]': word[:1],
        'w(0)[:2]': word[:2],
        'w(0)[:3]': word[:3],
        'w(0)[:4]': word[:4],
        'w(0)[-1:]': word[-1:],     
        'w(0)[-2:]': word[-2:],
        'w(0)[-3:]': word[-3:],
        'w(0)[-4:]': word[-4:],

        #morphological features and semantic features
        # 'pos(0)': pos,
        # 'chunk(0)': chunk,
        'word.islower': word.islower(),
        'word.lower': word.lower(),
        'isTitle': word[0].istitle(),
        'isNumber': word.isdigit(),
        'isUpper': word.isupper(),
        'isCapWithPeriod': word[0].istitle() and word[-1] == '.',
        'endsInDigit': word[-1].isdigit(),
        'containHyphen': '-' in word,
        'isDate': word[0].isdigit() and word[-1].isdigit() and '/' in word,
        'isCode': word[0].isdigit() and word[-1].istitle(),
        'isName': isName(word),
        'isMixCase': isMixCase(word),
        'd&comma': word[0].isdigit() and word[-1].isdigit() and ',' in word,
        'd&period': word[0].isdigit() and word[-1].isdigit() and '.' in word,
        'wordShape': wordShape(word)
    }

    if(i > 0):
        prev_word = sent[i-1][0]
        # prev_pos = sent[i-1][1]
        # prev_chunk = sent[i-1][2]
        features.update({
            'w(-1)': prev_word,
            # 'pos(-1)': prev_pos,
            # 'chunk(-1)': prev_chunk,
            # 'pos(-1)+pos(0)': prev_pos + ' ' + pos,
            # 'chunk(-1)+chunk(0)': prev_chunk + ' ' + chunk,
            'w(-1).lower': prev_word.lower(),
            'isTitle(-1)': prev_word[0].istitle(),
            'isNumber(-1)': prev_word.isdigit(),
            'isCapWithPeriod(-1)': prev_word[0].istitle() and prev_word[-1] == '.',
            'isName(-1)': isName(prev_word),
            'wordShape(-1)': wordShape(prev_word),
            'w(-1)+w(0)': prev_word + ' ' + word
        })
    else:
        features['BOS'] = True 

    if i > 1:
        prev_2_word = sent[i-2][0]
        # prev_2_pos = sent[i-2][1]
        # prev_2_chunk = sent[i-2][2]
        features.update({
            'w(-2)': prev_2_word,
            # 'pos(-2)': prev_2_pos,
            # 'chunk(-2)': prev_2_chunk,
            'w(-2)+w(-1)': prev_2_word + ' ' + prev_word,
            # 'pos(-2)+pos(-1)': prev_2_pos + ' ' + prev_pos,
            # 'chunk(-2)+chunk(-1)': prev_2_chunk + ' ' + prev_chunk
        })

    if i < (len(sent) - 1):
        next_word = sent[i+1][0]
        # next_pos = sent[i+1][1]
        # next_chunk = sent[i+1][2]
        features.update({
            'w(1)': next_word,
            # 'pos(1)': next_pos,
            # 'chunk(1)': next_chunk,
            # 'pos(0)+pos(1)': pos + ' ' + next_pos,
            # 'chunk(0)+chunk(1)': chunk + ' ' + next_chunk,
            'w(1).lower': next_word.lower(),
            'isTitle(1)': next_word[0].istitle(),
            'isNumber(1)': next_word.isdigit(),
            'isCapWithPeriod(1)': next_word[0].istitle() and next_word[-1] == '.',
            'isName(1)': isName(next_word),
            'wordShape(1)': wordShape(next_word),
            'w(0)+w(1)': word + ' ' + next_word
        })
    else:
        features['EOS'] = True 

    if i < (len(sent)-2):
        next_2_word = sent[i+2][0]
        # next_2_pos = sent[i+2][1]
        # next_2_chunk = sent[i+2][2]
        features.update({
            'w(2)': next_2_word,
            # 'pos(2)': next_2_pos,
            # 'chunk(2)': next_2_chunk,
            'w(1)+w(2)': next_word + ' ' + next_2_word,
            # 'pos(1)+pos(2)': next_pos + ' ' + next_2_pos,
            # 'chunk(1)+chunk(2)': next_chunk+' '+next_2_chunk,
            'w(2).isTitle()': next_2_word[0].istitle(),
            'w(2).isdigit': next_2_word[0].isdigit(),
        })

    return features

def get_features(sent):
    return [word2feature(sent, i) for i in range(len(sent))]

def get_labels(sent):
    return [label for token, _, _, label in sent]

def get_tokens(sent):
    return [token for token, _, _, label in sent]

X_train = [get_features(s) for s in train_sent]
y_train = [get_labels(s) for s in train_sent]
X_test = [get_features(s) for s in test_sent]
y_test = [get_labels(s) for s in test_sent]


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', # you can also use l2sgd algorithm but time training is slower than lbfgs
    c1=0.026,
    c2=0.037,
    max_iterations=100,
    all_possible_transitions=True
)

# crf.fit(X_train, y_train)
# pickle.dump(crf.classes_, open('labels.pkl', 'wb'))
labels = pickle.load(open('labels.pkl', 'rb'))
labels.remove('O')
print(labels)
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

# params_space = {
#     'c1': scipy.stats.expon(scale = 0.3),
#     'c2': scipy.stats.expon(scale = 0.3)
# }

# f1_score = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

# rs = RandomizedSearchCV(crf, params_space,
#                         cv = 3,
#                         verbose = 1,
#                         n_jobs = -1,
#                         n_iter = 50,
#                         scoring = f1_score
# )

# rs.fit(X_train, y_train)

# print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)
# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))



def main():
    crf.fit(X_train, y_train)

    filename = 'finalized_model.pkl'           
    pickle.dump(crf, open(filename, 'wb'))
    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels))
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    with open('weight.html', 'wb') as weight:  
        pickle.dump(eli5.show_weights(crf, top=100), weight)

if __name__ == '__main__':
    main()