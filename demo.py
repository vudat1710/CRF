import sys
import pickle
from CRF import get_features
from pyvi import ViTokenizer

def parse_raw_input(input):
    sentence = []
    s = []
    for sent in input.split('.'):
        for token in sent.strip().split():
            s.append((token.replace("_"," "), ))
        s.append(('.', '_'))
        sentence.append(s)
        s = []
    last_sent = sentence[-1]
    last_sent.remove(last_sent[-1])
    return sentence

def main():
    loaded_model = pickle.load(open('finalized_model_no_pos_chunk_name_process.pkl', 'rb'))
    result = {}
    text = input('Enter some text: \n\n')
    tokenized = ViTokenizer.tokenize(text)
    raw_text = parse_raw_input(tokenized)
    word_featured = [get_features(s) for s in raw_text]
    preds = loaded_model.predict(word_featured)
    temp_sent_list = tokenized.split('.')
    sent_list = []
    for i in range(len(temp_sent_list)):
        
        if len(temp_sent_list[i]) > 0:
            sent_list.append(temp_sent_list[i].strip())
    print("\n\nResult : \n")
    for i in range(len(sent_list)):
        result = []
        current_sent = sent_list[i]
        current_tag = preds[i]
        tokens = current_sent.split(' ')
        if len(current_tag)>len(tokens):
            tokens.append('.')
        if len(current_sent) > 0:
            for j in range(len(tokens)):
                result.append([tokens[j],current_tag[j]])
        print(str(i) + " : ", end = " ")
        for part in result:
            if part[1] == "O":
                print(part[0], end=" ")
            
            else:
                # print("<"+part[1]+">" + part[0] +"</"+part[1]+">", end = " ")
                print(part[0] +"/"+part[1], end = " ")
        print("\n")

if __name__ == "__main__":
    main()

