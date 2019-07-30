def modify_data(filename, newfilename):
    f = open(filename, mode='r', encoding='utf8')
    f1 = open(newfilename, mode='w', encoding='utf8')
    lines = f.readlines()
    for i in range(len(lines)):
        if 'B-PER' in lines[i]:
            tokens = lines[i].strip().split('\t')
            for j in range(1, min(4, len(lines) - i)):
                if 'I-PER' in lines[i+j]:
                    tokens2 = lines[i+j].strip().split('\t')
                    tokens[0] = tokens[0] + '_' + tokens2[0]
            lines[i] = tokens[0] + '\t' + tokens[1] +'\t' + tokens[2] + '\t' + tokens[3] + '\n'
        
        if 'I-PER' not in lines[i]:
            f1.write(lines[i])
    f1.close()
    f.close()

modify_data('train_preprocess.txt', 'train_preprocess_name.txt')
modify_data('test_preprocess.txt', 'test_preprocess_name.txt')
