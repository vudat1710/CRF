
def process(filename, newfilename):
    f = open(filename, mode='r', encoding='utf8')
    new_file = open(newfilename, mode='w', encoding='utf8')
    lines = f.readlines()

    for line in lines:
        new_file.write(line.replace(' ', '_'))
    f.close()
    new_file.close()


process('test.txt', 'test_preprocess.txt')
process('train.txt', 'train_preprocess.txt')